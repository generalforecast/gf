# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""sklearn-style estimators: `fit` creates a table, `predict` queries it.

They follow sklearn's estimator API, so `clone`, pipelines and cross-validation
work, using one local client shared by every estimator by default. A pandas or polars frame or an Arrow table with string,
boolean or datetime columns keeps them, with byte strings read as UTF-8 text: its table's columns take the frame's
kinds, as `generalforecast.columns.infer` reads them, and queries are frames like it. Each estimator holds one table: fitting again, `close`, or
the estimator going away drops it. A fitted estimator keeps the rows it was fit
on, so it pickles, as joblib's workers need, and makes its table again where it
is unpickled. They need the `sklearn` extra: `pip install generalforecast[sklearn]`.
"""

import contextlib
import threading
import uuid
import weakref
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import validate_data

from generalforecast._core import Strategy
from generalforecast.columns import infer
from generalforecast.native.client import Table
from generalforecast.native.local_client import LocalClient
from generalforecast.native.package import (
    admitted_spec,
    context_table,
    export_package,
    remote_result,
    target_name,
)
from generalforecast.packages import Package
from generalforecast.packages.validation import ensure
from generalforecast.tabular import to_arrow


class _TableClient(Protocol):
    @property
    def create_table(self) -> Callable[..., Table]: ...

    def drop_table(self, name: str) -> None: ...


_shared: dict[bool, LocalClient] = {}
_shared_lock = threading.Lock()


def _local(allow_downloads: bool = False) -> LocalClient:
    """The local client of estimators given none, one per download gate, on the process's engine."""
    with _shared_lock:
        client = _shared.get(allow_downloads)
        if client is None:
            client = _shared[allow_downloads] = LocalClient(
                allow_downloads=allow_downloads
            )
    return client


_loading: set[tuple[str, bool]] = set()
_loading_lock = threading.Lock()


def _load_soon(model: str, allow_downloads: bool) -> None:
    """Starts loading a model in the engine's workers on a thread, while the caller builds its data.

    A worker compiling the model takes seconds; `fit` waits only for what is
    left. A model that fails to load fails again, with its error, in `fit`.
    """
    with _loading_lock:
        if (model, allow_downloads) in _loading:
            return
        _loading.add((model, allow_downloads))

    def load() -> None:
        try:
            _local(allow_downloads)._model_ref(model)
        except Exception:  # noqa: BLE001
            with _loading_lock:
                _loading.discard((model, allow_downloads))

    threading.Thread(target=load, name="gf-load", daemon=True).start()


def _named(X: Any) -> bool:
    """Whether rows are a named frame; unnamed matrices use sklearn validation."""
    return any(
        hasattr(X, name) for name in ("collect_schema", "column_names", "columns")
    )


def _text(table: pa.Table) -> pa.Table:
    """Decodes byte-string columns, as sklearn's dataset fetchers give text, to strings."""
    for index, field in enumerate(table.schema):
        if pa.types.is_binary(field.type) or pa.types.is_large_binary(field.type):
            try:
                values = pc.cast(table.column(index), pa.string())
            except pa.ArrowInvalid:
                raise ValueError(
                    f"column {field.name!r} holds bytes that are not UTF-8 text"
                ) from None
            table = table.set_column(index, pa.field(field.name, pa.string()), values)
    return table


def _holds_text(X: Any) -> bool:
    array = np.asarray(X)
    if array.dtype.kind in "SU":
        return True
    return array.dtype.kind == "O" and any(
        isinstance(value, (str, bytes)) for value in array.ravel()
    )


def _holds_words(y: Any) -> bool:
    """Whether y holds text that doesn't read as numbers, as `"1.5"` does."""
    if not _holds_text(y):
        return False
    try:
        np.asarray(y, dtype=np.float64)
    except (TypeError, ValueError):
        return True
    return False


def _labels(y: Any) -> Any:
    """Labels with byte strings read as UTF-8 text, as sklearn takes no bytes labels."""
    array = np.asarray(y)
    if array.dtype.kind != "S" and not (
        array.dtype.kind == "O"
        and any(isinstance(value, bytes) for value in array.ravel())
    ):
        return y
    try:
        return np.array(
            [
                value.decode() if isinstance(value, bytes) else value
                for value in array.ravel()
            ],
            dtype=object,
        ).reshape(array.shape)
    except UnicodeDecodeError:
        raise ValueError("y holds bytes that are not UTF-8 text") from None


def _validated(estimator: BaseEstimator, X: Any, *args: Any, **options: Any) -> Any:
    """Unnamed rows validated as sklearn does, as float32, which holds no text."""
    try:
        return validate_data(
            estimator,
            X,
            *args,
            dtype=np.float32,
            order="C",
            ensure_all_finite="allow-nan",
            **options,
        )
    except ValueError as error:
        if not _holds_text(X):
            raise
        raise ValueError(
            f"{error}: an array is read as numbers, so pass text columns in a "
            "pandas or polars frame or an Arrow table, which reads them as categorical"
        ) from error


def _drop(client: _TableClient, name: str) -> None:
    """Drops an estimator's table, if its client can still be reached."""
    with contextlib.suppress(Exception):
        client.drop_table(name)


class _Estimator(BaseEstimator):
    # Set by sklearn's validation when fitting, as its estimators do.
    n_features_in_: int
    feature_names_in_: np.ndarray

    def __init__(
        self,
        model: Any,
        *,
        client: _TableClient | None = None,
        strategy: Strategy | None = None,
        page_size: int = 128,
        backend: Any = None,
        allow_downloads: bool = False,
    ) -> None:
        self.model = model
        self.client = client
        self.strategy = strategy
        self.page_size = page_size
        self.backend = backend
        self.allow_downloads = allow_downloads
        if client is None and backend is None and isinstance(model, str):
            _load_soon(model, allow_downloads)

    def set_params(self, **params: Any) -> Any:
        """Updates constructor parameters and invalidates fitted context."""
        result = super().set_params(**params)
        if params:
            self.close()
        return result

    def __sklearn_is_fitted__(self) -> bool:
        """Requires successfully prepared context, not residual validation fields."""
        return hasattr(self, "context_")

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        # Missing values are NaN, which the models impute themselves.
        tags.input_tags.allow_nan = True
        return tags

    def _rows(self, X: Any) -> Any:
        """Query rows as the fitted table reads them, checked against its columns."""
        if not self.__sklearn_is_fitted__():
            raise NotFittedError(
                f"call fit before predicting with this {type(self).__name__}"
            )
        kinds = getattr(self, "_kinds", None)
        if kinds is None:
            return _validated(self, X, reset=False)
        if not _named(X):
            raise ValueError(
                "this estimator was fit on a frame, so it predicts from frames "
                f"with its columns: {list(kinds)}"
            )
        table = _text(to_arrow(X))
        if table.column_names != list(kinds):
            raise ValueError(
                "feature names and order must match those passed during fit"
            )
        held = self.context_.select(list(kinds))
        if table.schema != held.schema:
            raise ValueError(
                "feature physical types must match those passed during fit"
            )
        return table

    def _typed_rows(self, X: Any, y: Any) -> tuple[Any, np.ndarray]:
        """A typed frame and its targets, with the fitted columns kept as sklearn keeps them."""
        X = _text(to_arrow(X))
        kinds = infer(X)
        targets = np.asarray(y)
        if targets.ndim != 1 or len(targets) != len(X):
            raise ValueError(
                f"expected {len(X)} targets in one column, got {targets.shape}"
            )
        self._kinds = kinds
        self.n_features_in_ = len(kinds)
        self.feature_names_in_ = np.asarray(list(kinds), dtype=object)
        return X, targets

    def _create(self, features: Any, classes: int | None, targets: np.ndarray) -> Table:
        self._release()
        features = self.context_.drop([self.target_])
        targets = np.array(targets, copy=True)
        client = (
            self.client if self.client is not None else _local(self.allow_downloads)
        )
        name = f"estimator-{uuid.uuid4().hex}"
        kinds = getattr(self, "_kinds", None)
        table = client.create_table(
            name,
            model=self._native_model(),
            features=None if kinds is not None else features.shape[1],
            columns=kinds,
            classes=classes,
            strategy=self.strategy,
            page_size=self.page_size,
        )
        self._dropper = weakref.finalize(self, _drop, client, name)
        # A server may be gone at exit; tables there outlive the process anyway.
        self._dropper.atexit = False
        try:
            table.insert(np.arange(len(features)), features, targets)
        except BaseException:
            self._release()
            raise
        self._fitted = (features, classes, targets)
        return table

    def _native_model(self) -> Any:
        if self.client is not None:
            return self.model
        from generalforecast.local_models import native_model

        task = "classification" if isinstance(self, Classifier) else "regression"
        return native_model(self.model, task)

    def _release(self) -> None:
        dropper = getattr(self, "_dropper", None)
        if dropper is not None:
            dropper()
        for name in ("table_", "_fitted", "_dropper"):
            self.__dict__.pop(name, None)

    def close(self) -> None:
        """Drops the table `fit` created, if any; fit again to predict."""
        self._release()
        for name in (
            "context_",
            "classes_",
            "n_features_in_",
            "feature_names_in_",
            "_spec",
            "target_",
        ):
            self.__dict__.pop(name, None)

    def _prepare(self, rows: Any, y: Any, *, named_target: str) -> None:
        names = list(
            getattr(
                self, "feature_names_in_", [f"f{i}" for i in range(self.n_features_in_)]
            )
        )
        context = context_table(rows, y, names, named_target)
        if self.backend is not None:
            if self.client is not None or self.strategy is not None:
                raise ValueError(
                    "backend cannot be combined with native client or strategy"
                )
            if not isinstance(self.model, str):
                raise ValueError("remote backend requires an admitted catalog model")
            # Package capacity binds remote execution and export, not local paging.
            task = "classification" if isinstance(self, Classifier) else "regression"
            self._spec = admitted_spec(
                self.model,
                task,
                context,
                target=named_target,
                inputs={"include": names},
            )
        self.target_ = named_target
        self.context_ = context

    def to_package(
        self, path: str | Path, *, name: str, id: str | None = None
    ) -> Package:
        """Exports exact fitted context and an offline checked immutable model pin.

        Custom native weights, context strategies and promoting a feature to an
        identifier cannot be represented by the declarative package format.
        """
        if not self.__sklearn_is_fitted__():
            raise NotFittedError("call fit before exporting a package")
        ensure(
            id is None and self.strategy is None and self.client is None,
            "estimator settings cannot be represented faithfully",
            "UnsupportedCapability",
        )
        names = [
            column for column in self.context_.column_names if column != self.target_
        ]
        task = "classification" if isinstance(self, Classifier) else "regression"
        spec = admitted_spec(
            self.model,
            task,
            self.context_,
            target=self.target_,
            inputs={"include": names},
        )
        return export_package(path, name=name, spec=spec, context=self.context_)

    def _remote(
        self, rows: Any, *, quantiles: Sequence[float] | None = None
    ) -> pa.Table:
        names = [
            column for column in self.context_.column_names if column != self.target_
        ]
        queries = to_arrow(
            rows, columns=names if isinstance(rows, np.ndarray) else None
        )
        result = remote_result(
            self.backend,
            spec=self._spec,
            context=self.context_,
            queries=queries,
            features=names,
            options={"quantiles": list(quantiles)} if quantiles is not None else {},
        )
        return result.table

    def _table(self) -> Table:
        """The table, made again from the fitted rows after unpickling."""
        table = getattr(self, "table_", None)
        if table is None and getattr(self, "_fitted", None) is not None:
            table = self.table_ = self._create(*self._fitted)
        if table is None:
            raise NotFittedError(
                f"call fit before predicting with this {type(self).__name__}"
            )
        return table

    def __getstate__(self) -> dict[str, Any]:
        """The estimator without its table, which belongs to this process."""
        state = dict(super().__getstate__())
        for name in ("table_", "_dropper"):
            state.pop(name, None)
        return state


class Classifier(ClassifierMixin, _Estimator):
    """Classifies rows in context of the rows it was fit on.

    Args:
        model: A model name or spec, or the model's contract calls and parameters.
        client: Where the table lives; a local client shared by estimators by default.
        strategy: How each query's context is chosen; exact by default.
        page_size: Rows per page.
    """

    def __init__(
        self,
        model: Any = "tabicl-classifier",
        *,
        client: _TableClient | None = None,
        strategy: Strategy | None = None,
        page_size: int = 128,
        backend: Any = None,
        allow_downloads: bool = False,
    ) -> None:
        """Keeps the settings, as sklearn's estimators do; `fit` makes the table."""
        super().__init__(
            model,
            client=client,
            strategy=strategy,
            page_size=page_size,
            backend=backend,
            allow_downloads=allow_downloads,
        )

    def fit(self, X: Any, y: Any) -> "Classifier":
        """Creates a table of the labeled rows.

        Args:
            X: Rows, with NaN for missing values.
            y: Any labels, with byte strings read as UTF-8 text; `classes_` keeps
                them in sorted order.

        Returns:
            The fitted estimator.
        """
        self.close()
        original_y = y
        self._kinds = None
        if _named(X):
            rows, y = self._typed_rows(X, y)
        else:
            rows, y = _validated(self, X, y)
        y = _labels(y)
        if type_of_target(y) not in ("binary", "multiclass"):
            raise ValueError(f"Unknown label type: {type_of_target(y)}")
        self.classes_, codes = np.unique(np.asarray(y), return_inverse=True)
        rows = rows if isinstance(rows, pa.Table) else np.array(rows, copy=True)
        names = list(
            getattr(
                self, "feature_names_in_", [f"f{i}" for i in range(self.n_features_in_)]
            )
        )
        named_target = target_name(original_y, names)
        self._prepare(rows, y, named_target=named_target)
        if self.backend is None:
            try:
                self.table_ = self._create(
                    rows, len(self.classes_), codes.astype(np.int32)
                )
            except BaseException:
                self.close()
                raise
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        """Probabilities over `classes_`, one row per query."""
        rows = self._rows(X)
        if self.backend is None:
            return self._table().predict(rows)
        table = self._remote(rows)
        ensure(
            "probabilities" in table.column_names,
            "backend did not return probabilities",
            "ProtocolError",
        )
        probabilities = table["probabilities"].to_pylist()
        keys = [
            str(label).lower() if isinstance(label, (bool, np.bool_)) else str(label)
            for label in self.classes_
        ]
        ensure(
            all(
                isinstance(row, dict) and set(row) == set(keys) for row in probabilities
            ),
            "backend returned different classes",
            "ProtocolError",
        )
        result = np.asarray(
            [[row[key] for key in keys] for row in probabilities], dtype=float
        )
        ensure(
            np.isfinite(result).all()
            and (result >= 0).all()
            and np.allclose(result.sum(axis=1), 1, atol=1e-5),
            "backend returned invalid probabilities",
            "ProtocolError",
        )
        return result

    def predict(self, X: Any) -> np.ndarray:
        """The most probable label for each query."""
        probabilities = self.predict_proba(X)
        return self.classes_[probabilities.argmax(axis=1)]

    def score(self, X: Any, y: Any, sample_weight: Any = None) -> float:
        """Mean accuracy on the labeled rows, as sklearn scores, with byte labels as text."""
        return super().score(X, _labels(y), sample_weight)


class Regressor(RegressorMixin, _Estimator):
    """Predicts values in context of the rows it was fit on.

    Args:
        model: A model name or spec, or the model's contract calls and parameters.
        client: Where the table lives; a local client shared by estimators by default.
        strategy: How each query's context is chosen; exact by default.
        page_size: Rows per page.
    """

    def __init__(
        self,
        model: Any = "tabicl-regressor",
        *,
        client: _TableClient | None = None,
        strategy: Strategy | None = None,
        page_size: int = 128,
        backend: Any = None,
        allow_downloads: bool = False,
    ) -> None:
        """Keeps the settings, as sklearn's estimators do; `fit` makes the table."""
        super().__init__(
            model,
            client=client,
            strategy=strategy,
            page_size=page_size,
            backend=backend,
            allow_downloads=allow_downloads,
        )

    def fit(self, X: Any, y: Any) -> "Regressor":
        """Creates a table of the rows and their targets.

        Args:
            X: Rows, with NaN for missing values.
            y: The targets.

        Returns:
            The fitted estimator.
        """
        self.close()
        if _holds_words(y):
            raise ValueError(
                "y holds text, but a Regressor predicts numbers; use a Classifier "
                "to predict labels"
            )
        original_y = y
        self._kinds = None
        if _named(X):
            rows, y = self._typed_rows(X, y)
        else:
            rows, y = _validated(self, X, y, y_numeric=True)
        rows = rows if isinstance(rows, pa.Table) else np.array(rows, copy=True)
        names = list(
            getattr(
                self, "feature_names_in_", [f"f{i}" for i in range(self.n_features_in_)]
            )
        )
        named_target = target_name(original_y, names)
        self._prepare(rows, y, named_target=named_target)
        if self.backend is None:
            try:
                self.table_ = self._create(rows, None, np.asarray(y, dtype=np.float32))
            except BaseException:
                self.close()
                raise
        return self

    def predict_quantiles(
        self, X: Any, quantiles: Sequence[float] | None = None
    ) -> np.ndarray:
        """Sorted quantiles per query.

        Args:
            X: Rows to predict.
            quantiles: Levels strictly between 0 and 1, such as `[0.05, 0.95]`
                for a 90% interval; every quantile the model predicts if None,
                where column k is level (k + 1) / (columns + 1).

        Returns:
            One row per query, one column per level.

        Remote backends require 1..99 explicit levels.
        """
        rows = self._rows(X)
        if self.backend is None:
            return self._table().predict(rows, quantiles=quantiles)
        levels = (
            list(quantiles)
            if quantiles is not None
            else [i / 1000 for i in range(1, 1000)]
        )
        ensure(
            0 < len(levels) <= 99,
            "remote quantiles require 1..99 explicit levels",
            "UnsupportedCapability",
        )
        ensure(
            levels == sorted(set(levels))
            and all(
                0 < level < 1 and abs(level * 1000 - round(level * 1000)) < 1e-9
                for level in levels
            ),
            "invalid quantile levels",
        )
        table = self._remote(rows, quantiles=levels)
        ensure(
            "quantiles" in table.column_names,
            "backend did not return quantiles",
            "ProtocolError",
        )
        values = table["quantiles"].to_pylist()
        keys = [str(float(level)) for level in levels]
        ensure(
            all(isinstance(row, dict) and set(row) == set(keys) for row in values),
            "backend returned different quantiles",
            "ProtocolError",
        )
        result = np.asarray([[row[key] for key in keys] for row in values], dtype=float)
        ensure(
            np.isfinite(result).all() and (np.diff(result, axis=1) >= 0).all(),
            "backend returned invalid quantiles",
            "ProtocolError",
        )
        return result

    def predict(self, X: Any) -> np.ndarray:
        """The median for each query."""
        return self.predict_quantiles(X, [0.5])[:, 0]
