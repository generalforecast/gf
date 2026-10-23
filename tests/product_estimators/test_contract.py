# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Estimator contracts without network access or downloaded model weights."""

import datetime
import socket

import numpy as np
import pyarrow as pa
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from generalforecast.contracts import PredictionResult
from generalforecast.estimators import Classifier, Forecaster, Regressor
from generalforecast.models.catalog import V1_POLICY
from generalforecast.packages import Package, PackageError, PackageLock


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("estimator contract tests must remain offline")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)


class Backend:
    """Protocol test double, not engine inference evidence."""

    def __init__(self):
        self.requests = []
        self.result = None

    def predict(self, **request):
        self.requests.append(request)
        if self.result is not None:
            return self.result
        count = len(request["queries"])
        if request["task"] == "classification":
            output = {
                "label": ["a"] * count,
                "probabilities": [{"a": 0.75, "b": 0.25}] * count,
            }
        else:
            levels = request["options"].get("quantiles", [0.1, 0.5, 0.9])
            output = {
                "value": [2.0] * count,
                "quantiles": [{str(float(q)): float(q) for q in levels}] * count,
            }
        return PredictionResult(
            pa.table(output), model=request["model"], task=request["task"]
        )


def history():
    return pa.table(
        {
            "store": ["a"] * 4 + ["b"] * 4,
            "time": [0, 1, 2, 3] * 2,
            "sales": [1.0, 2.0, 3.0, 4.0] * 2,
        }
    )


def future():
    # Interleaved and reversed rows must still map outputs to the original order.
    return pa.table({"store": ["b", "a", "b", "a"], "time": [5, 4, 4, 5]})


@pytest.mark.parametrize("estimator", [Classifier(), Regressor(), Forecaster()])
def test_clone_params_and_unfitted(estimator):
    copy = clone(estimator)
    assert copy.get_params() == estimator.get_params()
    assert copy.set_params(allow_downloads=True) is copy
    with pytest.raises(NotFittedError):
        copy.predict(np.ones((2, 2)))
    with pytest.raises(NotFittedError):
        copy.to_package("unused", name="unused")


@pytest.mark.parametrize("adapter", ["arrow", "numpy", "pandas", "polars", "lazy"])
def test_tabular_adapters_and_target_names(tmp_path, adapter):
    backend = Backend()
    frame = pa.table({"price": [1.0, 2.0, 3.0, 4.0], "stock": [2.0, 3.0, 4.0, 5.0]})
    if adapter == "numpy":
        X, y = (
            np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]),
            np.array(["a", "b", "a", "b"]),
        )
    elif adapter == "pandas":
        pd = pytest.importorskip("pandas")
        X, y = frame.to_pandas(), pd.Series(["a", "b", "a", "b"], name="outcome")
    elif adapter in ("polars", "lazy"):
        pl = pytest.importorskip("polars")
        X, y = pl.from_arrow(frame), pl.Series("outcome", ["a", "b", "a", "b"])
        X = X.lazy() if adapter == "lazy" else X
    else:
        X, y = frame, np.array(["a", "b", "a", "b"])
    estimator = Classifier(backend=backend).fit(X, y)
    np.testing.assert_array_equal(estimator.predict(X), ["a"] * 4)
    assert estimator.score(X, y) == 0.5
    exported = estimator.to_package(tmp_path / adapter, name="portable")
    assert isinstance(exported.lock, PackageLock)
    assert exported.lock.spec.target == (
        "outcome" if adapter in ("pandas", "polars", "lazy") else "target"
    )
    assert exported.snapshot_table().num_rows == 4
    assert exported.lock.model.name == "tabicl-v2-classifier"
    assert Package.load(tmp_path / adapter).check(frozen=True).lock == exported.lock


def test_export_identity_owns_context_and_never_overwrites(tmp_path):
    features = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
    target = np.array([1.0, 2.0, 3.0, 4.0])
    estimator = Regressor(backend=Backend()).fit(features, target)
    features[:] = 0
    target[:] = 0
    first = estimator.to_package(tmp_path / "one", name="portable")
    second = estimator.to_package(tmp_path / "two", name="portable")
    assert first.lock is not None and second.lock is not None
    assert first.lock.package_digest == second.lock.package_digest
    assert first.snapshot_table()["f0"].to_pylist() == [1.0, 2.0, 3.0, 4.0]
    assert first.snapshot_table()["target"].to_pylist() == [1.0, 2.0, 3.0, 4.0]
    with pytest.raises(PackageError, match="already exists"):
        estimator.to_package(tmp_path / "one", name="portable")
    with pytest.raises(PackageError, match="faithfully"):
        estimator.to_package(tmp_path / "three", name="portable", id="f0")
    estimator.set_params(page_size=64)
    with pytest.raises(NotFittedError):
        estimator.predict(np.ones((1, 2)))


def test_named_order_types_and_caller_buffers_are_preserved():
    a, b = np.arange(4.0), np.arange(4.0) + 2
    frame = pa.table({"a": a, "b": b})
    estimator = Regressor(backend=Backend()).fit(frame, np.arange(4.0))
    a[:] = 100
    assert estimator.context_["a"].to_pylist() == [0.0, 1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="names"):
        estimator.predict(frame.select(["b", "a"]))
    with pytest.raises(ValueError, match="types"):
        estimator.predict(pa.table({"a": [1, 2], "b": [2.0, 3.0]}))
    with pytest.raises(ValueError, match="frames"):
        estimator.predict(np.ones((2, 2)))


def test_byte_string_columns_are_read_as_text():
    backend = Backend()
    protocols = pa.array([b"tcp", b"udp", b"icmp", b"tcp"], pa.binary())
    raw = pa.table({"protocol": protocols, "bytes": [1.0, 2.0, 3.0, 4.0]})
    estimator = Regressor(backend=backend).fit(raw, np.arange(4.0))
    assert estimator.context_.schema.field("protocol").type == pa.string()
    assert estimator.context_["protocol"].to_pylist() == ["tcp", "udp", "icmp", "tcp"]
    estimator.predict(raw)
    assert backend.requests[-1]["queries"]["protocol"].to_pylist() == [
        "tcp",
        "udp",
        "icmp",
        "tcp",
    ]
    invalid = raw.set_column(0, "protocol", pa.array([b"\xff"] * 4, pa.binary()))
    with pytest.raises(ValueError, match="'protocol' holds bytes that are not UTF-8"):
        Regressor(backend=Backend()).fit(invalid, np.arange(4.0))
    rows = np.array([[b"tcp", 1.0], [b"udp", 2.0]], dtype=object)
    with pytest.raises(ValueError, match="pass text columns in a pandas or polars"):
        Regressor(backend=Backend()).fit(rows, np.arange(2.0))


@pytest.mark.parametrize(
    "fault", ["type", "model", "task", "rows", "classes", "probability"]
)
def test_backend_contract_failures_are_not_success(fault):
    backend = Backend()
    estimator = Classifier(backend=backend).fit(np.ones((4, 2)), ["a", "b", "a", "b"])
    table = pa.table({"probabilities": [{"a": 0.75, "b": 0.25}] * 2})
    model, task = estimator._spec.model, "classification"
    if fault == "model":
        model = "other"
    if fault == "task":
        task = "regression"
    if fault == "rows":
        table = table.slice(0, 1)
    if fault == "classes":
        table = pa.table({"probabilities": [{"a": 1.0}] * 2})
    if fault == "probability":
        table = pa.table({"probabilities": [{"a": -1.0, "b": 2.0}] * 2})
    backend.result = (
        {} if fault == "type" else PredictionResult(table, model=model, task=task)
    )
    with pytest.raises(PackageError) as error:
        estimator.predict(np.ones((2, 2)))
    assert error.value.code == "ProtocolError"


def test_byte_string_labels_are_read_as_text():
    X = pa.table({"x": [1.0, 2.0, 3.0, 4.0]})
    labels = np.array([b"a", b"b", b"a", b"b"], dtype=object)
    estimator = Classifier(backend=Backend()).fit(X, labels)
    np.testing.assert_array_equal(estimator.classes_, ["a", "b"])
    np.testing.assert_array_equal(estimator.predict(X), ["a"] * 4)
    assert estimator.score(X, labels) == 0.5
    with pytest.raises(ValueError, match="not UTF-8"):
        Classifier(backend=Backend()).fit(X, np.array([b"\xff", b"a", b"\xff", b"a"]))
    for target in (labels, labels.astype(str)):
        with pytest.raises(ValueError, match="use a Classifier to predict labels"):
            Regressor(backend=Backend()).fit(X, target)


def test_a_regressor_reads_numeric_text_and_forgets_its_table_on_a_text_refit():
    X = np.arange(4.0).reshape(-1, 1)
    numbers = np.array(["1", "2", "3", "4"], dtype=object)
    estimator = Regressor(backend=Backend()).fit(X, numbers)
    assert estimator.predict(X).shape == (4,)
    with pytest.raises(ValueError, match="use a Classifier to predict labels"):
        estimator.fit(X, np.array(["a", "b", "a", "b"], dtype=object))
    with pytest.raises(NotFittedError):
        estimator.predict(X)


def test_forecast_cadence_horizon_and_export(tmp_path):
    backend = Backend()
    estimator = Forecaster(
        target="sales", series="store", horizon=2, backend=backend
    ).fit(history())
    assert estimator.predict(future()).shape == (4,)
    request = backend.requests[-1]
    assert request["options"]["horizon"] == 2
    assert request["options"]["series"] == ["store"]
    exported = estimator.to_package(tmp_path / "forecast", name="forecast")
    assert isinstance(exported.lock, PackageLock)
    assert exported.lock.schema.frequency == 1
    assert exported.lock.spec.horizon == 2
    assert exported.lock.spec.series == ("store",)
    assert exported.snapshot_table().equals(estimator.context_)
    with pytest.raises(PackageError, match="horizon"):
        estimator.predict(future().slice(0, 1))
    with pytest.raises(PackageError, match="grid"):
        estimator.predict(pa.table({"store": ["a", "a"], "time": [5, 6]}))
    with pytest.raises(PackageError, match="absent"):
        estimator.predict(pa.table({"store": ["c", "c"], "time": [4, 5]}))
    estimator.set_params(horizon=3)
    with pytest.raises(NotFittedError):
        estimator.predict(future())


@pytest.mark.parametrize("times", [[0, 1, 3, 4], [0, 1, 1, 2]])
def test_forecaster_rejects_irregular_or_duplicate_context(times):
    with pytest.raises(PackageError):
        Forecaster(backend=Backend()).fit(
            pa.table({"time": times, "target": [1.0, 2.0, 3.0, 4.0]})
        )


def test_forecaster_datetime_and_separate_target():
    day = datetime.date(2026, 1, 1)
    features = pa.table({"time": [day + datetime.timedelta(days=i) for i in range(4)]})
    estimator = Forecaster(backend=Backend(), horizon=2).fit(
        features, [1.0, 2.0, 3.0, 4.0]
    )
    query = pa.table({"time": [day + datetime.timedelta(days=i) for i in (4, 5)]})
    assert estimator.predict(query).shape == (2,)
    assert estimator._frequency == 86400


def test_local_defaults_fail_without_checkpoints_and_never_download(monkeypatch):
    def unavailable(entry, *, surface="local", allow_downloads):
        assert allow_downloads is False
        raise RuntimeError("verified model artifacts are not cached")

    monkeypatch.setattr("generalforecast.models.artifacts.checkpoint", unavailable)
    estimator = Classifier()
    with pytest.raises(RuntimeError, match="not cached"):
        estimator.fit(np.ones((4, 2)), ["a", "b", "a", "b"])
    with pytest.raises(NotFittedError):
        estimator.predict(np.ones((1, 2)))


def test_package_capacity_binds_remote_and_export_not_local_paging(monkeypatch):
    def loading(entry, **_):
        raise RuntimeError("local loader reached")

    monkeypatch.setattr("generalforecast.models.artifacts.checkpoint", loading)
    count = V1_POLICY.context_rows + 1
    rows, target = np.ones((count, 2)), np.arange(float(count))
    with pytest.raises(RuntimeError, match="loader reached"):
        Regressor().fit(rows, target)
    with pytest.raises(PackageError) as error:
        Regressor(backend=Backend()).fit(rows, target)
    assert error.value.code == "CapacityExceeded"


def test_regression_backend_levels_and_unsupported_default():
    estimator = Regressor(backend=Backend()).fit(np.ones((4, 2)), np.arange(4.0))
    assert estimator.predict_quantiles(np.ones((2, 2)), [0.1, 0.5, 0.9]).shape == (2, 3)
    assert estimator.backend.requests[-1]["options"]["outputs"] == [
        "value",
        "quantiles",
    ]
    with pytest.raises(PackageError, match="explicit levels"):
        estimator.predict_quantiles(np.ones((2, 2)))
    with pytest.raises(PackageError, match="invalid quantile"):
        estimator.predict_quantiles(np.ones((2, 2)), [0.5, 0.1])
    with pytest.raises(PackageError, match="invalid quantile"):
        estimator.predict_quantiles(np.ones((2, 2)), [0.1234])


def test_failed_refit_invalidates_old_context_and_name_collisions():
    estimator = Classifier(backend=Backend()).fit(np.ones((4, 2)), ["a", "b"] * 2)
    with pytest.raises(ValueError):
        estimator.fit(np.ones((4, 2)), ["a"])
    assert not estimator.__sklearn_is_fitted__()
    frame = pa.table({"target": [1.0, 2.0, 3.0, 4.0]})
    estimator.fit(frame, ["a", "b"] * 2)
    assert estimator.target_ == "_target"


def test_forecaster_explicit_covariates_and_missing_future_rows():
    context = (
        history()
        .append_column("past", pa.array([1.0] * 8))
        .append_column("known", pa.array([2.0] * 8))
    )
    estimator = Forecaster(
        target="sales",
        series="store",
        horizon=2,
        backend=Backend(),
        past_inputs=("past",),
        future_inputs=("known",),
    ).fit(context)
    with pytest.raises(PackageError, match="known-future"):
        estimator.predict(future())
    query = future().append_column("known", pa.array([3.0] * 4))
    assert estimator.predict(query).shape == (4,)
    with pytest.raises(PackageError):
        Forecaster(target="sales", series="store", horizon=2, backend=Backend()).fit(
            context
        )


def test_related_forecasts_reject_nonportable_group_semantics():
    context = history().append_column("known", pa.array([1.0] * 4 + [2.0] * 4))
    with pytest.raises(PackageError, match="historical covariates"):
        Forecaster(
            target="sales",
            series="store",
            horizon=2,
            future_inputs=("known",),
            backend=Backend(),
        ).fit(context)
    context = history().append_column("known", pa.array([1.0] * 8))
    estimator = Forecaster(
        target="sales",
        series="store",
        horizon=2,
        future_inputs=("known",),
        backend=Backend(),
    ).fit(context)
    with pytest.raises(PackageError, match="future covariates"):
        estimator.predict(
            future().append_column("known", pa.array([1.0, 1.0, 2.0, 2.0]))
        )
