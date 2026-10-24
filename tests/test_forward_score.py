"""Offline tests for the forward scorer (served predictions vs realised outcomes)."""

from __future__ import annotations

import importlib.util
import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from rl_crypto.models import DatasetRow, to_iso

_SPEC = importlib.util.spec_from_file_location(
    "forward_score", Path(__file__).resolve().parents[1] / "tools" / "forward_score.py"
)
assert _SPEC is not None and _SPEC.loader is not None
forward = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(forward)

START = datetime(2026, 1, 1, tzinfo=UTC)


def _rows(n: int) -> tuple[DatasetRow, ...]:
    return tuple(
        DatasetRow(
            ticker="BTC/USD",
            timestamp=to_iso(START + timedelta(hours=i)) or "",
            open=100.0 + i,
            high=101.0 + i,
            low=99.0 + i,
            close=100.0 + i,
            volume=50.0,
            has_price=True,
        )
        for i in range(n)
    )


def _record(
    index: int,
    *,
    probability: float | None = 0.7,
    bundle: str = "2026-01-01T00:00:00Z",
    expected_return: float = 0.001,
    include_metadata: bool = True,
) -> dict:
    stamp = to_iso(START + timedelta(hours=index))
    predictions: dict = {"fwd_return": expected_return, "fwd_direction": 1.0}
    probabilities: dict = {"fwd_direction": probability} if probability is not None else {}
    payload: dict = {
        "schema": "rl-crypto/ml/1",
        "ticker": "BTC/USD",
        "generated_at": stamp,
        "horizon": 1,
        "feature_timestamp": stamp,
        "predictions": predictions,
        "probabilities": probabilities,
        "features": {"ret_1": 0.001},
        "bundle_trained_at": bundle,
    }
    if include_metadata:
        payload["feature_version"] = "rl-crypto/features/2"
        payload["metric_space"] = {
            "fwd_return": "raw",
            "fwd_direction": "raw",
        }
    return {"generated_at": stamp, "predictions": payload}


def test_target_rows_baseline_uses_only_earlier_outcomes():
    """If the baseline saw the row it scores, skill would be 0.5, not 1.0."""
    result = forward._target_rows("fwd_return", [(0.0, 0.0, None), (1.0, 1.0, None)])
    assert result["n"] == 1  # the first row has no prior outcome, so no baseline
    assert result["model"] == pytest.approx(0.0)
    assert result["baseline"] == pytest.approx(1.0)
    assert result["skill"] == pytest.approx(1.0)


def test_score_ticker_joins_resolved_and_counts_the_rest():
    rows = _rows(40)
    records = [
        _record(i) for i in range(0, 35)
    ]
    # A feature bar that is no longer in the archive.
    stale = _record(0)
    stale["predictions"]["feature_timestamp"] = "2025-12-01T00:00:00Z"
    records.append(stale)
    # A prediction whose forward bar has not landed yet.
    records.append(_record(39))

    result = forward.score_ticker("BTC/USD", records, rows, fee_bps=0, slippage_bps=0)

    assert result["counts"]["resolved"] == 35
    assert result["counts"]["unjoined"] == 1
    # Row 39 is the last bar, so its h=1 label is still unknown.
    assert result["counts"]["unresolved"] == 1
    assert len(result["bundles"]) == 1
    bundle = result["bundles"][0]
    assert bundle["bundle"] == "2026-01-01T00:00:00Z"
    assert bundle["resolved"] == 35
    assert bundle["targets"]["fwd_return"]["kind"] == "regression"
    assert bundle["targets"]["fwd_direction"]["kind"] == "classification"
    assert bundle["economic"]["model"]["trades"] >= 1
    assert bundle["window_met"] is False


def test_missing_prediction_is_skipped_not_imputed():
    rows = _rows(40)
    record = _record(5, probability=None)
    record["predictions"]["predictions"] = {}  # no point predictions at all

    result = forward.score_ticker("BTC/USD", [record], rows, fee_bps=0, slippage_bps=0)

    assert result["counts"]["resolved"] == 1
    assert result["skipped_positions"] == 1
    bundle = result["bundles"][0]
    assert bundle["targets"] == {}
    assert bundle["economic"]["model"]["bars"] == 0


def test_costs_reduce_the_forward_policy_return():
    rows = _rows(40)
    records = [_record(i) for i in range(35)]

    free = forward.score_ticker("BTC/USD", records, rows, fee_bps=0, slippage_bps=0)
    costly = forward.score_ticker("BTC/USD", records, rows, fee_bps=100, slippage_bps=50)

    free_return = free["bundles"][0]["economic"]["model"]["total_return"]
    costly_return = costly["bundles"][0]["economic"]["model"]["total_return"]
    assert costly_return < free_return


def test_bundle_versions_are_scored_separately():
    rows = _rows(40)
    records = [_record(i, bundle="2026-01-01T00:00:00Z") for i in range(10)]
    records += [_record(i, bundle="2026-02-01T00:00:00Z") for i in range(10, 35)]

    result = forward.score_ticker("BTC/USD", records, rows, fee_bps=0, slippage_bps=0)

    versions = {bundle["bundle"] for bundle in result["bundles"]}
    assert versions == {"2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z"}


def test_forward_score_reads_a_repo_root_and_reports_an_empty_ticker(tmp_path):
    ticker_dir = tmp_path / "data" / "btc-usd"
    ticker_dir.mkdir(parents=True)
    rows = _rows(40)
    (ticker_dir / "candles.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "ticker": row.ticker,
                    "timestamp": row.timestamp,
                    "open": row.open,
                    "high": row.high,
                    "low": row.low,
                    "close": row.close,
                    "volume": row.volume,
                }
            )
            for row in rows
        )
        + "\n"
    )
    (ticker_dir / "live.jsonl").write_text(
        "\n".join(json.dumps(_record(i)) for i in range(35)) + "\n"
    )

    payload = forward.forward_score(tmp_path, tickers=["BTC/USD"])

    assert payload["summary"]["tickers"] == 1
    assert payload["summary"]["resolved"] == 35
    assert payload["results"][0]["ticker"] == "BTC/USD"


def test_cli_json_output(tmp_path, capsys):
    ticker_dir = tmp_path / "data" / "btc-usd"
    ticker_dir.mkdir(parents=True)
    (ticker_dir / "candles.jsonl").write_text("")
    (ticker_dir / "live.jsonl").write_text("")

    assert forward.main(["--root", str(tmp_path), "--tickers", "BTC/USD", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["action"] == "forward-score"
    assert payload["results"][0]["error"] == "no live records"


def test_log_space_target_rows_score_logs_against_a_log_baseline():
    """``fwd_volatility`` is fitted in log space; the forward test must match."""
    observations = [(0.01, 0.02, None), (0.04, 0.04, None)]
    result = forward._target_rows("fwd_volatility", observations, space="log")
    assert result["metric_space"] == "log"
    assert result["n"] == 1
    # log(0.04) predicted exactly, log(0.01) constant baseline.
    assert result["model"] == pytest.approx(0.0)
    assert result["baseline"] == pytest.approx(abs(math.log(0.04) - math.log(0.01)))
    assert result["skill"] == pytest.approx(1.0)

    raw = forward._target_rows("fwd_volatility", observations, space="raw")
    assert raw["metric_space"] == "raw"
    assert raw["model"] == pytest.approx(0.0)
    assert raw["baseline"] == pytest.approx(0.03)


def test_metric_spaces_reads_the_deployed_bundle_metadata(tmp_path):
    from rl_crypto.ml.store import save_bundle
    from rl_crypto.ml.tier1 import train_tier1
    from rl_crypto.samples import build_samples

    from .conftest import make_dataset_rows

    samples = build_samples(make_dataset_rows(160, wiggle=0.03))
    bundle = train_tier1(
        samples,
        targets=("fwd_volatility",),
        model_names={"fwd_volatility": "log_ridge"},
        holdout_fraction=0.2,
    )
    save_bundle(bundle, tmp_path / "models" / "btc-usd.json")
    assert forward._metric_spaces(tmp_path, "BTC/USD") == {"fwd_volatility": "log"}


def test_old_records_are_excluded_from_confirmation_windows():
    rows = _rows(420)
    records = [_record(i, include_metadata=False) for i in range(400)]
    result = forward.score_ticker("BTC/USD", records, rows, fee_bps=0, slippage_bps=0)
    assert result["counts"]["resolved"] == 400
    assert result["counts"]["metadata_missing"] == 400
    bundle = result["bundles"][0]
    assert bundle["confirmation_resolved"] == 0
    assert bundle["excluded_missing_metadata"] == 400
    assert bundle["window_met"] is False


def test_new_records_can_meet_the_confirmation_window():
    rows = _rows(420)
    records = [_record(i) for i in range(400)]
    result = forward.score_ticker("BTC/USD", records, rows, fee_bps=0, slippage_bps=0)
    assert result["counts"]["metadata_missing"] == 0
    bundle = result["bundles"][0]
    assert bundle["confirmation_resolved"] == 400
    assert bundle["confirmation_window_days"] >= forward.MIN_FORWARD_DAYS
    assert bundle["window_met"] is True
