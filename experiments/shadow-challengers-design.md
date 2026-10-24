# Shadow challengers — design note (not implemented)

**Status:** proposal only. Nothing in this note is built in the hardening
branch, and no live recorder or workflow change is made.

## Idea

A provisional candidate should keep logging predictions after the backtest
verdict, without ever publishing a signal or touching the champion. Those
predictions are the cheapest possible forward test: they are made before the
outcome exists, so they cannot be overfit by the loop that produced them.

The shadow record would be a parallel `shadow_predictions` stream in the live
recorder, keyed by `(ticker, timestamp, bundle_version, lineage)`, carrying the
model name, its predicted value for each target, and the feature version. Once
the horizon elapses, `forward_score.py` (or a shadow variant) joins the stream
to realised outcomes and reports skill vs the constant baseline and after-cost
policy PnL, exactly like the champion.

## Files it would touch

| File | Change |
| --- | --- |
| `src/rl_crypto/pipeline/live.py` | write shadow rows after the champion row; never let a shadow failure fail the champion write |
| `src/rl_crypto/cache.py` | schema/validation for `live.jsonl` shadow rows (or a separate `shadow.jsonl` per ticker) |
| `src/rl_crypto/ml/serve.py` | load a candidate bundle in a read-only path; no cache mutation |
| `tools/forward_score.py` | score shadow rows when present; keep the champion path unchanged |
| `experiments/hypotheses.json` | a `shadow` field on a provisional lineage (bundle path, max days, max tickers) |
| `tools/model_improve.py` | only the driver may register/expire a shadow; a provisional verdict is the trigger |
| `.github/workflows/production.yml` | no new job; the existing live step writes the extra rows |
| `tests/test_record_live.py`, `tests/test_ml_serve.py`, `tests/test_forward_score.py` | shadow write/score, failure isolation, attribution |

## Risks

1. **Leakage via selection.** A human (or the loop) could watch shadow results
   and promote the best shadow, which is just another holdout. The shadow
   lineage must be pre-registered before its first prediction and its result
   may not be used to pick a config.
2. **Cost and latency in the live path.** A second model per target doubles
   inference and disk. Cap it to one candidate, one ticker set, a fixed window
   (e.g. 14 days), and make a shadow failure non-fatal.
3. **Attribution ambiguity.** Rows written before a bundle swap or without a
   `bundle_trained_at` are unidentifiable. The champion scorer already refuses
   `unknown` attribution; shadows need the same guard.
4. **Schema drift.** `live.jsonl` is the production cache; a malformed shadow
   row must not break the cache audit. Prefer a separate file or a namespaced
   field with a schema version.
5. **False confidence.** A shadow that matches the backtest is still one
   sample. Require the same minimum window and sample count as the champion
   forward test before calling anything confirmed.
