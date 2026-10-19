# PR #1: minimal target-weight cap repair

Base: `36870fb42c4c6346cdd227d1509d1c832ebf2a71` (PR #1 head).
Only production change: `MomentumStrategy._apply_max_weight` clips weights
without redistributing or renormalising. For 80%/20%, cap 30%, the output is
30%/20%, leaving 50% cash. This conservative policy can retain cash even when
other names have capacity. Maximising investment is not part of this patch.
The internal method assumes finite nonnegative input weights summing to at most
one, as produced by `_compute_weights`; it is not a general constraint solver.

## Weight transmission

`evaluate -> _compute_weights -> _apply_max_weight -> signal.weights ->
Portfolio.desired_orders -> PaperBroker.execute -> Portfolio.apply_execution`.
The engine passes transaction cost to desired_orders and the broker. Neither
Portfolio nor PaperBroker renormalises weights. At zero cost and valid prices,
orders reach target holdings and leave the unused allocation as cash. Fresh
purchases with fees also preserve the tested caps and cash. Repeated costed
rebalancing does NOT universally preserve post-fee NAV caps (see below).

## Reproduce offline

Python 3.11+; install `numpy>=1.24 pandas>=1.5` if absent. No Yahoo data, account,
network, training artifact or live trading access is used by these tests.

```bash
python -m unittest discover -s tests -v
python -m compileall -q momentum_trader main.py tests
```

Actual local result: 12 test methods, 8 passed and 4 expected failures;
`OK (expected failures=4)`. Python dependencies: numpy 2.3.5, pandas 2.2.3.
One test checks 1,000 deterministic random portfolios (seed 271828).
Compile check passed. On the unchanged PR #1 strategy, the same suite returned
exit 1, `FAILED (failures=9, expected failures=4)`; failures include subtests.
These are local results, not GitHub Actions results.

Assertions explicitly check nonnegative weights, single-name cap, sum <= 1,
nonnegative cash, and positive equity. Scenarios cover concentration, insufficient
names, inverse volatility, equal-weight fallback, idempotence, empty/zero weights,
repeated zero-cost rebalances, cash preservation, costed initial purchases,
rotation, and liquidation.

## Unresolved execution invariants: BLOCKED

The four expected failures are reproductions of existing defects, not passing
risk checks. Unexpected success will fail unittest so markers must be reviewed
when sizing is repaired. To run their raw assertions as a strict diagnostic:

```bash
PYTHONPATH=tests python - <<'PY'
from test_weight_caps import KnownExecutionGaps
for name in sorted(n for n in dir(KnownExecutionGaps) if n.startswith('test_')):
    test = KnownExecutionGaps(name)
    try:
        getattr(KnownExecutionGaps, name).__wrapped__(test)
    except AssertionError as exc:
        print(name, 'FAIL:', exc)
    else:
        raise RuntimeError('Expected defect no longer reproduced: ' + name)
raise SystemExit(1)  # Known invariant violations remain; deliberately nonzero.
PY
```

1. Selling B while A remains at 30% of pre-fee NAV reduces NAV and makes A >30%.
   Example: cash 50,000; A 30,000; B 20,000; exit B at 1% fees -> A/NAV
   = 30,000/99,800 = 30.06012%.
2. Repeated costed rebalances can similarly exceed the cap because the existing
   adjustment shrinks sell quantities and targets pre-fee equity.
3. Rotating a fully invested 4-name book into four new names at 1% cost can leave
   cash -1,000: new purchases consume all initial NAV including buy costs, then
   sell costs consume another 1,000. Gross exceeds NAV.
4. Explicit zero targets leave residual holdings because negative quantities are
   divided by 1 + fee. This is the independent existing Codex review issue.

Portfolio/broker production code is deliberately unchanged in this narrowly
scoped strategy patch. End-to-end post-fee cap and no-leverage certification is
therefore NOT achieved. A separate execution-sizing repair should solve targets
against post-fee equity, account for sell proceeds and costs, execute sells before
buys if enforcing intermediate cash constraints, and reject infeasible orders.

Other gaps: no sector gross caps, aggregate risk budgeting, liquidity/lot-size
constraints, pending-order reservations, broker-level independent cap checking,
or protection from market-price drift between rebalances. Direct Portfolio calls
can supply invalid/leveraged weights; missing prices can prevent liquidation.
The strategy constructor lacks the config's cap validation. None is repaired here.

## Isolation and authority

This is a stacked draft against PR #1's branch, not main. No changes to PR #2,
training outputs, Research-Shadow CI, dependencies, or workflows. No merge,
live broker, or risk permission changes. Tests do not establish PIT, OOS,
portfolio suitability, investment edge, or real-world execution safety.
