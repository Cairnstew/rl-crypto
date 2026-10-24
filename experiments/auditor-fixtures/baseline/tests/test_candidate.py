from rl_crypto.candidate import policy_pnl


def test_pnl_charges_the_close():
    assert policy_pnl([1.0], [0.01], fee_bps=10, slippage_bps=5) < 0.01
