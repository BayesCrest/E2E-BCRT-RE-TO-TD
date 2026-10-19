"""Offline regression tests; expected failures expose independent execution gaps."""
import random
import unittest

import pandas as pd

from momentum_trader.broker import PaperBroker
from momentum_trader.portfolio import Portfolio, Position
from momentum_trader.strategy import MomentumStrategy


class InvariantAssertions:
    def assert_weights(self, weights, cap):
        self.assertLessEqual(sum(weights.values()), 1 + 1e-12)
        for weight in weights.values():
            self.assertGreaterEqual(weight, 0)
            self.assertLessEqual(weight, cap + 1e-12)

    def assert_holdings(self, portfolio, cap):
        self.assertGreater(portfolio.equity, 0)
        self.assertGreaterEqual(portfolio.cash, -1e-8)
        weights = {t: p.quantity * portfolio.market_prices[t] / portfolio.equity
                   for t, p in portfolio.positions.items()}
        self.assert_weights(weights, cap)



class WeightCapTests(InvariantAssertions, unittest.TestCase):
    def test_original_reproducer(self):
        self.assertEqual(MomentumStrategy._apply_max_weight({'AAA': .8, 'BBB': .2}, .3),
                         {'AAA': .3, 'BBB': .2})

    def test_insufficient_names_and_extreme_concentration(self):
        for raw in ({'A': 1.}, {'A': .999999, 'B': .000001}, {'A': .5, 'B': .5}):
            with self.subTest(raw=raw):
                weights = MomentumStrategy._apply_max_weight(raw, .3)
                self.assert_weights(weights, .3)
                self.assertLess(sum(weights.values()), 1)

    def test_empty_zero_and_already_feasible(self):
        for raw in ({}, {'A': 0.}, {'A': .2, 'B': .1}):
            self.assertEqual(MomentumStrategy._apply_max_weight(raw, .3), raw)

    def test_randomised_invariants_and_idempotence(self):
        rng = random.Random(271828)
        for _ in range(1000):
            raw = {str(i): rng.random() for i in range(rng.randint(1, 30))}
            total = sum(raw.values())
            raw = {t: w / total for t, w in raw.items()}
            cap = rng.uniform(.01, 1)
            weights = MomentumStrategy._apply_max_weight(raw, cap)
            self.assert_weights(weights, cap)
            self.assertEqual(MomentumStrategy._apply_max_weight(weights, cap), weights)
            self.assertTrue(all(weights[t] <= raw[t] for t in raw))

    def test_inverse_volatility_and_equal_weight_fallback(self):
        strategy = MomentumStrategy(4, max_weight=.3)
        scores = pd.Series({'A': 2., 'B': 1.})
        self.assertEqual(strategy._compute_weights(scores, pd.Series({'A': .1, 'B': .4})),
                         {'A': .3, 'B': .2})
        self.assertEqual(strategy._compute_weights(scores, None), {'A': .3, 'B': .3})

    def test_evaluate_to_orders_to_holdings_retains_cash(self):
        history = pd.DataFrame({'A': [100, 102, 103, 106, 108, 110],
                                'B': [100, 101, 104, 105, 109, 111]})
        for fee in (0.,):
            with self.subTest(fee=fee):
                portfolio = Portfolio(100000.)
                broker = PaperBroker(portfolio, fee)
                strategy = MomentumStrategy(4, top_n=2, max_weight=.3)
                for end in range(4, len(history) + 1):
                    window = history.iloc[:end]
                    portfolio.update_market_prices(window.iloc[-1].to_dict())
                    weights = strategy.evaluate(window).weights
                    self.assert_weights(weights, .3)
                    orders = portfolio.desired_orders(weights, fee)
                    broker.execute(orders)
                    self.assert_holdings(portfolio, .3)
                    self.assertGreater(portfolio.cash, 39000.)

    def test_fresh_costed_orders_preserve_cash_and_caps(self):
        for fee in (.0005, .01):
            portfolio = Portfolio(100000.)
            portfolio.update_market_prices({'A': 100., 'B': 100.})
            weights = MomentumStrategy._apply_max_weight({'A': .8, 'B': .2}, .3)
            orders = portfolio.desired_orders(weights, fee)
            PaperBroker(portfolio, fee).execute(orders)
            self.assert_holdings(portfolio, .3)
            self.assertAlmostEqual(portfolio.cash, 50000.)
            self.assertAlmostEqual(portfolio.positions['A'].quantity, 300 / (1 + fee))

    def test_zero_cost_rotation_and_explicit_liquidation(self):
        portfolio = Portfolio(100000.)
        portfolio.update_market_prices({'A': 100., 'B': 100., 'C': 100.})
        broker = PaperBroker(portfolio, 0.)
        for weights in ({'A': .3, 'B': .2}, {'B': .3, 'C': .3}, {}):
            broker.execute(portfolio.desired_orders(weights))
            self.assert_holdings(portfolio, .3)
        self.assertEqual(portfolio.positions, {})
        self.assertEqual(portfolio.cash, 100000.)


class KnownExecutionGaps(InvariantAssertions, unittest.TestCase):
    # These remain outside the minimal strategy patch. Unexpected success fails
    # the suite so the markers must be revisited when execution sizing is fixed.
    @unittest.expectedFailure
    def test_repeated_fee_rebalance_can_breach_cap(self):
        portfolio = Portfolio(100000.)
        portfolio.update_market_prices({'A': 100., 'B': 100.})
        broker = PaperBroker(portfolio, .01)
        for prices in ({'A': 100., 'B': 100.}, {'A': 110., 'B': 101.}):
            portfolio.update_market_prices(prices)
            broker.execute(portfolio.desired_orders({'A': .3, 'B': .3}, .01))
            self.assert_holdings(portfolio, .3)

    @unittest.expectedFailure
    def test_fee_on_other_sale_can_breach_post_fee_nav_cap(self):
        portfolio = Portfolio(50000.)
        portfolio.update_market_prices({'A': 100., 'B': 100.})
        portfolio.positions = {'A': Position('A', 300.), 'B': Position('B', 200.)}
        broker = PaperBroker(portfolio, .01)
        broker.execute(portfolio.desired_orders({'A': .3}, .01))
        self.assert_holdings(portfolio, .3)

    @unittest.expectedFailure
    def test_turnover_from_fully_invested_book_can_create_leverage(self):
        portfolio = Portfolio(1.)
        portfolio.cash = 0.
        portfolio.update_market_prices({t: 100. for t in 'ABCDEFGH'})
        portfolio.positions = {t: Position(t, 250.) for t in 'ABCD'}
        weights = {t: .25 for t in 'EFGH'}
        PaperBroker(portfolio, .01).execute(portfolio.desired_orders(weights, .01))
        self.assert_holdings(portfolio, .3)

    @unittest.expectedFailure
    def test_explicit_zero_target_with_fee_leaves_residual(self):
        portfolio = Portfolio(70000.)
        portfolio.update_market_prices({'A': 100.})
        portfolio.positions = {'A': Position('A', 300.)}
        PaperBroker(portfolio, .01).execute(portfolio.desired_orders({'A': 0.}, .01))
        self.assertNotIn('A', portfolio.positions)


if __name__ == '__main__':
    unittest.main()
