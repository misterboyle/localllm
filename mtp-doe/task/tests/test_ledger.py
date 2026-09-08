"""Behavioral tests for the ledger package. DO NOT MODIFY THIS DIRECTORY."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ledger import engine, report


class TestRounding(unittest.TestCase):
    def test_half_up_not_bankers(self):
        # banker's rounding gives 2.34 / 0.12 — must be half-up
        self.assertEqual(engine.round_half_up(2.345, 2), 2.35)
        self.assertEqual(engine.round_half_up(0.125, 2), 0.13)

    def test_half_up_negative_away_from_zero(self):
        self.assertEqual(engine.round_half_up(-2.345, 2), -2.35)

    def test_float_noise(self):
        self.assertEqual(engine.round_half_up(0.1 + 0.2, 2), 0.3)


class TestCategoryTotals(unittest.TestCase):
    def test_refunds_subtract(self):
        txns = [
            {"date": "2026-01-05", "category": "food", "type": "charge", "amount": 10.0},
            {"date": "2026-01-06", "category": "food", "type": "refund", "amount": 2.5},
        ]
        self.assertEqual(engine.category_totals(txns), {"food": 7.5})


class TestMonthly(unittest.TestCase):
    def test_refunds_subtract_per_month(self):
        txns = [
            {"date": "2026-01-10", "category": "tools", "type": "charge", "amount": 10.10},
            {"date": "2026-01-20", "category": "tools", "type": "refund", "amount": 3.05},
            {"date": "2026-02-02", "category": "tools", "type": "charge", "amount": 5.00},
        ]
        stmt = report.monthly_statement(txns)
        self.assertEqual(stmt.get(("2026-01", "tools")), 7.05)
        self.assertEqual(stmt.get(("2026-02", "tools")), 5.0)

    def test_zero_net_pair_dropped(self):
        txns = [
            {"date": "2026-03-01", "category": "misc", "type": "charge", "amount": 5.0},
            {"date": "2026-03-15", "category": "misc", "type": "refund", "amount": 5.0},
        ]
        self.assertNotIn(("2026-03", "misc"), report.monthly_statement(txns))

    def test_line_format(self):
        txns = [
            {"date": "2026-04-01", "category": "tools", "type": "charge", "amount": 5.5},
        ]
        self.assertEqual(report.statement_lines(report.monthly_statement(txns)),
                         ["2026-04 tools 5.50"])


if __name__ == "__main__":
    unittest.main()
