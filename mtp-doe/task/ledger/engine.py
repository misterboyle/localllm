"""Ledger engine: load transactions and compute rounded category totals."""
import csv
from decimal import Decimal


def round_half_up(value, digits=2):
    """Round to `digits` decimal places using half-up convention.

    2.345 -> 2.35, 0.125 -> 0.13 (NEVER banker's rounding).
    """
    # BUG: uses Python's built-in round() which is banker's rounding.
    return round(value, digits)


def load_transactions(path):
    """Read a transactions CSV into a list of dicts with float amount."""
    txns = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            txns.append({
                "date": row["date"],
                "category": row["category"],
                "type": row["type"],          # "charge" or "refund"
                "amount": float(row["amount"]),
            })
    return txns


def month_of(txn):
    return txn["date"][:7]


def category_totals(txns):
    """Total per category: charges positive, refunds subtract. Half-up rounding."""
    totals = {}
    for t in txns:
        amt = t["amount"] if t["type"] == "charge" else -t["amount"]
        totals[t["category"]] = totals.get(t["category"], 0.0) + amt
    return {cat: round_half_up(v) for cat, v in totals.items()}
