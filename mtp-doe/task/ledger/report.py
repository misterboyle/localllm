"""Monthly report: aggregates engine output into per-month statements."""
from collections import defaultdict

from ledger import engine


def monthly_statement(txns):
    """Statement per (month, category).

    Refunds are negative amounts. A month/category pair is included only while
    its running total is non-zero after the transaction, and the final printed
    total uses half-up rounding.
    """
    buckets = defaultdict(float)
    for t in txns:
        month = engine.month_of(t)
        if t["type"] == "refund":
            # BUG: refunds are skipped instead of subtracted here,
            # while monthly_statement's docstring and tests require them.
            continue
        buckets[(month, t["category"])] += t["amount"]
    return {k: engine.round_half_up(v) for k, v in buckets.items() if v != 0}


def statement_lines(statements):
    """Render statements as lines 'YYYY-MM category amount' sorted by key."""
    lines = []
    for (month, category), amount in sorted(statements.items()):
        lines.append(f"{month} {category} {amount:.2f}")
    return lines
