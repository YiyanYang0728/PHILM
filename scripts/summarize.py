#!/usr/bin/env python3
"""Summarize PHILM prediction metrics.

Python replacement for scripts/summarize.sh. It expects the same single
argument: an output prefix such as results/PHILM_predict_test.
"""

from __future__ import annotations

import argparse
import math
from decimal import Decimal, DivisionByZero, InvalidOperation, getcontext
from pathlib import Path
from typing import Iterable, List


getcontext().prec = 50


def load_first_column(path: Path) -> List[float]:
    values: List[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if '""' in line:
                continue
            if line_number == 1:
                continue
            first_field = line.rstrip("\n").split("\t", 1)[0].strip()
            if not first_field:
                continue
            try:
                value = float(first_field)
            except ValueError:
                continue
            if math.isnan(value):
                continue
            values.append(value)
    return values


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return float("nan")
    return sum(values) / len(values)


def format_number(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.15g}"


def percent(count: int, total: int) -> str:
    if total == 0:
        return "nan"
    try:
        result = (Decimal(count) / Decimal(total)) * Decimal(100)
    except (DivisionByZero, InvalidOperation):
        return "nan"
    return format(result.quantize(Decimal("0.00000000000000000001")), "f")


def top_mean(values: List[float], n: int) -> str:
    return format_number(mean(sorted(values, reverse=True)[:n]))


def summarize_metric(path: Path, display_name: str, count_name: str, threshold: float) -> None:
    values = load_first_column(path)
    count = sum(1 for value in values if value > threshold)
    total = len(values)

    print(f"Mean {display_name}: {format_number(mean(values))}")
    print(f"No. of {count_name} > {threshold:g}: {count}")
    print(f"Perc of {count_name} > {threshold:g}: {percent(count, total)}")
    print(f"Top 50 {count_name} mean: {top_mean(values, 50)}")
    print(f"Top 20 {count_name} mean: {top_mean(values, 20)}")
    print(f"Top 10 {count_name} mean: {top_mean(values, 10)}")


def summarize_group(prefix: Path, suffix: str) -> None:
    summarize_metric(Path(f"{prefix}_pcc{suffix}.tsv"), "PCC", "pcc", 0.8)
    print("-----------------------------------")
    summarize_metric(Path(f"{prefix}_cos_sim{suffix}.tsv"), "Cos Sim", "cos_sim", 0.8)
    print("-----------------------------------")
    summarize_metric(Path(f"{prefix}_r2{suffix}.tsv"), "R2", "R2", 0.6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize PHILM metrics.")
    parser.add_argument("prefix", help="Output prefix, e.g. results/PHILM_predict_test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefix = Path(args.prefix)

    print("#############feature-wise metrics#############")
    summarize_group(prefix, "_ft")
    print("")
    print("#############sample-wise metrics#############")
    summarize_group(prefix, "")


if __name__ == "__main__":
    main()
