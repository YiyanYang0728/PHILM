#!/usr/bin/env python3
"""Permutation p-values for normalized PHILM interaction scores.

Null model:
  For each phage, pool all permuted normalized_score values across bacteria
  and permutation files. Each observed phage-bacterium score is tested against
  that phage-specific pooled null.

P-value:
  One-sided positive upper tail, p = (b + 1) / (m + 1), where b is the number
  of null scores >= the observed score and m is the null size.

Adjustment:
  Benjamini-Hochberg FDR across all observed pairs.
"""

from __future__ import annotations

import argparse
import glob
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


NORM_TOL = 1e-6
NORM_FRAC = 0.99


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute pooled-per-phage permutation p-values for normalized_score."
    )
    parser.add_argument(
        "--observed",
        default="results/PHILM_gradient_signed_interactions.train.tsv",
        help="Observed interaction TSV.",
    )
    parser.add_argument(
        "--perm_glob",
        default="results/permutation_null/perm_*/PHILM_perm_*.raw_gradient.tsv",
        help="Glob pattern for permutation interaction TSV files. Quote this in the shell.",
    )
    parser.add_argument(
        "--out",
        default="results/PHILM_pvalues.tsv",
        help="Output TSV path.",
    )
    parser.add_argument(
        "--norm_mode",
        choices=("auto", "force", "off"),
        default="auto",
        help=(
            "How to handle normalized_score. off: use as-is; force: recompute "
            "from raw_score per phage; auto: recompute only when normalized_score "
            "appears to be a copy of raw_score."
        ),
    )
    return parser.parse_args()


def read_tsv(path: str, required_cols: Iterable[str]) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {', '.join(missing)}")
    return df


def norm_is_missing(df: pd.DataFrame) -> bool:
    if "normalized_score" not in df.columns:
        return True
    diff = (df["normalized_score"] - df["raw_score"]).abs()
    return bool((diff <= NORM_TOL).mean(skipna=True) >= NORM_FRAC)


def recompute_norm(df: pd.DataFrame, label: str, warn_sd0: bool = False) -> pd.DataFrame:
    if "raw_score" not in df.columns:
        raise ValueError(f"Cannot recompute normalized_score: raw_score missing in {label}")

    def zscore(group: pd.Series) -> pd.Series:
        sd = group.std(ddof=1)
        if pd.isna(sd) or sd == 0:
            if warn_sd0:
                print(
                    f"  [warn] phage {group.name} has sd=0; z set to 0",
                    file=sys.stderr,
                )
            return pd.Series(np.zeros(len(group)), index=group.index)
        return (group - group.mean()) / sd

    df = df.copy()
    df["normalized_score"] = df.groupby("phage", sort=False)["raw_score"].transform(zscore)
    return df


def apply_norm(df: pd.DataFrame, norm_mode: str, label: str) -> pd.DataFrame:
    if norm_mode == "off":
        return df
    do_recompute = norm_mode == "force" or (norm_mode == "auto" and norm_is_missing(df))
    if do_recompute:
        return recompute_norm(df, label, warn_sd0=(label == "observed"))
    return df


def build_null(perm_files: List[str], norm_mode: str) -> Dict[str, np.ndarray]:
    null_chunks: Dict[str, List[np.ndarray]] = defaultdict(list)

    for index, path in enumerate(perm_files, start=1):
        required = ["phage", "normalized_score"]
        if norm_mode != "off":
            required.append("raw_score")
        df = read_tsv(path, required)
        df = apply_norm(df, norm_mode, path)

        for phage, values in df.groupby("phage", sort=False)["normalized_score"]:
            arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
            arr = arr[np.isfinite(arr)]
            if len(arr):
                null_chunks[str(phage)].append(arr)

        if index % 50 == 0:
            print(f"  ...processed {index}/{len(perm_files)} files", file=sys.stderr)

    null_sorted: Dict[str, np.ndarray] = {}
    for phage, chunks in null_chunks.items():
        null_sorted[phage] = np.sort(np.concatenate(chunks))
    return null_sorted


def empirical_pvalue(observed: float, null_sorted: Optional[np.ndarray]) -> float:
    if null_sorted is None or len(null_sorted) == 0 or not np.isfinite(observed):
        return np.nan
    first_ge = np.searchsorted(null_sorted, observed, side="left")
    b = len(null_sorted) - first_ge
    return (b + 1) / (len(null_sorted) + 1)


def bh_adjust(pvalues: Iterable[float]) -> np.ndarray:
    pvals = np.asarray(list(pvalues), dtype=float)
    adjusted = np.full(pvals.shape, np.nan, dtype=float)
    finite = np.isfinite(pvals)
    if not finite.any():
        return adjusted

    finite_indices = np.where(finite)[0]
    finite_pvals = pvals[finite]
    order = np.argsort(finite_pvals)
    ranked = finite_pvals[order]
    n = len(ranked)

    ranked_adjusted = ranked * n / np.arange(1, n + 1)
    ranked_adjusted = np.minimum.accumulate(ranked_adjusted[::-1])[::-1]
    ranked_adjusted = np.clip(ranked_adjusted, 0, 1)

    adjusted_indices = finite_indices[order]
    adjusted[adjusted_indices] = ranked_adjusted
    return adjusted


def main() -> None:
    args = parse_args()

    perm_files = sorted(glob.glob(args.perm_glob))
    if not perm_files:
        raise SystemExit(f"No permutation files matched: {args.perm_glob}")
    print(f"Found {len(perm_files)} permutation files.", file=sys.stderr)

    observed_required = ["phage", "bacteria", "normalized_score"]
    if args.norm_mode != "off":
        observed_required.append("raw_score")
    obs = read_tsv(args.observed, observed_required)

    if args.norm_mode == "auto":
        status = "MISSING -> recomputing from raw" if norm_is_missing(obs) else "present -> using as-is"
        print(f"Observed normalization: {status}", file=sys.stderr)
    obs = apply_norm(obs, args.norm_mode, "observed")

    null_sorted = build_null(perm_files, args.norm_mode)

    result = obs[["phage", "bacteria", "normalized_score"]].copy()
    result["pval_normalized_score"] = [
        empirical_pvalue(score, null_sorted.get(str(phage)))
        for phage, score in zip(result["phage"], result["normalized_score"])
    ]
    result["padj_BH_normalized_score"] = bh_adjust(result["pval_normalized_score"])
    result["n_null_normalized_score"] = [
        len(null_sorted.get(str(phage), [])) for phage in result["phage"]
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, sep="\t", index=False)

    qvals = result["padj_BH_normalized_score"]
    print(f"Wrote {len(result)} pairs to {out_path}", file=sys.stderr)
    print(
        "  normalized_score  : significant pairs  "
        f"q<0.05: {(qvals < 0.05).sum()} | q<0.10: {(qvals < 0.10).sum()}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
