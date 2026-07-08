#!/usr/bin/env python3

"""
Split microbial profile tables into training, validation, and test sets.

Input tables are expected to have:
    - rows as taxa/features
    - columns as samples
    - the first column as feature names / row names

By default, the script:
    - filters features below a relative-abundance cutoff only if --threshold is provided
    - removes all-zero features
    - removes samples with all-zero phage abundances
    - splits common samples into train/val/test using an 8:1:1 ratio
    - writes transformed matrices without row names or column names
    - writes normalized relative-abundance matrices with the suffix "_no_clr.tsv"
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


PSEUDOCOUNT = 1e-10


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Split paired bacterial/archaeal and phage abundance profiles "
            "into train, validation, and test sets."
        )
    )

    parser.add_argument(
        "-b",
        "--bact-arc",
        required=True,
        help="Input bacterial/archaeal profile table. Rows are taxa and columns are samples.",
    )
    parser.add_argument(
        "-p",
        "--phage",
        required=True,
        help="Input phage profile table. Rows are taxa and columns are samples.",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        required=True,
        help="Output directory for the split matrices and sample/feature name files.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Optional abundance cutoff. Values below this cutoff are set to zero "
            "before removing all-zero features."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=777,
        help="Random seed used to shuffle samples before splitting. Default: 777.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of samples assigned to the training set. Default: 0.8.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of samples assigned to the validation set. Default: 0.1.",
    )
    parser.add_argument(
        "--transform",
        choices=["log2", "clr", "none"],
        default="log2",
        help=(
            "Transformation applied to normalized profiles before writing "
            "Bact_arc_train.tsv, Phage_train.tsv, etc. "
            "Default: log2, matching the original script."
        ),
    )

    return parser.parse_args()


def validate_args(args):
    if args.threshold is not None and args.threshold < 0:
        raise ValueError("--threshold must be non-negative.")

    if not 0 < args.train_ratio < 1:
        raise ValueError("--train-ratio must be between 0 and 1.")

    if not 0 <= args.val_ratio < 1:
        raise ValueError("--val-ratio must be between 0 and 1.")

    if args.train_ratio + args.val_ratio >= 1:
        raise ValueError("--train-ratio + --val-ratio must be less than 1.")


def read_profile(path):
    df = pd.read_table(path, sep="\t", index_col=0)
    df = df.apply(pd.to_numeric)
    return df


def clr_transform(matrix):
    """
    Apply centered log-ratio transformation to a sample-by-feature matrix.

    A pseudocount is assumed to have already been added before calling this
    function, so all values should be positive.
    """
    log_matrix = np.log(matrix)
    return log_matrix - log_matrix.mean(axis=1, keepdims=True)


def normalize_and_transform(df, transform="log2"):
    """
    Normalize each sample to sum to one and optionally transform the profile.

    Parameters
    ----------
    df
        DataFrame with samples as rows and features as columns.
    transform
        One of {"log2", "clr", "none"}.

    Returns
    -------
    profile_df
        Row-normalized relative-abundance profile.
    transformed_df
        Transformed profile used for model input.
    """
    data = df.to_numpy(dtype=float)
    data = data + PSEUDOCOUNT

    row_sums = data.sum(axis=1, keepdims=True)
    if np.any(row_sums == 0):
        raise ValueError("At least one sample has a zero total abundance after preprocessing.")

    profile = data / row_sums

    if transform == "log2":
        transformed = np.log2(profile * 1e6 + 1)
    elif transform == "clr":
        transformed = clr_transform(profile)
    elif transform == "none":
        transformed = profile
    else:
        raise ValueError(f"Unsupported transform: {transform}")

    profile_df = pd.DataFrame(profile, index=df.index, columns=df.columns)
    transformed_df = pd.DataFrame(transformed, index=df.index, columns=df.columns)

    return profile_df, transformed_df


def write_table(df, path):
    df.to_csv(path, sep="\t", header=False, index=False)


def write_list(values, path):
    with open(path, "w") as handle:
        handle.write("\n".join(map(str, values)) + "\n")


def split_samples(samples, train_ratio, val_ratio, seed):
    samples = np.array(samples)
    np.random.RandomState(seed).shuffle(samples)

    n_samples = len(samples)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)

    train_samples = samples[:n_train]
    val_samples = samples[n_train : n_train + n_val]
    test_samples = samples[n_train + n_val :]

    return train_samples, val_samples, test_samples


def run_split(
    bact_arc_table,
    phage_table,
    outdir,
    threshold=None,
    seed=777,
    train_ratio=0.8,
    val_ratio=0.1,
    transform="log2",
):
    outdir = Path(outdir)
    os.makedirs(outdir, exist_ok=True)

    # Input format: rows are taxa/features and columns are samples.
    bact_arc_df = read_profile(bact_arc_table)
    phage_df = read_profile(phage_table)

    if threshold is not None:
        print(f"Will filter dataset based on cutoff: {threshold}")
        bact_arc_df[bact_arc_df < threshold] = 0.0
        phage_df[phage_df < threshold] = 0.0

    # Remove all-zero features.
    bact_arc_df = bact_arc_df.loc[bact_arc_df.sum(axis=1) != 0, :]
    phage_df = phage_df.loc[phage_df.sum(axis=1) != 0, :]

    # Preserve the original script behavior:
    # remove samples whose phage profiles are all zero.
    nonzero_phage_samples = phage_df.columns[phage_df.sum(axis=0) != 0]
    bact_arc_df = bact_arc_df.loc[:, bact_arc_df.columns.isin(nonzero_phage_samples)]
    phage_df = phage_df.loc[:, phage_df.columns.isin(nonzero_phage_samples)]

    samples = np.intersect1d(bact_arc_df.columns.values, phage_df.columns.values)

    if len(samples) == 0:
        raise ValueError("No overlapping samples were found between the two input tables.")

    train_samples, val_samples, test_samples = split_samples(
        samples=samples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    print(
        f"Number of samples: train={len(train_samples)}, "
        f"val={len(val_samples)}, test={len(test_samples)}"
    )

    # Output format: rows are samples and columns are taxa/features.
    bact_arc_train = bact_arc_df[train_samples].T
    bact_arc_val = bact_arc_df[val_samples].T
    bact_arc_test = bact_arc_df[test_samples].T

    phage_train = phage_df[train_samples].T
    phage_val = phage_df[val_samples].T
    phage_test = phage_df[test_samples].T

    bact_arc_train_profile, bact_arc_train_transformed = normalize_and_transform(
        bact_arc_train, transform=transform
    )
    bact_arc_val_profile, bact_arc_val_transformed = normalize_and_transform(
        bact_arc_val, transform=transform
    )
    bact_arc_test_profile, bact_arc_test_transformed = normalize_and_transform(
        bact_arc_test, transform=transform
    )

    phage_train_profile, phage_train_transformed = normalize_and_transform(
        phage_train, transform=transform
    )
    phage_val_profile, phage_val_transformed = normalize_and_transform(
        phage_val, transform=transform
    )
    phage_test_profile, phage_test_transformed = normalize_and_transform(
        phage_test, transform=transform
    )

    print(
        "Transformed matrix shapes:",
        bact_arc_train_transformed.shape,
        bact_arc_val_transformed.shape,
        bact_arc_test_transformed.shape,
        phage_train_transformed.shape,
        phage_val_transformed.shape,
        phage_test_transformed.shape,
    )

    # Transformed matrices for PHILM model input.
    write_table(bact_arc_train_transformed, outdir / "Bact_arc_train.tsv")
    write_table(bact_arc_val_transformed, outdir / "Bact_arc_val.tsv")
    write_table(bact_arc_test_transformed, outdir / "Bact_arc_test.tsv")

    write_table(phage_train_transformed, outdir / "Phage_train.tsv")
    write_table(phage_val_transformed, outdir / "Phage_val.tsv")
    write_table(phage_test_transformed, outdir / "Phage_test.tsv")

    # Row-normalized relative-abundance profiles.
    # The filenames are preserved for compatibility with the original workflow.
    write_table(bact_arc_train_profile, outdir / "Bact_arc_train_no_clr.tsv")
    write_table(bact_arc_val_profile, outdir / "Bact_arc_val_no_clr.tsv")
    write_table(bact_arc_test_profile, outdir / "Bact_arc_test_no_clr.tsv")

    write_table(phage_train_profile, outdir / "Phage_train_no_clr.tsv")
    write_table(phage_val_profile, outdir / "Phage_val_no_clr.tsv")
    write_table(phage_test_profile, outdir / "Phage_test_no_clr.tsv")

    write_list(bact_arc_train_transformed.columns.values, outdir / "Bact_arc_feature_names.txt")
    write_list(phage_train_transformed.columns.values, outdir / "Phage_feature_names.txt")

    write_list(train_samples, outdir / "train_samples.txt")
    write_list(val_samples, outdir / "val_samples.txt")
    write_list(test_samples, outdir / "test_samples.txt")

    print(f"Finished writing split data to: {outdir}")


def main():
    args = parse_args()
    validate_args(args)

    run_split(
        bact_arc_table=args.bact_arc,
        phage_table=args.phage,
        outdir=args.outdir,
        threshold=args.threshold,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        transform=args.transform,
    )

if __name__ == "__main__":
    main()
