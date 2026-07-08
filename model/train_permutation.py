#!/usr/bin/env python
"""
Train one PHILM null-permutation model with fixed hyperparameters.

This script is designed for HPC array jobs. Each run performs exactly one
sample-label permutation by shuffling the rows of the input/phage profile X
relative to the output/bacteria profile Y, then trains one MLP_NODE model using
fixed hyperparameters from the config file.

Run from the PHILM repo root, for example:

  python model/train_permutation.py \
      -c config/config_train_permutation.yaml \
      --perm-id 17

For a SLURM array, use something like:

  python model/train_permutation.py \
      -c config/config_train_permutation.yaml \
      --perm-id ${SLURM_ARRAY_TASK_ID}

The trained model can then be passed to your gradient-interaction inference
script as --model-path.
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

# When this script is placed in model/, this makes local imports robust.
SCRIPT_DIR = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(SCRIPT_DIR))

from trainer import train_model  # noqa: E402


def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh) or {}


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def ensure_same_n_rows(x: pd.DataFrame, y: pd.DataFrame, x_name: str, y_name: str) -> None:
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"{x_name} and {y_name} must have the same number of rows. "
            f"Got {x.shape[0]} and {y.shape[0]}."
        )


def make_permutation(
    n: int,
    rng: np.random.Generator,
    method: str = "shuffle",
    avoid_self_pairs: bool = True,
    cyclic_shift: Optional[int] = None,
) -> np.ndarray:
    """
    Return indices idx where X_perm = X.iloc[idx].reset_index(drop=True).

    Row k of Y is then paired with original X row idx[k].
    """
    if n <= 1:
        raise ValueError("Need at least 2 samples to make a permutation null model.")

    if method == "cyclic_shift":
        if cyclic_shift is None:
            cyclic_shift = int(rng.integers(1, n))
        cyclic_shift = int(cyclic_shift) % n
        if cyclic_shift == 0:
            cyclic_shift = 1
        return (np.arange(n) + cyclic_shift) % n

    if method != "shuffle":
        raise ValueError("permutation.method must be either 'shuffle' or 'cyclic_shift'.")

    if not avoid_self_pairs:
        return rng.permutation(n)

    # Derangement-like shuffle: avoid pairing any sample with its own X row.
    # For normal microbiome sample sizes this succeeds quickly.
    for _ in range(10000):
        idx = rng.permutation(n)
        if not np.any(idx == np.arange(n)):
            return idx

    raise RuntimeError(
        "Could not generate a self-pair-free permutation after 10000 tries. "
        "Set permutation.avoid_self_pairs: false if your sample size is very small."
    )


def apply_x_permutation(
    x: pd.DataFrame,
    y: pd.DataFrame,
    rng: np.random.Generator,
    method: str,
    avoid_self_pairs: bool,
    cyclic_shift: Optional[int] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    ensure_same_n_rows(x, y, "X", "Y")
    idx = make_permutation(
        n=x.shape[0],
        rng=rng,
        method=method,
        avoid_self_pairs=avoid_self_pairs,
        cyclic_shift=cyclic_shift,
    )
    x_perm = x.iloc[idx, :].reset_index(drop=True)
    return x_perm, idx


def format_template(template: str, perm_id: int, seed: int) -> str:
    values = {
        "perm_id": perm_id,
        "perm_id_padded": f"{perm_id:03d}",
        "seed": seed,
    }
    return template.format(**values)


def resolve_output_paths(config: Dict, perm_id: int, seed: int) -> Tuple[str, str, str, str]:
    perm_cfg = config.get("permutation", {}) or {}
    model_cfg = config.get("model", {}) or {}

    out_dir_template = perm_cfg.get(
        "out_dir_template",
        "results/permutation_null/perm_{perm_id_padded}",
    )
    out_dir = format_template(out_dir_template, perm_id=perm_id, seed=seed)

    model_path_template = model_cfg.get(
        "path_template",
        os.path.join(out_dir, "PHILM_perm_{perm_id_padded}.pth"),
    )
    para_path_template = model_cfg.get(
        "para_path_template",
        os.path.join(out_dir, "PHILM_perm_{perm_id_padded}.params.yaml"),
    )
    log_path_template = model_cfg.get(
        "log_path_template",
        os.path.join(out_dir, "PHILM_perm_{perm_id_padded}.train.log"),
    )
    map_path_template = perm_cfg.get(
        "map_path_template",
        os.path.join(out_dir, "PHILM_perm_{perm_id_padded}.sample_permutation.tsv"),
    )

    model_path = format_template(model_path_template, perm_id=perm_id, seed=seed)
    para_path = format_template(para_path_template, perm_id=perm_id, seed=seed)
    log_path = format_template(log_path_template, perm_id=perm_id, seed=seed)
    map_path = format_template(map_path_template, perm_id=perm_id, seed=seed)

    for path in [model_path, para_path, log_path, map_path]:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    return model_path, para_path, log_path, map_path


def save_permutation_map(
    map_path: str,
    train_idx: Optional[np.ndarray],
    val_idx: Optional[np.ndarray],
) -> None:
    with open(map_path, "w") as out:
        out.write("split\ty_row_after_permutation\tx_original_row\n")
        if train_idx is not None:
            for y_row, x_row in enumerate(train_idx):
                out.write(f"train\t{y_row}\t{int(x_row)}\n")
        if val_idx is not None:
            for y_row, x_row in enumerate(val_idx):
                out.write(f"val\t{y_row}\t{int(x_row)}\n")


def save_params(
    para_path: str,
    config: Dict,
    perm_id: int,
    seed: int,
    train_idx: Optional[np.ndarray],
    val_idx: Optional[np.ndarray],
    model_path: str,
    log_path: str,
    final_val_loss: float,
) -> None:
    model_cfg = config.get("model", {}) or {}
    perm_cfg = config.get("permutation", {}) or {}
    data_cfg = config.get("data", {}) or {}

    out = {
        "permutation_id": perm_id,
        "seed": seed,
        "null_model": "shuffle X/phage sample rows relative to fixed Y/bacteria rows",
        "permutation_method": perm_cfg.get("method", "shuffle"),
        "avoid_self_pairs": bool(perm_cfg.get("avoid_self_pairs", True)),
        "permute_train_X": train_idx is not None,
        "permute_val_X": val_idx is not None,
        "train_X": data_cfg.get("train_X"),
        "train_Y": data_cfg.get("train_Y"),
        "val_X": data_cfg.get("val_X"),
        "val_Y": data_cfg.get("val_Y"),
        "model_path": model_path,
        "log_path": log_path,
        "final_val_loss_returned_by_train_model": float(final_val_loss),
        "batch_size": int(model_cfg["batch_size"]),
        "patience": int(model_cfg["patience"]),
        "num_epochs": int(model_cfg["num_epochs"]),
        "hidden_size": int(model_cfg["hidden_size"]),
        "learning_rate": float(model_cfg["learning_rate"]),
        "weight_decay": float(model_cfg["weight_decay"]),
    }

    with open(para_path, "w") as fh:
        yaml.safe_dump(out, fh, sort_keys=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one fixed-hyperparameter PHILM permutation-null model."
    )
    parser.add_argument("-c", "--config", required=True, help="Permutation training config YAML.")
    parser.add_argument(
        "--perm-id",
        type=int,
        default=None,
        help="Permutation ID. Overrides permutation.id in config. Use SLURM_ARRAY_TASK_ID here.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Exact random seed. If omitted, seed = permutation.base_seed + perm_id.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="auto, cpu, cuda, cuda:0, etc. Overrides runtime.device in config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    data_cfg = config["data"]
    model_cfg = config["model"]
    perm_cfg = config.get("permutation", {}) or {}
    runtime_cfg = config.get("runtime", {}) or {}

    perm_id = args.perm_id
    if perm_id is None:
        perm_id = int(perm_cfg.get("id", 0))

    if args.seed is not None:
        seed = int(args.seed)
    else:
        seed = int(perm_cfg.get("base_seed", 19910728)) + int(perm_id)

    set_global_seed(seed)
    rng = np.random.default_rng(seed)

    device_arg = args.device or runtime_cfg.get("device", "auto")
    device = choose_device(device_arg)

    print(f"Using device: {device}", flush=True)
    print(f"Permutation ID: {perm_id}", flush=True)
    print(f"Seed: {seed}", flush=True)

    train_X_data = pd.read_table(data_cfg["train_X"], header=None)
    train_Y_data = pd.read_table(data_cfg["train_Y"], header=None)
    val_X_data = pd.read_table(data_cfg["val_X"], header=None)
    val_Y_data = pd.read_table(data_cfg["val_Y"], header=None)

    ensure_same_n_rows(train_X_data, train_Y_data, "train_X", "train_Y")
    ensure_same_n_rows(val_X_data, val_Y_data, "val_X", "val_Y")

    method = perm_cfg.get("method", "shuffle")
    avoid_self_pairs = bool(perm_cfg.get("avoid_self_pairs", True))
    permute_train = bool(perm_cfg.get("permute_train_X", True))
    permute_val = bool(perm_cfg.get("permute_val_X", True))

    train_idx = None
    val_idx = None

    if permute_train:
        train_X_data, train_idx = apply_x_permutation(
            x=train_X_data,
            y=train_Y_data,
            rng=rng,
            method=method,
            avoid_self_pairs=avoid_self_pairs,
        )

    if permute_val:
        val_X_data, val_idx = apply_x_permutation(
            x=val_X_data,
            y=val_Y_data,
            rng=rng,
            method=method,
            avoid_self_pairs=avoid_self_pairs,
        )

    model_path, para_path, log_path, map_path = resolve_output_paths(
        config=config,
        perm_id=perm_id,
        seed=seed,
    )

    save_permutation_map(map_path, train_idx=train_idx, val_idx=val_idx)

    print(f"Training X shape: {train_X_data.shape}", flush=True)
    print(f"Training Y shape: {train_Y_data.shape}", flush=True)
    print(f"Validation X shape: {val_X_data.shape}", flush=True)
    print(f"Validation Y shape: {val_Y_data.shape}", flush=True)
    print(f"Model output path: {model_path}", flush=True)
    print(f"Log path: {log_path}", flush=True)
    print(f"Permutation map path: {map_path}", flush=True)

    final_val_loss = train_model(
        train_X_data=train_X_data,
        train_Y_data=train_Y_data,
        val_X_data=val_X_data,
        val_Y_data=val_Y_data,
        device=device,
        num_epochs=int(model_cfg["num_epochs"]),
        batch_size=int(model_cfg["batch_size"]),
        learning_rate=float(model_cfg["learning_rate"]),
        hidden_size=int(model_cfg["hidden_size"]),
        patience=int(model_cfg["patience"]),
        weight_decay=float(model_cfg["weight_decay"]),
        model_path=model_path,
        log_path=log_path,
    )

    save_params(
        para_path=para_path,
        config=config,
        perm_id=perm_id,
        seed=seed,
        train_idx=train_idx,
        val_idx=val_idx,
        model_path=model_path,
        log_path=log_path,
        final_val_loss=final_val_loss,
    )

    print(f"Saved permutation parameters: {para_path}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
