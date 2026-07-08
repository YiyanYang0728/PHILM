#!/usr/bin/env python
"""
Infer phage -> bacteria signed interaction scores from a trained PHILM model
using gradient-based sensitivity.

The main score is computed as either:

  grad_x_input: mean_samples( x_i * d y_j / d x_i )
  gradient:     mean_samples(       d y_j / d x_i )

where x_i is input phage feature i and y_j is predicted bacterial feature j.

Output columns:
  phage    bacteria    raw_score    normalized_score

Optional phage-wise normalization can be applied after the interaction
score matrix is computed, so it does not change the model input profiles.

Run from the PHILM repo root, for example:

  python scripts/infer_gradient_interactions.py \
      --data-dir data \
      --split train \
      --model-path results/PHILM_best_model.pth \
      --best-params results/PHILM_best_params.yaml \
      --out results/PHILM_gradient_signed_interactions.train.tsv

If you already have a merged config containing model.hidden_size, you can use:

  python scripts/infer_gradient_interactions.py \
      -c my_config_test.train.yaml \
      --split train \
      --out results/PHILM_gradient_signed_interactions.train.tsv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml


# Allow running as: python scripts/infer_gradient_interactions.py from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "model"
sys.path.insert(0, str(MODEL_DIR))

from model import MLP_NODE  # noqa: E402
from pathlib import Path

def resolve_path(path):
    return Path(path).expanduser().resolve()

def load_yaml(path: Optional[str]) -> Dict:
    if not path:
        return {}
    with open(path, "r") as fh:
        loaded = yaml.safe_load(fh)
    return loaded or {}


def read_names(path: str) -> List[str]:
    with open(path, "r") as fh:
        names = [line.rstrip("\n") for line in fh]
    # Preserve names exactly except for final newline. Drop truly empty trailing lines.
    return [name for name in names if name != ""]


def get_from_config_data(config: Dict, split: str) -> Optional[str]:
    data_cfg = config.get("data", {}) or {}
    candidates = []
    if split == "train":
        candidates = ["train_X", "test_X"]
    elif split in {"val", "validation"}:
        candidates = ["val_X", "validation_X", "test_X"]
    elif split == "test":
        candidates = ["test_X"]
    for key in candidates:
        if key in data_cfg:
            return data_cfg[key]
    return None


def default_profile_paths(data_dir: str, split: str, input_prefix: str, use_no_clr: bool) -> List[str]:
    suffix = "_no_clr.tsv" if use_no_clr else ".tsv"
    if split == "all":
        return [
            os.path.join(data_dir, f"{input_prefix}_train{suffix}"),
            os.path.join(data_dir, f"{input_prefix}_val{suffix}"),
            os.path.join(data_dir, f"{input_prefix}_test{suffix}"),
        ]
    if split == "validation":
        split = "val"
    return [os.path.join(data_dir, f"{input_prefix}_{split}{suffix}")]


def load_input_matrix(paths: Iterable[str]) -> np.ndarray:
    matrices = []
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input profile not found: {path}")
        arr = pd.read_table(path, header=None).to_numpy(dtype=np.float32)
        matrices.append(arr)
    if len(matrices) == 1:
        return matrices[0]
    return np.vstack(matrices)


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_model(
    model_path: str,
    input_dim: int,
    output_dim: int,
    hidden_size: int,
    device: torch.device,
) -> MLP_NODE:
    model = MLP_NODE(
        input_dim=input_dim,
        hidden_size=hidden_size,
        output_dim=output_dim,
    )

    try:
        state = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(model_path, map_location=device)

    # This repo saves model.state_dict(), but support common wrappers too.
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def infer_signed_gradient_scores(
    model: MLP_NODE,
    X: np.ndarray,
    batch_size: int,
    score_mode: str,
    device: torch.device,
    progress_every: int = 1,
) -> np.ndarray:
    """
    Return a matrix with shape [n_input_features, n_output_features].
    Rows are phage features; columns are bacterial features.
    """
    if score_mode not in {"grad_x_input", "gradient"}:
        raise ValueError("score_mode must be 'grad_x_input' or 'gradient'")

    n_samples, n_input = X.shape

    with torch.no_grad():
        x_probe = torch.tensor(X[:1], dtype=torch.float32, device=device)
        n_output = model(x_probe).shape[1]

    signed_sum = torch.zeros((n_input, n_output), dtype=torch.float32, device=device)
    total_n = 0

    # Important: do not wrap this section in torch.no_grad(); we need autograd.
    for batch_idx, start in enumerate(range(0, n_samples, batch_size), start=1):
        stop = min(start + batch_size, n_samples)
        xb = torch.tensor(X[start:stop], dtype=torch.float32, device=device)
        xb.requires_grad_(True)

        yb = model(xb)
        bsz = xb.shape[0]
        total_n += bsz

        n_output = yb.shape[1]
        for j in range(n_output):
            # Since samples are independent in the model, summing y[:, j] gives
            # per-sample gradients dy[b, j] / dx[b, i] in grad[b, i].
            target = yb[:, j].sum()
            retain_graph = j < (n_output - 1)
            grad = torch.autograd.grad(
                target,
                xb,
                retain_graph=retain_graph,
                create_graph=False,
                allow_unused=False,
            )[0]

            if score_mode == "grad_x_input":
                attr = grad * xb
            else:
                attr = grad

            signed_sum[:, j] += attr.sum(dim=0)

        del xb, yb
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if progress_every > 0 and batch_idx % progress_every == 0:
            print(
                f"Processed samples {start + 1}-{stop} / {n_samples}",
                file=sys.stderr,
                flush=True,
            )

    signed_score = signed_sum / float(total_n)
    return signed_score.detach().cpu().numpy()



def normalize_scores_by_phage(
    score: np.ndarray,
    mode: str = "none",
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Normalize interaction scores across all bacterial outputs separately for
    each input phage feature.

    score shape: [n_phage, n_bacteria]

    Modes
    -----
    none:
        Return scores unchanged.

    signed_zscore:
        For each phage i:
            z_ij = (score_ij - mean_j(score_ij)) / std_j(score_ij)
        This makes each phage's signed interactions comparable across bacteria,
        but the normalized sign means above/below that phage's average score.

    abs_zscore_keep_sign:
        For each phage i, z-score the absolute interaction strengths, then
        restore the original direction:
            z_ij = sign(score_ij) * (abs(score_ij) - mean_j(abs(score_ij))) / std_j(abs(score_ij))
        This is useful when you want phage-wise comparable strength while
        preserving the original positive/negative direction.
    """
    if mode == "none":
        return score

    if mode not in {"signed_zscore", "abs_zscore_keep_sign"}:
        raise ValueError("mode must be one of: none, signed_zscore, abs_zscore_keep_sign")

    score = score.astype(np.float32, copy=False)

    if mode == "signed_zscore":
        values = score
        sign = None
    else:
        values = np.abs(score)
        sign = np.sign(score)

    row_mean = values.mean(axis=1, keepdims=True)
    row_std = values.std(axis=1, ddof=0, keepdims=True)

    normalized = np.zeros_like(values, dtype=np.float32)
    valid_rows = (row_std[:, 0] > eps)
    normalized[valid_rows, :] = (
        (values[valid_rows, :] - row_mean[valid_rows, :]) /
        (row_std[valid_rows, :] + eps)
    )

    if sign is not None:
        normalized = normalized * sign

    return normalized


def write_long_table(
    raw_score: np.ndarray,
    normalized_score: np.ndarray,
    input_names: List[str],
    output_names: List[str],
    out_path: str,
    input_col: str = "phage",
    output_col: str = "bacteria",
    header: bool = True,
    sort_by: str = "none",
    top_n: Optional[int] = None,
    min_abs_score: Optional[float] = None,
    rank_score: str = "normalized",
    float_format: str = ".8g",
) -> None:
    """
    Write a long-format table with both raw and normalized scores.

    Sorting, top-N selection, and min_abs_score filtering are applied to
    rank_score, which can be either "normalized" or "raw".
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if raw_score.shape != normalized_score.shape:
        raise ValueError(
            f"raw_score shape {raw_score.shape} != normalized_score shape {normalized_score.shape}"
        )

    n_input, n_output = raw_score.shape
    if len(input_names) != n_input:
        raise ValueError(f"Input name count ({len(input_names)}) != score rows ({n_input})")
    if len(output_names) != n_output:
        raise ValueError(f"Output name count ({len(output_names)}) != score columns ({n_output})")

    if rank_score == "normalized":
        ranking_matrix = normalized_score
    elif rank_score == "raw":
        ranking_matrix = raw_score
    else:
        raise ValueError("rank_score must be one of: normalized, raw")

    flat = ranking_matrix.ravel()

    if sort_by == "none":
        indices = range(flat.size)
    elif sort_by == "abs":
        order = np.argsort(-np.abs(flat))
        indices = order[:top_n] if top_n is not None else order
    elif sort_by == "signed":
        order = np.argsort(-flat)
        indices = order[:top_n] if top_n is not None else order
    else:
        raise ValueError("sort_by must be one of: none, abs, signed")

    written = 0
    with open(out_path, "w") as out:
        if header:
            out.write(f"{input_col}\t{output_col}\traw_score\tnormalized_score\n")

        for flat_idx in indices:
            i = int(flat_idx) // n_output
            j = int(flat_idx) % n_output
            rank_val = float(ranking_matrix[i, j])
            raw_val = float(raw_score[i, j])
            normalized_val = float(normalized_score[i, j])

            if min_abs_score is not None and abs(rank_val) < min_abs_score:
                continue

            out.write(
                f"{input_names[i]}\t{output_names[j]}\t"
                f"{format(raw_val, float_format)}\t"
                f"{format(normalized_val, float_format)}\n"
            )
            written += 1

            if sort_by == "none" and top_n is not None and written >= top_n:
                break

    print(f"Wrote {written} interaction rows to: {out_path}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer phage-bacteria signed scores using gradients from a trained PHILM model."
    )

    parser.add_argument("-c", "--config", default=None,
                        help="Optional YAML config. Useful for model path, batch size, hidden_size, and data path.")
    parser.add_argument("--best-params", default="results/PHILM_best_params.yaml",
                        help="YAML containing hidden_size, usually results/PHILM_best_params.yaml.")
    parser.add_argument("--model-path", default=None,
                        help="Trained model state_dict path. Overrides config model.path.")
    parser.add_argument("--hidden-size", type=int, default=None,
                        help="Hidden size used during training. Overrides config/best-params.")

    parser.add_argument("--data-dir", default="data", help="Data directory containing profile/name files.")
    parser.add_argument("--split", default="train", choices=["train", "val", "validation", "test", "all"],
                        help="Which normalized profile to use for attribution. Default: train.")
    parser.add_argument("--input-profile", default=None,
                        help="Optional explicit input profile path. If set, --split is ignored for the profile path.")
    parser.add_argument("--use-no-clr", action="store_true",
                        help="Use *_no_clr.tsv files when constructing profile path from --split.")

    parser.add_argument("--input-prefix", default="Phage",
                        help="Input profile prefix. Default: Phage.")
    parser.add_argument("--output-prefix", default="Bact_arc",
                        help="Output feature-name prefix. Default: Bact_arc.")
    parser.add_argument("--input-feature-names", default=None,
                        help="Input feature names. Default: data/Phage_feature_names.txt.")
    parser.add_argument("--output-feature-names", default=None,
                        help="Output feature names. Default: data/Bact_arc_feature_names.txt.")

    parser.add_argument("--score-mode", default="gradient", choices=["gradient", "grad_x_input"],
                        help="gradient = mean(dy_j/dx_i). grad_x_input = mean(x_i * dy_j/dx_i); Default: gradient.")
    parser.add_argument("--phage-normalize", default="signed_zscore",
                        choices=["signed_zscore", "abs_zscore_keep_sign", "none"],
                        help=(
                            "Normalize final interaction scores across bacteria separately for each phage. "
                            "none = no normalization. signed_zscore = (score - phage_mean) / phage_std. "
                            "abs_zscore_keep_sign = z-score abs(score) per phage, then restore original sign. "
                            "Default: signed_zscore."
                        ))
    parser.add_argument("--phage-normalize-eps", type=float, default=1e-12,
                        help="Small value used to avoid division by zero during phage-wise normalization. Default: 1e-12.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size for gradient inference. If omitted, uses config model.batch_size, otherwise 16.")
    parser.add_argument("--device", default="auto",
                        help="auto, cpu, cuda, cuda:0, etc. Default: auto.")

    parser.add_argument("--out", default="results/PHILM_gradient_signed_interactions.tsv",
                        help="Output TSV path.")
    parser.add_argument("--sort-by", default="none", choices=["none", "abs", "signed"],
                        help="Sort output rows. Use abs for strongest interactions first. Default: none.")
    parser.add_argument("--rank-score", default="normalized", choices=["normalized", "raw"],
                        help=(
                            "Score column used for --sort-by, --top-n, and --min-abs-score. "
                            "Default: normalized, which preserves the previous behavior when "
                            "--phage-normalize is not none."
                        ))
    parser.add_argument("--top-n", type=int, default=None,
                        help="Only write the top N rows after sorting, or first N rows if --sort-by none.")
    parser.add_argument("--min-abs-score", type=float, default=None,
                        help="Only write rows with abs(selected ranking score) >= this value.")
    parser.add_argument("--no-header", action="store_true", help="Write output without a header line.")
    parser.add_argument("--progress-every", type=int, default=1,
                        help="Print progress every N batches to stderr. Use 0 to disable.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    best_params = load_yaml(args.best_params) if args.best_params and os.path.exists(args.best_params) else {}

    base_dir = REPO_ROOT
    data_dir = resolve_path(args.data_dir)

    model_path = args.model_path or (config.get("model", {}) or {}).get("path")
    if model_path is None:
        model_path = "results/PHILM_best_model.pth"
    model_path = resolve_path(model_path)

    hidden_size = args.hidden_size
    if hidden_size is None:
        hidden_size = (config.get("model", {}) or {}).get("hidden_size")
    if hidden_size is None:
        hidden_size = best_params.get("hidden_size")
    if hidden_size is None:
        raise ValueError(
            "Could not determine hidden_size. Provide --hidden-size, use a config with model.hidden_size, "
            "or provide --best-params results/PHILM_best_params.yaml."
        )
    hidden_size = int(hidden_size)

    batch_size = args.batch_size
    if batch_size is None:
        batch_size = (config.get("model", {}) or {}).get("batch_size", 16)
    batch_size = int(batch_size)

    if args.input_profile is not None:
        input_paths = [resolve_path(args.input_profile)]
    else:
        cfg_profile = get_from_config_data(config, args.split) if args.config else None
        if cfg_profile is not None and args.split != "all":
            input_paths = [resolve_path(cfg_profile)]
        else:
            input_paths = [resolve_path(p) for p in default_profile_paths(
                data_dir=data_dir,
                split=args.split,
                input_prefix=args.input_prefix,
                use_no_clr=args.use_no_clr,
            )]

    input_feature_names = args.input_feature_names or os.path.join(data_dir, f"{args.input_prefix}_feature_names.txt")
    output_feature_names = args.output_feature_names or os.path.join(data_dir, f"{args.output_prefix}_feature_names.txt")
    input_feature_names = resolve_path(input_feature_names)
    output_feature_names = resolve_path(output_feature_names)

    input_names = read_names(input_feature_names)
    output_names = read_names(output_feature_names)
    X = load_input_matrix(input_paths)

    if X.shape[1] != len(input_names):
        raise ValueError(
            f"Input profile has {X.shape[1]} columns, but {input_feature_names} has {len(input_names)} names."
        )

    device = choose_device(args.device)
    print(f"Using device: {device}", file=sys.stderr)
    print(f"Input profile path(s): {', '.join(map(str, input_paths))}", file=sys.stderr)    
    print(f"Input matrix shape: {X.shape[0]} samples x {X.shape[1]} features", file=sys.stderr)
    print(f"Output features: {len(output_names)}", file=sys.stderr)
    print(f"Model path: {model_path}", file=sys.stderr)
    print(f"Hidden size: {hidden_size}", file=sys.stderr)
    print(f"Score mode: {args.score_mode}", file=sys.stderr)
    print(f"Phage-wise normalization: {args.phage_normalize}", file=sys.stderr)

    model = load_model(
        model_path=model_path,
        input_dim=X.shape[1],
        output_dim=len(output_names),
        hidden_size=hidden_size,
        device=device,
    )

    raw_score = infer_signed_gradient_scores(
        model=model,
        X=X,
        batch_size=batch_size,
        score_mode=args.score_mode,
        device=device,
        progress_every=args.progress_every,
    )

    normalized_score = normalize_scores_by_phage(
        score=raw_score,
        mode=args.phage_normalize,
        eps=args.phage_normalize_eps,
    )

    out_path = resolve_path(args.out)
    write_long_table(
        raw_score=raw_score,
        normalized_score=normalized_score,
        input_names=input_names,
        output_names=output_names,
        out_path=out_path,
        input_col="phage",
        output_col="bacteria",
        header=not args.no_header,
        sort_by=args.sort_by,
        top_n=args.top_n,
        min_abs_score=args.min_abs_score,
        rank_score=args.rank_score,
    )


if __name__ == "__main__":
    main()


