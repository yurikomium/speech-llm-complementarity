"""
nway_fusion.py  –  N-way late fusion of prediction CSVs

Generalizes llm_audio_fusion.py to accept an arbitrary number of prediction
CSVs and fuse them using leave-one-fold-out optimization.

All fusion operates in the 1-9 score scale (same as llm_audio_fusion.py).

Usage
-----
  python nway_fusion.py \
    --pred-csvs t1:log/.../results_sentt5.csv a1:log/.../results_hubert.csv \
    --methods simple_avg weighted_avg \
    --speaker female \
    --output-dir log/modality_comparison/fusion_2way/sess2/F2M/t1+a1

  # 3-way example
  python nway_fusion.py \
    --pred-csvs t1:path1 a1:path2 l1:path3 \
    --methods simple_avg weighted_avg \
    --speaker female \
    --output-dir log/.../t1+a1+l1
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd


# ── Per-participant CCC used for weight selection ────────────────────────────

def _per_participant_ccc(y_true, y_pred, participant_ids):
    grouped = defaultdict(lambda: ([], []))
    for i, pid in enumerate(participant_ids):
        grouped[pid][0].append(y_true[i])
        grouped[pid][1].append(y_pred[i])

    cccs = []
    for trues, preds in grouped.values():
        if len(trues) < 2:
            continue
        t = np.array(trues)
        p = np.array(preds)
        mu_t, mu_p = t.mean(), p.mean()
        sig_t, sig_p = t.std(), p.std()
        if sig_t == 0.0:
            continue
        if sig_p == 0.0:
            cccs.append(0.0)
            continue
        r = np.corrcoef(t, p)[0, 1]
        denom = sig_t**2 + sig_p**2 + (mu_t - mu_p)**2
        if denom == 0.0:
            continue
        cccs.append(2.0 * r * sig_t * sig_p / denom)

    return float(np.mean(cccs)) if cccs else float("nan")


# ── Fusion methods ────────────────────────────────────────────────────────────

def _optimize_simple_avg(pred_arrays, y_true, pids):
    return {}


def _apply_simple_avg(pred_arrays, params):
    return np.mean(list(pred_arrays.values()), axis=0)


def _optimize_weighted_avg(pred_arrays, y_true, pids):
    """
        For N=2: 21-point grid search over the first predictor's weight.
    For N>=3: scipy SLSQP with simplex constraint (sum=1, all>=0).
    """
    labels = list(pred_arrays.keys())
    N = len(labels)
    preds_matrix = np.column_stack([pred_arrays[k] for k in labels])

    if N == 2:
        best_w, best_ccc = 0.5, -float("inf")
        for i in range(21):
            w = i / 20.0
            fused = w * preds_matrix[:, 0] + (1.0 - w) * preds_matrix[:, 1]
            c = _per_participant_ccc(y_true, fused, pids)
            if not np.isnan(c) and c > best_ccc:
                best_ccc = c
                best_w = w
        weights = {labels[0]: round(best_w, 4), labels[1]: round(1.0 - best_w, 4)}
        return {"weights": weights, "val_ccc": round(best_ccc, 6)}
    else:
        from scipy.optimize import minimize

        def neg_ccc(w):
            fused = preds_matrix @ w
            return -_per_participant_ccc(y_true, fused, pids)

        w0 = np.ones(N) / N
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        bounds = [(0.0, 1.0)] * N
        result = minimize(neg_ccc, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints,
                          options={"maxiter": 500, "ftol": 1e-8})
        w_opt = result.x
        fused = preds_matrix @ w_opt
        val_ccc = _per_participant_ccc(y_true, fused, pids)
        weights = {labels[i]: round(float(w_opt[i]), 4) for i in range(N)}
        return {"weights": weights, "val_ccc": round(val_ccc, 6)}


def _apply_weighted_avg(pred_arrays, params):
    weights = params["weights"]
    result = np.zeros(len(next(iter(pred_arrays.values()))))
    for label, w in weights.items():
        result += w * pred_arrays[label]
    return result


_METHODS = {
    "simple_avg":   (_optimize_simple_avg,   _apply_simple_avg),
    "weighted_avg": (_optimize_weighted_avg,  _apply_weighted_avg),
}
ALL_METHODS = tuple(_METHODS.keys())

_REQUIRED_PREDICTION_COLUMNS = {
    "pair_key", "y_true", "y_pred", "fold_group", "female_id", "male_id",
}
# Experiment-time CSVs recorded 13-item means at six decimal places in some
# paths and at full precision in others.  This admits only that serialization
# difference while still rejecting a substantively different target label.
_Y_TRUE_MATCH_ATOL = 1e-6


def _load_prediction_sources(pred_specs: list[str]) -> dict[str, pd.DataFrame]:
    """Load prediction CSVs and fail fast on ambiguous or invalid inputs."""
    if len(pred_specs) < 2:
        raise ValueError("Need at least 2 prediction CSVs for fusion")

    source_dfs: dict[str, pd.DataFrame] = {}
    for spec in pred_specs:
        if ":" not in spec:
            raise ValueError(f"Expected label:path format, got: {spec!r}")
        label, path = spec.split(":", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Prediction label must not be empty: {spec!r}")
        if label in source_dfs:
            raise ValueError(f"Duplicate prediction label: {label!r}")

        df = pd.read_csv(path)
        missing = _REQUIRED_PREDICTION_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"Prediction CSV {path!r} is missing columns: {sorted(missing)}"
            )
        if df.empty:
            raise ValueError(f"Prediction CSV {path!r} is empty")

        duplicate_pairs = df.loc[df["pair_key"].duplicated(), "pair_key"]
        if not duplicate_pairs.empty:
            examples = sorted(duplicate_pairs.astype(str).unique())[:5]
            raise ValueError(
                f"Prediction CSV {path!r} has duplicate pair_key values: "
                f"{examples}"
            )

        for col in ("y_true", "y_pred", "fold_group"):
            numeric = pd.to_numeric(df[col], errors="coerce")
            if not np.isfinite(numeric.to_numpy(dtype=float)).all():
                raise ValueError(
                    f"Prediction CSV {path!r} has non-finite {col} values"
                )
            df[col] = numeric
        if not np.equal(df["fold_group"], np.round(df["fold_group"])).all():
            raise ValueError(
                f"Prediction CSV {path!r} has non-integer fold_group values"
            )
        df["fold_group"] = df["fold_group"].astype(int)

        for col in ("pair_key", "female_id", "male_id"):
            values = df[col].astype("string")
            if values.isna().any() or values.str.strip().eq("").any():
                raise ValueError(
                    f"Prediction CSV {path!r} has missing {col} values"
                )
            df[col] = values.astype(str)

        source_dfs[label] = df
        print(f"  {label}: {path}  ({len(df)} pairs)")

    if not any("kaisu" in df.columns for df in source_dfs.values()):
        raise ValueError(
            "At least one prediction CSV must contain the kaisu column"
        )
    return source_dfs


def _merge_prediction_sources(
    source_dfs: dict[str, pd.DataFrame],
    allow_kaisu_mismatch: bool = False,
) -> pd.DataFrame:
    """Join sources after verifying identical pair sets and metadata."""
    labels = list(source_dfs)
    first = labels[0]
    reference_keys = set(source_dfs[first]["pair_key"])
    for label in labels[1:]:
        label_keys = set(source_dfs[label]["pair_key"])
        if label_keys != reference_keys:
            only_first = sorted(reference_keys - label_keys)[:5]
            only_label = sorted(label_keys - reference_keys)[:5]
            raise ValueError(
                f"pair_key set mismatch between {first!r} and {label!r}: "
                f"only in {first!r}={only_first}, only in {label!r}={only_label}"
            )
    merged = source_dfs[first][["pair_key"]].copy()
    for label, df in source_dfs.items():
        cols = {
            "pair_key": "pair_key",
            "y_pred": f"y_pred_{label}",
            "y_true": f"y_true_{label}",
            "fold_group": f"fold_group_{label}",
            "female_id": f"female_id_{label}",
            "male_id": f"male_id_{label}",
            "kaisu": f"kaisu_{label}",
        }
        renamed = df.rename(
            columns={key: value for key, value in cols.items() if key in df.columns}
        )
        selected = ["pair_key"] + [
            value
            for key, value in cols.items()
            if key != "pair_key" and key in df.columns
        ]
        merged = merged.merge(renamed[selected], on="pair_key", how="inner")

    if merged.empty:
        raise ValueError("Prediction CSVs have no common pair_key values")

    for label in labels[1:]:
        truth_matches = np.isclose(
            merged[f"y_true_{first}"].to_numpy(dtype=float),
            merged[f"y_true_{label}"].to_numpy(dtype=float),
            rtol=0.0,
            atol=_Y_TRUE_MATCH_ATOL,
        )
        if not truth_matches.all():
            examples = merged.loc[~truth_matches, "pair_key"].tolist()[:5]
            raise ValueError(
                f"y_true mismatch between {first!r} and {label!r}: {examples}"
            )

        for col in ("fold_group", "female_id", "male_id"):
            matches = (
                merged[f"{col}_{first}"].astype(str)
                == merged[f"{col}_{label}"].astype(str)
            )
            if not matches.all():
                examples = merged.loc[~matches, "pair_key"].tolist()[:5]
                raise ValueError(
                    f"{col} mismatch between {first!r} and {label!r}: "
                    f"{examples}"
                )

    kaisu_labels = [
        label for label in labels if f"kaisu_{label}" in merged.columns
    ]
    kaisu_reference = kaisu_labels[0]
    for label in kaisu_labels[1:]:
        matches = (
            merged[f"kaisu_{kaisu_reference}"].astype(str)
            == merged[f"kaisu_{label}"].astype(str)
        )
        if not matches.all():
            examples = merged.loc[~matches, "pair_key"].tolist()[:5]
            message = (
                f"kaisu mismatch between {kaisu_reference!r} and "
                f"{label!r}: {examples}"
            )
            if not allow_kaisu_mismatch:
                raise ValueError(
                    message
                    + "; pass --allow-kaisu-mismatch only for known legacy "
                    "metadata and place the canonical source first"
                )
            print(
                "WARNING: " + message + "; using kaisu from the first "
                "available source",
                file=sys.stderr,
            )

    return merged


def _build_selection_mask(
    fold_groups: np.ndarray,
    female_ids: np.ndarray,
    male_ids: np.ndarray,
    test_fold: int,
    strict_participant_disjoint: bool = False,
) -> np.ndarray:
    """Return the weight-selection rows for one held-out test fold."""
    test_mask = fold_groups == test_fold
    selection_mask = ~test_mask
    if strict_participant_disjoint:
        test_participants = set(female_ids[test_mask]) | set(male_ids[test_mask])
        selection_mask &= ~np.isin(female_ids, list(test_participants))
        selection_mask &= ~np.isin(male_ids, list(test_participants))
    return selection_mask


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="N-way late fusion of prediction CSVs"
    )
    p.add_argument("--pred-csvs", type=str, nargs="+", required=True,
                   help="label:path pairs (e.g. t1:path/to/results.csv)")
    p.add_argument("--methods", type=str, nargs="+",
                   default=list(ALL_METHODS),
                   help=f"Fusion methods (default: {ALL_METHODS})")
    p.add_argument("--speaker", type=str, required=True,
                   choices=["female", "male"])
    p.add_argument(
        "--strict-participant-disjoint",
        action="store_true",
        help=(
            "Sensitivity protocol: exclude every row involving a test-fold "
            "participant from that fold's weight-selection pool."
        ),
    )
    p.add_argument(
        "--allow-kaisu-mismatch",
        action="store_true",
        help=(
            "Compatibility mode for archived predictions with the known "
            "legacy kaisu/conv metadata bug. Other metadata must still match; "
            "kaisu is taken from the first source that provides it."
        ),
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Allow a reduced fold set for synthetic/offline smoke testing.",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    from evaluation_metrics import compute_metrics, print_all_metrics

    for m in args.methods:
        if m not in _METHODS:
            raise ValueError(f"Unknown method: {m!r}. Available: {ALL_METHODS}")

    id_col = "female_id" if args.speaker == "female" else "male_id"
    partner_col = "male_id" if args.speaker == "female" else "female_id"

    # ── Parse, load, and validate CSVs ───────────────────────────────────
    source_dfs = _load_prediction_sources(args.pred_csvs)
    labels = list(source_dfs.keys())
    merged = _merge_prediction_sources(
        source_dfs,
        allow_kaisu_mismatch=args.allow_kaisu_mismatch,
    )

    n_common = len(merged)
    print(f"\nCommon pairs: {n_common}")

    # Extract arrays
    first = labels[0]
    y_true = merged[f"y_true_{first}"].values
    fold_groups = merged[f"fold_group_{first}"].values.astype(int)
    id_col_key = f"{id_col}_{first}"
    pids = merged[id_col_key].values if id_col_key in merged.columns else None
    female_ids = merged[f"female_id_{first}"].values
    male_ids = merged[f"male_id_{first}"].values

    pred_arrays_all = {label: merged[f"y_pred_{label}"].values for label in labels}

    folds = sorted(set(fold_groups))
    expected_folds = list(range(1, 26))
    if not args.smoke and folds != expected_folds:
        raise ValueError(
            "Fusion requires all 25 paper folds; "
            f"found {folds}"
        )
    print(f"Folds: {len(folds)}  ({min(folds)}-{max(folds)})")
    print(f"Methods: {', '.join(args.methods)}\n")
    protocol = (
        "strict participant-disjoint sensitivity"
        if args.strict_participant_disjoint
        else "pooled held-out predictions from all non-test folds"
    )
    print(f"Weight-selection protocol: {protocol}\n")

    # ── Leave-one-fold-out fusion ────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fusion_param_log = []
    method_predictions = {m: np.empty(n_common) for m in args.methods}

    for fold in folds:
        test_mask = fold_groups == fold
        selection_mask = _build_selection_mask(
            fold_groups,
            female_ids,
            male_ids,
            fold,
            args.strict_participant_disjoint,
        )

        selection_preds = {
            k: v[selection_mask] for k, v in pred_arrays_all.items()
        }
        selection_y = y_true[selection_mask]
        selection_pids = pids[selection_mask] if pids is not None else None

        test_preds = {k: v[test_mask] for k, v in pred_arrays_all.items()}

        for method in args.methods:
            opt_fn, apply_fn = _METHODS[method]
            params = opt_fn(selection_preds, selection_y, selection_pids)
            fused = apply_fn(test_preds, params)
            fused = np.clip(fused, 1.0, 9.0)
            method_predictions[method][test_mask] = fused

            loggable = {
                k: v for k, v in params.items() if not k.startswith("_")
            }
            fusion_param_log.append((fold, method, loggable))

    # ── Save results ─────────────────────────────────────────────────────
    kaisu_col = next(
        f"kaisu_{label}"
        for label in labels
        if f"kaisu_{label}" in merged.columns
    )
    base_cols = {
        "fold_group": fold_groups,
        "pair_key": merged["pair_key"].values,
        "y_true": y_true,
        "female_id": female_ids,
        "male_id": male_ids,
        "kaisu": merged[kaisu_col].values,
    }

    eval_results = {}

    # Single-source baselines
    for label in labels:
        df_single = pd.DataFrame({**base_cols, "y_pred": pred_arrays_all[label]})
        eval_results[label] = {
            "df": df_single, "id_col": id_col, "partner_col": partner_col}

    # Fusion results
    for method in args.methods:
        df_out = pd.DataFrame({
            **base_cols,
            "y_pred": method_predictions[method],
        })
        csv_path = args.output_dir / f"fusion_{method}.csv"
        df_out.to_csv(csv_path, index=False)
        eval_results[f"fusion_{method}"] = {
            "df": df_out, "id_col": id_col, "partner_col": partner_col}

    print_all_metrics(eval_results)

    # ── result.txt ───────────────────────────────────────────────────────
    result_path = args.output_dir / "result.txt"
    lines = [
        "=" * 60,
        f"N-way Fusion: {' + '.join(labels)}",
        f"Pairs    : {n_common}",
        f"Speaker  : {args.speaker}  (id_col={id_col})",
        f"Methods  : {', '.join(args.methods)}",
        "=" * 60,
    ]
    for name, info in eval_results.items():
        metrics = compute_metrics(info["df"], id_col)
        lines.append(f"\n{'='*50}")
        lines.append(f"  {name}")
        lines.append(f"{'='*50}")
        lines.append(f"  {'Metric':<20} {'participant-macro':>18}")
        lines.append(f"  {'-'*40}")
        lines.append(f"  {'MAE':<20} {metrics['mae']:>18.4f}")
        lines.append(f"  {'Pearson r':<20} {metrics['pearson_r']:>18.4f}")
        lines.append(f"  {'CCC':<20} {metrics['ccc']:>18.4f}")
    result_path.write_text("\n".join(lines) + "\n")
    print(f"\nResult saved to: {result_path}")

    # ── fusion_params.txt ────────────────────────────────────────────────
    if fusion_param_log:
        param_path = args.output_dir / "fusion_params.txt"
        with param_path.open("w") as fh:
            fh.write(f"# N-way Fusion: {' + '.join(labels)}\n")
            fh.write(f"# {'fold':<6} {'method':<16} params\n")
            fh.write(f"# {'-'*60}\n")
            for fold_id, method, params_dict in fusion_param_log:
                fh.write(f"  {fold_id:<6} {method:<16} {json.dumps(params_dict)}\n")

            methods_in_log = sorted(set(m for _, m, _ in fusion_param_log))
            fh.write(f"\n# {'='*60}\n")
            fh.write("# Summary\n")
            fh.write(f"# {'='*60}\n")
            for method in methods_in_log:
                entries = [p for f, m, p in fusion_param_log if m == method]
                if not entries or not entries[0]:
                    fh.write(f"\n# {method}: parameter-free\n")
                    continue
                fh.write(f"\n# {method}:\n")
                # Log val_ccc stats
                val_cccs = [e.get("val_ccc", None) for e in entries]
                val_cccs = [v for v in val_cccs if v is not None]
                if val_cccs:
                    fh.write(f"#   val_ccc  mean={np.mean(val_cccs):.6f}  "
                             f"std={np.std(val_cccs):.6f}\n")
        print(f"Fusion params saved to: {param_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
