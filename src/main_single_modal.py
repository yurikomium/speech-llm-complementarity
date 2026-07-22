"""
main_single_modal.py  –  Single-modality score prediction (25-fold LOFO CV)

Trains a SingleModalModel(d) on arbitrary-dimension per-utterance embeddings
and outputs prediction CSVs in the same format as main.py.

Usage
-----
  # Corpus-derived modalities (converted to .pt by convert_csv_to_pt.py)
  python main_single_modal.py --modality bert --embed-dim 768 --session 2 \\
      --emb-dir data/preprocessed_bert/sess2 --speaker female

  # Pretrained encoder modalities
  python main_single_modal.py --modality sentt5 --embed-dim 768 --session 2 \\
      --emb-dir data/preprocessed --embed-key text_embeddings --speaker female

  python main_single_modal.py --modality hubert --embed-dim 1024 --session 2 \\
      --emb-dir data/preprocessed --embed-key audio_embeddings --speaker female

  # Smoke test (folds 1-3 only)
  python main_single_modal.py --modality bert --embed-dim 768 --session 2 \\
      --emb-dir data/preprocessed_bert/sess2 --speaker female --smoke
"""

import argparse
import re
import csv
import sys
import datetime
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_SPLIT_DIR = str(_PROJECT_ROOT / "data" / "cv_folds")
_DEFAULT_OUT_DIR   = str(_PROJECT_ROOT / "log" / "modality_comparison" / "singles")

# ── .pt filename patterns ─────────────────────────────────────────────────────

# New format: {pair_key}_{speaker}.pt  (synthetic example: F900__M900_female.pt)
_NEW_PT_RE = re.compile(r"^([FM]\d+__[FM]\d+)_(female|male)\.pt$")

# Existing format: {kaisu}_{session}_{conv}_{room}_{F}_{M}_{speaker}.pt
_OLD_PT_RE = re.compile(
    r"^(\d+)_(\d+)_(\d+)_([A-E])_([FM]\d+)_([FM]\d+)_(female|male)\.pt$"
)

SPEAKER_CONFIG = {
    "female": {
        "direction": "F2M",
        "id_col": "female_id",
        "partner_col": "male_id",
    },
    "male": {
        "direction": "M2F",
        "id_col": "male_id",
        "partner_col": "female_id",
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_emb_index(emb_dir: Path, session: int) -> dict:
    """
    Scan emb_dir and return {pair_key: list of .pt paths}.

    Supports both new ({pair_key}_{speaker}.pt) and old
    ({kaisu}_{sess}_{conv}_{room}_{F}_{M}_{speaker}.pt) filename formats.
    """
    index: dict[str, list] = defaultdict(list)
    for pt_path in emb_dir.glob("*.pt"):
        m = _NEW_PT_RE.match(pt_path.name)
        if m:
            index[m.group(1)].append(pt_path)
            continue
        m = _OLD_PT_RE.match(pt_path.name)
        if m and int(m.group(2)) == session:
            pair_key = f"{m.group(5)}__{m.group(6)}"
            index[pair_key].append(pt_path)
    return index


def load_split_keys(split_dir: Path, fold: int, split: str) -> list[str]:
    path = split_dir / f"fold{fold}_{split}.txt"
    if not path.exists():
        return []
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def _participants_from_pair_keys(pair_keys: list[str]) -> set[str]:
    participants: set[str] = set()
    for pair_key in pair_keys:
        parts = pair_key.split("__")
        if (
            len(parts) != 2
            or not re.fullmatch(r"F\d+", parts[0])
            or not re.fullmatch(r"M\d+", parts[1])
        ):
            raise ValueError(f"Invalid pair_key in split: {pair_key!r}")
        participants.update(parts)
    return participants


def validate_participant_disjoint_splits(
    train_keys: list[str],
    val_keys: list[str],
    test_keys: list[str],
) -> None:
    """Fail if pair keys or participants overlap across CV splits."""
    split_keys = {
        "train": set(train_keys),
        "val": set(val_keys),
        "test": set(test_keys),
    }
    split_participants = {
        name: _participants_from_pair_keys(keys)
        for name, keys in split_keys.items()
    }
    comparisons = [("train", "val"), ("train", "test"), ("val", "test")]
    for left, right in comparisons:
        pair_overlap = split_keys[left] & split_keys[right]
        if pair_overlap:
            raise ValueError(
                f"Pair overlap between {left} and {right}: "
                f"{sorted(pair_overlap)}"
            )
        participant_overlap = (
            split_participants[left] & split_participants[right]
        )
        if participant_overlap:
            raise ValueError(
                f"Participant overlap between {left} and {right}: "
                f"{sorted(participant_overlap)}"
            )


def load_samples(
    pair_keys: list[str],
    emb_index: dict,
    target: str,
    speaker: str,
    embed_key: str = "embeddings",
) -> list[dict]:
    """
    Load .pt files and return sample dicts.

    Each sample contains the embedding under `embed_key` and metadata.
    The embedding is also stored as `embed_key` for the training functions.

    Only the rating-side participant's own utterance embeddings are loaded,
    matching the supervised branches reported in the paper.
    """
    key_set = set(pair_keys)
    samples = []
    for pkey, pt_paths in emb_index.items():
        if pkey not in key_set:
            continue
        for pt_path in pt_paths:
            data = torch.load(pt_path, map_location="cpu", weights_only=False)

            # Speaker filter
            if data.get("speaker") != speaker:
                continue

            # Prefer self-contained payload metadata. Existing old-format
            # files remain supported via the first filename field.
            m = _OLD_PT_RE.match(pt_path.name)
            filename_kaisu = int(m.group(1)) if m else 1
            kaisu = int(data.get("kaisu", filename_kaisu))

            emb = data.get(embed_key)
            if emb is None:
                print(f"  [WARN] key '{embed_key}' not found in {pt_path.name}, skipping")
                continue

            sample = {
                "pair_key":    data["pair_key"],
                "female_id":   data["female_id"],
                "male_id":     data["male_id"],
                "speaker":     data["speaker"],
                "fold_group":  data["fold_group"],
                "session":     data.get("session", 0),
                embed_key:     emb,
                "_target_orig":   float(data[target]),
                "_target_scaled": 0.0,
                "_kaisu":         kaisu,
            }

            samples.append(sample)

    return samples


def fit_scaler(train_samples: list) -> tuple[float, float]:
    vals = np.array([s["_target_orig"] for s in train_samples], dtype=np.float64)
    mu  = float(vals.mean())
    sig = float(vals.std())
    if sig == 0.0:
        sig = 1.0
    return mu, sig


def apply_scaler(samples: list, mu: float, sig: float) -> None:
    for s in samples:
        s["_target_scaled"] = (s["_target_orig"] - mu) / sig


def inv_scale(score: float, mu: float, sig: float) -> float:
    return max(1.0, min(9.0, score * sig + mu))


def append_csv(out_path: Path, rows: list[dict]) -> None:
    fieldnames = ["fold_group", "pair_key", "y_true", "y_pred",
                  "female_id", "male_id", "kaisu"]
    write_header = not out_path.exists()
    with out_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def run_inference_single(
    model,
    samples: list,
    device: torch.device,
    mu: float,
    sig: float,
    fold_group: int,
    embed_key: str = "embeddings",
) -> list[dict]:
    """Run inference and return CSV-ready rows (1-9 scale)."""
    model.eval()
    rows = []
    with torch.no_grad():
        for s in samples:
            x    = s[embed_key].unsqueeze(0).to(device).float()
            mask = torch.ones(1, x.shape[1], dtype=torch.bool, device=device)

            z = model(x, mask).item()

            y_pred = inv_scale(z, mu, sig)

            rows.append({
                "fold_group": fold_group,
                "pair_key":   s["pair_key"],
                "y_true":     round(s["_target_orig"], 6),
                "y_pred":     round(y_pred, 6),
                "female_id":  s["female_id"],
                "male_id":    s["male_id"],
                "kaisu":      s["_kaisu"],
            })
    return rows


# ── Evaluation & reporting (reuse from main.py patterns) ─────────────────────

def save_result_txt(eval_results: dict, out_path: Path) -> None:
    from evaluation_metrics import (
        compute_metrics,
        pairwise_accuracy,
        top1_accuracy,
    )

    lines = []
    lines.append("=" * 60)
    lines.append("FINAL EVALUATION  (all folds combined)")
    lines.append("=" * 60)

    for modality, info in eval_results.items():
        df          = info["df"]
        id_col      = info["id_col"]
        partner_col = info.get("partner_col")
        metrics = compute_metrics(df, id_col)

        lines.append(f"\n{'='*55}")
        lines.append(f"  {modality}")
        lines.append(f"{'='*55}")
        lines.append(f"  Participants : {df[id_col].nunique()}")
        lines.append(f"  Pairs        : {df['pair_key'].nunique()}")
        lines.append(f"  Samples      : {len(df)}")
        lines.append(f"  {'Metric':<20} {'participant-macro':>18}")
        lines.append(f"  {'-'*40}")
        lines.append(f"  {'MAE':<20} {metrics['mae']:>18.4f}")
        lines.append(f"  {'Pearson r':<20} {metrics['pearson_r']:>18.4f}")
        lines.append(f"  {'CCC':<20} {metrics['ccc']:>18.4f}")
        if partner_col:
            acc, n = top1_accuracy(df, id_col, partner_col)
            pw, n_pw, n_pairs = pairwise_accuracy(df, id_col, partner_col)
            lines.append(f"  {'Top-1 Acc':<20} {acc:>16.4f}  (n={n})")
            lines.append(
                f"  {'Pairwise Acc':<20} {pw:>16.4f}  "
                f"(participants={n_pw}, pairs={n_pairs})"
            )

    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nResult saved to: {out_path}")


def save_scatter_plot(df, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    yt = df["y_true"].values
    yp = df["y_pred"].values
    ax.scatter(yt, yp, alpha=0.5, s=20)
    lim_min = min(yt.min(), yp.min()) - 0.2
    lim_max = max(yt.max(), yp.max()) + 0.2
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", linewidth=1)
    ax.set_xlabel("y_true")
    ax.set_ylabel("y_pred")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Single-modality score prediction – 25-fold LOFO CV"
    )
    p.add_argument("--modality", type=str, required=True,
                   help="Modality name (e.g. bert, jliwc, opensmile, sentt5, hubert)")
    p.add_argument("--embed-dim", type=int, required=True,
                   help="Embedding dimension (e.g. 768, 71, 338, 88, 1024)")
    p.add_argument("--session", type=int, required=True, choices=[1, 2])
    p.add_argument("--emb-dir", type=Path, required=True,
                   help="Directory containing .pt files")
    p.add_argument("--embed-key", type=str, default="embeddings",
                   help="Key in .pt dict for embeddings (default: embeddings)")
    p.add_argument("--target", type=str, default="like_score",
                   choices=["like_score", "love_score"])
    p.add_argument("--split-dir", type=Path, default=Path(_DEFAULT_SPLIT_DIR))
    p.add_argument("--out-dir", type=Path, default=Path(_DEFAULT_OUT_DIR))
    p.add_argument("--speaker", type=str, required=True,
                   choices=["female", "male"])
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--max-epochs", type=int, default=500)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke test: run folds 1-3 only")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--loss-fn", type=str, default="ccc",
                   choices=["ccc"],
                   help="Per-participant CCC loss used in the paper")
    return p.parse_args()


def main():
    args = parse_args()

    # Seed for reproducibility
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    sys.path.insert(0, str(Path(__file__).parent))
    from training import SingleModalModel, fit_single

    SESSION      = args.session
    TARGET       = args.target
    EMBED_KEY    = args.embed_key
    EMBED_DIM    = args.embed_dim
    MODALITY     = args.modality
    SPEAKER      = args.speaker

    _target_dir  = "Like" if TARGET == "like_score" else "Love"
    speaker_config = SPEAKER_CONFIG[SPEAKER]
    _speaker_dir = speaker_config["direction"]
    SPLIT_DIR = args.split_dir / _target_dir / f"sess{SESSION}" / _speaker_dir

    # Output dir: out_dir/sess{N}/{direction}/{modality}/
    EXP_DIR = args.out_dir / f"sess{SESSION}" / _speaker_dir / MODALITY
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    # Check for existing results to avoid accidental overwrite
    existing = list(EXP_DIR.glob("results_*.csv"))
    if existing:
        print(f"[ERROR] Existing results found in {EXP_DIR}:")
        for f in existing:
            print(f"  {f.name}")
        print("Use a different --out-dir to avoid overwriting.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{'='*60}")
    print(f"Single-Modality Training: {MODALITY}")
    print(f"{'='*60}")
    print(f"Device       : {device}")
    print(f"Session      : {SESSION}")
    print(f"Speaker      : {SPEAKER}")
    print(f"Target       : {TARGET}")
    print(f"Loss fn      : {args.loss_fn}")
    print(f"Embed dim    : {EMBED_DIM}")
    print(f"Embed key    : {EMBED_KEY}")
    print(f"Emb dir      : {args.emb_dir}")
    print(f"Split dir    : {SPLIT_DIR}")
    print(f"Output dir   : {EXP_DIR}")
    if args.smoke:
        print("** SMOKE TEST: folds 1-3 only **")

    # Save config
    config = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "modality": MODALITY,
        "embed_dim": EMBED_DIM,
        "embed_key": EMBED_KEY,
        "session": SESSION,
        "target": TARGET,
        "speaker": SPEAKER,
        "loss_fn": args.loss_fn,
        "lr": args.lr,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "seed": args.seed,
        "emb_dir": str(args.emb_dir),
        "split_dir": str(SPLIT_DIR),
        "smoke": args.smoke,
    }
    (EXP_DIR / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    # Build index
    print("\nBuilding embedding index...")
    emb_index = build_emb_index(args.emb_dir, SESSION)
    total_files = sum(len(v) for v in emb_index.values())
    print(f"  Found {total_files} .pt files ({len(emb_index)} pairs)")

    # Discover folds
    fold_files = sorted(SPLIT_DIR.glob("fold*_train.txt"))
    folds = sorted(int(re.search(r"fold(\d+)_train", f.name).group(1))
                   for f in fold_files)
    if args.smoke:
        folds = [f for f in folds if f <= 3]
    elif folds != list(range(1, 26)):
        raise ValueError(
            "Full training requires fold1 through fold25; "
            f"found {folds}"
        )
    print(f"  Folds: {folds}\n")

    # CV loop
    results_csv = EXP_DIR / f"results_{MODALITY}.csv"
    start_time = datetime.datetime.now()

    for i, fold in enumerate(folds):
        print(f"{'─'*60}")
        print(f"Fold {fold:2d}  ({i+1}/{len(folds)})")
        print(f"{'─'*60}")

        train_keys = load_split_keys(SPLIT_DIR, fold, "train")
        val_keys   = load_split_keys(SPLIT_DIR, fold, "val")
        test_keys  = load_split_keys(SPLIT_DIR, fold, "test")

        if not train_keys or not val_keys or not test_keys:
            message = (
                f"fold {fold} has a missing or empty split: "
                f"train={len(train_keys)}, val={len(val_keys)}, "
                f"test={len(test_keys)}"
            )
            if args.smoke:
                print(f"  [SKIP] {message}")
                continue
            raise ValueError(message)

        validate_participant_disjoint_splits(
            train_keys, val_keys, test_keys,
        )

        train_samples = load_samples(train_keys, emb_index, TARGET, SPEAKER, EMBED_KEY)
        val_samples   = load_samples(val_keys,   emb_index, TARGET, SPEAKER, EMBED_KEY)
        test_samples  = load_samples(test_keys,  emb_index, TARGET, SPEAKER, EMBED_KEY)

        sample_counts = (
            len(train_samples), len(val_samples), len(test_samples),
        )
        key_counts = (len(train_keys), len(val_keys), len(test_keys))
        if sample_counts != key_counts:
            message = (
                f"fold {fold} is missing embeddings: "
                f"samples={sample_counts}, split keys={key_counts}"
            )
            if args.smoke:
                print(f"  [SKIP] {message}")
                continue
            raise ValueError(message)

        print(f"  train={len(train_samples)}  val={len(val_samples)}  "
              f"test={len(test_samples)}")

        # Scale targets
        mu, sig = fit_scaler(train_samples)
        apply_scaler(train_samples, mu, sig)
        apply_scaler(val_samples,   mu, sig)
        apply_scaler(test_samples,  mu, sig)

        # Model save path
        model_dir = EXP_DIR / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_save_path = str(model_dir / f"best_model_fold{fold:02d}.pt")

        # Train
        model = SingleModalModel(EMBED_DIM)
        print(f"  Training (loss={args.loss_fn}, patience={args.patience})...")
        model = fit_single(
            model,
            train_samples=train_samples,
            val_samples=val_samples,
            device=device,
            loss_fn=args.loss_fn,
            lr=args.lr,
            patience=args.patience,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            verbose=args.verbose,
            embed_key=EMBED_KEY,
            save_path=model_save_path,
        )

        # Inference
        if not test_samples:
            print("  [SKIP] no test samples")
            continue

        rows = run_inference_single(model, test_samples, device, mu, sig, fold, EMBED_KEY)
        append_csv(results_csv, rows)

        # Progress log
        elapsed = (datetime.datetime.now() - start_time).total_seconds()
        per_fold = elapsed / (i + 1)
        remaining = per_fold * (len(folds) - i - 1)
        print(f"  Written {len(rows)} test rows  "
              f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

    # ── Final evaluation ──────────────────────────────────────────────────────
    elapsed_total = (datetime.datetime.now() - start_time).total_seconds()
    print(f"\n{'='*60}")
    print(f"FINAL EVALUATION: {MODALITY}  ({elapsed_total:.0f}s total)")
    print(f"{'='*60}")

    if not results_csv.exists():
        print("No results generated.")
        return

    import pandas as pd
    from evaluation_metrics import print_all_metrics

    df = pd.read_csv(results_csv)
    if df.empty:
        print("Results CSV is empty.")
        return
    if not args.smoke:
        result_folds = sorted(df["fold_group"].astype(int).unique().tolist())
        if result_folds != list(range(1, 26)):
            raise ValueError(
                "Results do not cover all 25 paper folds; "
                f"found {result_folds}"
            )

    id_col = speaker_config["id_col"]
    partner_col = speaker_config["partner_col"]
    eval_results = {
        MODALITY: {"df": df, "id_col": id_col, "partner_col": partner_col}
    }

    print_all_metrics(eval_results)
    save_result_txt(eval_results, EXP_DIR / "result.txt")
    save_scatter_plot(df, EXP_DIR / "scatter_predictions.png", MODALITY)

    print("\nDone.")


if __name__ == "__main__":
    main()
