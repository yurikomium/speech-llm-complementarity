"""
convert_csv_to_pt.py  –  Convert corpus per-utterance CSVs to .pt files

Usage
-----
  python convert_csv_to_pt.py --modality bert --session 2
  python convert_csv_to_pt.py --modality opensmile --session 1
  python convert_csv_to_pt.py --modality jliwc --session 2

Each modality reads an explicitly supplied per-utterance CSV (--input-csv),
or uses the conventional staging path based on SD_DATA_ROOT. It splits the
rows by (pair_key, speaker) and writes one .pt file per pair x speaker in the
format expected by main_single_modal.py.

Output directory: data/preprocessed_{modality}/sess{session}/
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ── Paths ─────────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Corpus data root (external dataset). Set via the
# SD_DATA_ROOT environment variable.
_SD_DATA_ROOT = Path(os.environ.get("SD_DATA_ROOT", str(_PROJECT_ROOT)))

_FEATURES_BASE = _SD_DATA_ROOT / "data" / "features"
_PAIRS_CSV = _PROJECT_ROOT / "data" / "document" / "fold_assignments_full.csv"

# ── Modality definitions ──────────────────────────────────────────────────────

_MODALITY_CONFIG = {
    "bert": {
        "subdir": "bert_per_utterance/per_utterance_features",
        "file_sess2": "bert_per_utterance_624.csv.gz",
        "file_sess1": "bert_per_utterance_624_sess1.csv.gz",
        "feature_prefix": "emb_",
        "meta_cols": ["pair_key", "speaker", "minute_idx", "utt_id"],
    },
    "opensmile": {
        "subdir": "opensmile_per_utterance/per_utterance_features",
        "file_sess2": "opensmile_per_utterance_624.csv.gz",
        "file_sess1": "opensmile_per_utterance_624_sess1.csv.gz",
        "feature_prefix": "sm_",
        "meta_cols": ["pair_key", "utt_id", "speaker", "minute_idx"],
    },
    "jliwc": {
        "subdir": "jliwc_per_utterance/per_utterance_features",
        "file_sess2": "jliwc_per_utterance_624.csv.gz",
        "file_sess1": "jliwc_per_utterance_624_sess1.csv.gz",
        "feature_cols": ["WC", "dur"]
                        + [f"cat_{i}" for i in range(69)],  # 71D total
        "meta_cols": ["pair_key", "utt_id", "speaker", "minute_idx"],
    },
}

SCORE_DIVISOR = 13.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_pairs_index(pairs_csv: Path) -> dict:
    """Load fold_assignments CSV and return {pair_key: row_dict}."""
    df = pd.read_csv(pairs_csv)
    df = df[df["exclude_reason"].fillna("") != "duplicate_pair"].reset_index(drop=True)
    return {row["pair_key"]: row for _, row in df.iterrows()}


def get_feature_cols(df: pd.DataFrame, config: dict) -> list[str]:
    """Determine feature column names from the config and dataframe."""
    if "feature_cols" in config:
        return config["feature_cols"]
    prefix = config["feature_prefix"]
    return [c for c in df.columns if c.startswith(prefix)]


def convert(
    modality: str,
    session: int,
    output_dir: Path,
    pairs_csv: Path,
    input_csv: Path | None = None,
    chunk_size: int = 50000,
    smoke: bool = False,
):
    config = _MODALITY_CONFIG[modality]
    csv_path = input_csv or (
        _FEATURES_BASE / config["subdir"] / config[f"file_sess{session}"]
    )

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_index = load_pairs_index(pairs_csv)

    # Determine feature columns from first chunk
    first_chunk = pd.read_csv(csv_path, nrows=5)
    feature_cols = get_feature_cols(first_chunk, config)
    dim = len(feature_cols)
    use_cols = config["meta_cols"] + feature_cols

    print(f"Modality   : {modality}")
    print(f"Session    : {session}")
    print(f"CSV        : {csv_path}")
    print(f"Feature dim: {dim} ({feature_cols[0]} .. {feature_cols[-1]})")
    print(f"Output dir : {output_dir}")
    print(f"Pairs index: {len(pairs_index)} pairs")
    if smoke:
        print("** SMOKE TEST: processing first chunk only **")
    print()

    # Accumulate rows per (pair_key, speaker)
    # For memory efficiency, process in chunks
    groups: dict[tuple[str, str], list[pd.DataFrame]] = {}

    n_rows_total = 0
    for i, chunk in enumerate(pd.read_csv(csv_path, usecols=use_cols, chunksize=chunk_size)):
        n_rows_total += len(chunk)
        for (pk, spk), grp in chunk.groupby(["pair_key", "speaker"]):
            key = (pk, spk)
            if key not in groups:
                groups[key] = []
            groups[key].append(grp)

        if smoke and i == 0:
            break

    print(f"Read {n_rows_total} rows, {len(groups)} (pair_key, speaker) groups")

    # Convert each group to .pt
    n_saved = 0
    n_skipped = 0

    for (pair_key, speaker), chunks in groups.items():
        df_group = pd.concat(chunks).sort_values("utt_id").reset_index(drop=True)

        if pair_key not in pairs_index:
            n_skipped += 1
            continue

        pair_row = pairs_index[pair_key]
        fold_group = int(pair_row["fold_group"])
        kaisu = int(pair_row["kaisu"])
        female_id = str(pair_row["female_id"])
        male_id = str(pair_row["male_id"])

        sess_suffix = f"sess{session}"
        if speaker == "female":
            like_score = float(pair_row[f"like_F_to_M_{sess_suffix}"]) / SCORE_DIVISOR
            love_score = float(pair_row[f"love_F_to_M_{sess_suffix}"]) / SCORE_DIVISOR
        else:
            like_score = float(pair_row[f"like_M_to_F_{sess_suffix}"]) / SCORE_DIVISOR
            love_score = float(pair_row[f"love_M_to_F_{sess_suffix}"]) / SCORE_DIVISOR

        embeddings = torch.tensor(
            df_group[feature_cols].values, dtype=torch.float32
        )

        payload = {
            "pair_key": pair_key,
            "female_id": female_id,
            "male_id": male_id,
            "speaker": speaker,
            "fold_group": fold_group,
            "kaisu": kaisu,
            "session": session,
            "embeddings": embeddings,
            "like_score": like_score,
            "love_score": love_score,
        }

        out_path = output_dir / f"{pair_key}_{speaker}.pt"
        torch.save(payload, out_path)
        n_saved += 1

    print(f"Saved: {n_saved}  Skipped (no pair info): {n_skipped}")
    print("Done.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert corpus per-utterance CSVs to .pt files."
    )
    parser.add_argument(
        "--modality", required=True,
        choices=list(_MODALITY_CONFIG.keys()),
        help="Which modality to convert.",
    )
    parser.add_argument(
        "--session", type=int, required=True, choices=[1, 2],
        help="Session number.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: data/preprocessed_{modality}/sess{session}).",
    )
    parser.add_argument(
        "--input-csv", type=str, default=None,
        help=("Explicit per-utterance feature CSV. If omitted, use the "
              "SD_DATA_ROOT-based conventional path."),
    )
    parser.add_argument(
        "--pairs-csv", type=str, default=str(_PAIRS_CSV),
        help=f"Path to fold_assignments CSV (default: {_PAIRS_CSV}).",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke test: process first chunk only.",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        out = _PROJECT_ROOT / "data" / f"preprocessed_{args.modality}" / f"sess{args.session}"
    else:
        out = Path(args.output_dir)

    convert(
        modality=args.modality,
        session=args.session,
        output_dir=out,
        pairs_csv=Path(args.pairs_csv),
        input_csv=Path(args.input_csv) if args.input_csv else None,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
