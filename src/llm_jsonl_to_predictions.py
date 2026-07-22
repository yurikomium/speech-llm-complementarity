"""Convert saved LLM JSONL outputs to the common out-of-fold CSV schema.

The prompt scripts save thirteen 1--9 item scores for every pair.  The paper's
LLM prediction is their arithmetic mean.  This converter joins the saved
outputs with the fold-assignment table and writes the schema consumed by
``nway_fusion.py``.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


Q_KEYS = tuple(f"Q{i}" for i in range(1, 14))
DIRECTION_TO_TARGET = {
    "f_to_m": "like_F_to_M_sess{session}",
    "m_to_f": "like_M_to_F_sess{session}",
}
SCORE_DIVISOR = 13.0


def _prediction_mean(record: dict) -> float:
    scores = record.get("scores")
    if not isinstance(scores, dict):
        raise ValueError("record has no scores object")
    missing = [key for key in Q_KEYS if key not in scores]
    if missing:
        raise ValueError(f"record is missing scores: {missing}")
    values = [float(scores[key]) for key in Q_KEYS]
    if any(value < 1 or value > 9 for value in values):
        raise ValueError("LLM item scores must be in the 1--9 range")
    return sum(values) / SCORE_DIVISOR


def _exclude_constant_rater_groups(
    pairs: pd.DataFrame,
    direction: str,
    target_col: str,
) -> pd.DataFrame:
    """Apply the paper's per-(rater, fold-group) constant-score exclusion."""
    rater_col = "female_id" if direction == "f_to_m" else "male_id"
    stats = pairs.groupby([rater_col, "fold_group"])[target_col].agg(
        size="size", nunique="nunique",
    )
    constant_groups = set(
        stats.loc[
            stats["size"].ge(2) & stats["nunique"].eq(1)
        ].index
    )
    if not constant_groups:
        return pairs.copy()
    excluded = [
        (rater, fold_group) in constant_groups
        for rater, fold_group in zip(pairs[rater_col], pairs["fold_group"])
    ]
    return pairs.loc[~pd.Series(excluded, index=pairs.index)].copy()


def convert_jsonl(
    input_jsonl: Path,
    pairs_csv: Path,
    output_csv: Path,
    direction: str,
    session: int,
) -> pd.DataFrame:
    pairs_all = pd.read_csv(pairs_csv)
    if "exclude_reason" in pairs_all.columns:
        pairs_all = pairs_all[
            pairs_all["exclude_reason"].fillna("") != "duplicate_pair"
        ]
    target_col = DIRECTION_TO_TARGET[direction].format(session=session)
    required_pair_cols = {
        "pair_key", "female_id", "male_id", "fold_group", "kaisu", target_col
    }
    missing_pair_cols = required_pair_cols - set(pairs_all.columns)
    if missing_pair_cols:
        raise ValueError(
            f"fold-assignment CSV is missing columns: {sorted(missing_pair_cols)}"
        )

    if pairs_all["pair_key"].duplicated().any():
        raise ValueError("pair_key must be unique after duplicate-pair exclusion")
    for col in ("fold_group", "kaisu", target_col):
        numeric = pd.to_numeric(pairs_all[col], errors="coerce")
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError(f"fold-assignment CSV has non-finite {col} values")
        pairs_all[col] = numeric
    for col in ("pair_key", "female_id", "male_id"):
        values = pairs_all[col].astype("string")
        if values.isna().any() or values.str.strip().eq("").any():
            raise ValueError(f"fold-assignment CSV has missing {col} values")
        pairs_all[col] = values.astype(str)
    pairs_all = pairs_all.set_index("pair_key", drop=False)
    pairs = _exclude_constant_rater_groups(
        pairs_all.reset_index(drop=True), direction, target_col,
    ).set_index("pair_key", drop=False)

    rows = []
    seen_jsonl = set()
    seen_canonical = set()
    with input_jsonl.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("direction") != direction:
                continue
            if "session" in record and int(record["session"]) != session:
                raise ValueError(
                    f"line {line_number}: record session={record['session']} "
                    f"does not match --session {session}"
                )
            pair_key = str(record["pair_key"])
            if pair_key in seen_jsonl:
                raise ValueError(f"duplicate pair_key in JSONL: {pair_key}")
            if pair_key not in pairs_all.index:
                raise ValueError(
                    f"line {line_number}: pair_key not found in fold table: {pair_key}"
                )
            seen_jsonl.add(pair_key)
            if pair_key not in pairs.index:
                continue
            seen_canonical.add(pair_key)
            pair = pairs.loc[pair_key]
            rows.append(
                {
                    "fold_group": int(pair["fold_group"]),
                    "pair_key": pair_key,
                    "y_true": float(pair[target_col]) / SCORE_DIVISOR,
                    "y_pred": _prediction_mean(record),
                    "female_id": str(pair["female_id"]),
                    "male_id": str(pair["male_id"]),
                    "kaisu": int(pair["kaisu"]),
                }
            )

    if not rows:
        raise ValueError(f"no records found for direction={direction}")

    missing_pairs = sorted(set(pairs.index) - seen_canonical)
    if missing_pairs:
        raise ValueError(
            f"JSONL is missing {len(missing_pairs)} canonical pairs; "
            f"examples: {missing_pairs[:5]}"
        )

    output = pd.DataFrame(rows).sort_values(
        ["fold_group", "pair_key"]
    ).reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--pairs-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--direction", choices=sorted(DIRECTION_TO_TARGET), required=True
    )
    parser.add_argument("--session", type=int, choices=[1, 2], required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = convert_jsonl(
        args.input_jsonl,
        args.pairs_csv,
        args.output_csv,
        args.direction,
        args.session,
    )
    print(f"saved: {args.output_csv} ({len(output)} rows)")


if __name__ == "__main__":
    main()
