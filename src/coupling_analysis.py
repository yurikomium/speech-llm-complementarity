"""
Coupling analysis for 2-way late fusion (LLM + speech predictor).

Computes, for each of four conditions (Session 1/2 x F2M/M2F), the
Spearman rho between:

  (a) per-participant Pearson r of the speech predictor (HuBERT), and
  (b) per-participant fusion gain over LLM (Delta r = r_Fusion - r_LLM),

together with a Fisher-z 95% CI on rho. This quantifies the
per-participant coupling between speech-predictor accuracy and fusion
gain reported as the central evidence in the main text.

Methodology
-----------
  - per-participant Pearson r on matched pair_keys for LLM, Speech, Fusion
  - Delta = r_Fusion - r_LLM per participant
  - Spearman rho(r_speech, Delta) across participants in each condition
  - Fisher-z 95% CI on rho

Inputs (under --results-dir):
  singles/{session}/{direction}/{llm_name}/results_{llm_name}.csv
  singles/{session}/{direction}/{speech_name}/results_{speech_name}.csv
  fusion_2way/{session}/{direction}/{llm_name}+{speech_name}/fusion_{fusion_method}.csv

Each CSV must contain columns: pair_key, y_true, y_pred, female_id, male_id.

Usage
-----
  python src/coupling_analysis.py \\
      --results-dir /path/to/log/modality_comparison \\
      --output-dir /path/to/out
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from evaluation_metrics import pearson_r  # noqa: E402

ID_COL = {"F2M": "female_id", "M2F": "male_id"}


def per_participant_r_on_pairs(df, allowed_pairs, id_col):
    out = {}
    for pid, sub in df.groupby(id_col):
        allowed = allowed_pairs.get(pid)
        if allowed is None:
            continue
        sub2 = sub[sub["pair_key"].isin(allowed)]
        if len(sub2) < 2:
            continue
        r, _ = pearson_r(sub2["y_true"].values, sub2["y_pred"].values)
        if not np.isnan(r):
            out[pid] = r
    return out


def fisher_z_ci(rho, n, alpha=0.05):
    if abs(rho) >= 1 or n < 4:
        return (np.nan, np.nan)
    z = 0.5 * np.log((1 + rho) / (1 - rho))
    se = 1.0 / np.sqrt(n - 3)
    from scipy.stats import norm
    zc = norm.ppf(1 - alpha / 2)
    zl, zu = z - zc * se, z + zc * se
    return (np.tanh(zl), np.tanh(zu))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results-dir", type=Path, required=True,
                    help="Base directory containing singles/ and fusion_2way/ subtrees.")
    ap.add_argument("--llm-name", default="llm",
                    help="Modality name used in the LLM CSV path (default: llm).")
    ap.add_argument(
        "--speech-name", "--nonverbal-name", dest="speech_name",
        default="hubert",
        help="Modality name used in the speech CSV path (default: hubert).",
    )
    ap.add_argument("--fusion-method", default="weighted_avg",
                    help="Fusion method suffix in fusion_{method}.csv (default: weighted_avg).")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Directory to write coupling.csv.")
    ap.add_argument("--output-name", default="coupling.csv",
                    help="Filename for the coupling results CSV (default: coupling.csv).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    base = args.results_dir
    llm = args.llm_name
    speech = args.speech_name
    fusion_combo = f"{llm}+{speech}"
    fusion_file = f"fusion_{args.fusion_method}.csv"

    rows = []

    for session in ["sess1", "sess2"]:
        for direction in ["F2M", "M2F"]:
            id_col = ID_COL[direction]

            paths = {
                "LLM":    base / f"singles/{session}/{direction}/{llm}/results_{llm}.csv",
                "Speech": base / f"singles/{session}/{direction}/{speech}/results_{speech}.csv",
                "Fusion": base / f"fusion_2way/{session}/{direction}/{fusion_combo}/{fusion_file}",
            }
            dfs = {k: pd.read_csv(p) for k, p in paths.items()}

            # intersect pairs across all three sources, per participant
            grps = {k: df.groupby(id_col)["pair_key"].apply(set) for k, df in dfs.items()}
            common_participants = (
                set(grps["LLM"].index)
                & set(grps["Speech"].index)
                & set(grps["Fusion"].index)
            )
            common_pairs = {}
            for pid in common_participants:
                s = grps["LLM"][pid] & grps["Speech"][pid] & grps["Fusion"][pid]
                if len(s) >= 2:
                    common_pairs[pid] = s

            r_vals = {k: per_participant_r_on_pairs(df, common_pairs, id_col) for k, df in dfs.items()}
            participants = sorted(
                set(r_vals["LLM"].keys())
                & set(r_vals["Speech"].keys())
                & set(r_vals["Fusion"].keys())
            )
            n = len(participants)
            rN = np.array([r_vals["Speech"][pid] for pid in participants])
            rL = np.array([r_vals["LLM"][pid] for pid in participants])
            rF = np.array([r_vals["Fusion"][pid] for pid in participants])
            delta = rF - rL

            rho, p = spearmanr(rN, delta)
            lo, hi = fisher_z_ci(rho, n)

            rows.append({
                "session": session,
                "direction": direction,
                "n": n,
                "rho_rSpeech_vs_delta_fusion_minus_llm": round(float(rho), 3),
                "CI_lo": round(float(lo), 3),
                "CI_hi": round(float(hi), 3),
                "p_spearman": round(float(p), 4),
            })

    df_out = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.output_dir / args.output_name
    df_out.to_csv(out_csv, index=False)
    print(f"saved: {out_csv}\n")

    print("=" * 90)
    print(f"Coupling rho(r_{speech}, Delta_r_Fusion-LLM) for 2-way fusion ({llm}+{speech})")
    print("=" * 90)
    print()
    print(df_out.to_string(index=False))


if __name__ == "__main__":
    main()
