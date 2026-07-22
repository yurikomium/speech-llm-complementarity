"""
Evaluation metrics for Like/Love score prediction.

Functions:
  ccc(y_true, y_pred)            -> float
  mae(y_true, y_pred)            -> float
  pearson_r(y_true, y_pred)      -> (r, p_value)
  compute_metrics(df, id_col)    -> dict  (participant-macro metrics)
  pairwise_accuracy(...)         -> (macro_accuracy, n_participants, n_pairs)
  top1_accuracy(...)             -> (macro_accuracy, n_participants)
  print_all_metrics(results)     -> None
"""

import numpy as np
from scipy import stats


def ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Concordance Correlation Coefficient."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mu_a = y_true.mean()
    mu_p = y_pred.mean()
    sigma_a = y_true.std()
    sigma_p = y_pred.std()
    if sigma_a == 0.0 or sigma_p == 0.0:
        return 0.0
    r = np.corrcoef(y_true, y_pred)[0, 1]
    denom = sigma_a**2 + sigma_p**2 + (mu_a - mu_p)**2
    if denom == 0.0:
        return 0.0
    return float(2.0 * r * sigma_a * sigma_p / denom)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray):
    """Returns (r, p_value). Returns (nan, nan) if constant input."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.std() == 0 or y_pred.std() == 0:
        return float("nan"), float("nan")
    r, p = stats.pearsonr(y_true, y_pred)
    return float(r), float(p)


def compute_metrics(df, id_col: str) -> dict:
    """
    Compute participant-macro metrics used in the paper.

    Parameters
    ----------
    df     : DataFrame with columns y_true, y_pred, and id_col
    id_col : column name used to group participants

    Returns
    -------
    dict with keys {mae, pearson_r, ccc}.
    """
    participant_ids = df[id_col].unique()
    per_mae, per_r, per_ccc = [], [], []

    for pid in participant_ids:
        mask = df[id_col] == pid
        yt = df.loc[mask, "y_true"].values
        yp = df.loc[mask, "y_pred"].values
        if len(yt) < 2:
            # CCC / Pearson are undefined for a single sample — skip participant
            per_mae.append(mae(yt, yp))
            continue
        per_mae.append(mae(yt, yp))
        r, _ = pearson_r(yt, yp)
        per_r.append(r)
        per_ccc.append(ccc(yt, yp))

    return {
        "mae":       float(np.nanmean(per_mae)) if per_mae else float("nan"),
        "pearson_r": float(np.nanmean(per_r))   if per_r   else float("nan"),
        "ccc":       float(np.nanmean(per_ccc)) if per_ccc else float("nan"),
    }


def top1_accuracy(df, id_col: str, partner_col: str, min_partners: int = 2) -> tuple:
    """
    Tie-aware participant-macro Top-1 accuracy.

    Every partner tied at the ground-truth maximum is valid.  If predictions
    tie at the maximum, the participant receives fractional credit equal to
    the proportion of predicted-top partners that belong to the true-top set.

    Only participants with >= min_partners distinct partners are included.

    Returns
    -------
    (accuracy, n_eligible) : (float, int)
        accuracy     -- fraction correct (nan if no eligible participants)
        n_eligible   -- number of participants included in the calculation
    """
    import pandas as pd

    credits = []

    for pid in df[id_col].unique():
        sub = df[df[id_col] == pid]
        grp = sub.groupby(partner_col)[["y_true", "y_pred"]].mean()
        if len(grp) < min_partners:
            continue
        true_top = set(grp.index[grp["y_true"] == grp["y_true"].max()])
        pred_top = set(grp.index[grp["y_pred"] == grp["y_pred"].max()])
        credits.append(len(true_top & pred_top) / len(pred_top))

    acc = float(np.mean(credits)) if credits else float("nan")
    return acc, len(credits)


def pairwise_accuracy(
    df,
    id_col: str,
    partner_col: str,
    min_partners: int = 2,
) -> tuple[float, int, int]:
    """Participant-macro pairwise ranking accuracy.

    Ground-truth ties are excluded. Predicted ties count as incorrect. Partner
    duplicates, if present, are averaged before constructing pairs.

    Returns
    -------
    (macro_accuracy, n_participants, n_pairs)
    """
    participant_scores = []
    n_pairs = 0

    for pid in df[id_col].unique():
        sub = df[df[id_col] == pid]
        grp = sub.groupby(partner_col)[["y_true", "y_pred"]].mean()
        if len(grp) < min_partners:
            continue

        y_true = grp["y_true"].to_numpy(dtype=float)
        y_pred = grp["y_pred"].to_numpy(dtype=float)
        correct = 0
        eligible = 0
        for i in range(len(grp) - 1):
            for j in range(i + 1, len(grp)):
                true_diff = y_true[i] - y_true[j]
                if true_diff == 0:
                    continue
                pred_diff = y_pred[i] - y_pred[j]
                eligible += 1
                if pred_diff != 0 and np.sign(pred_diff) == np.sign(true_diff):
                    correct += 1

        if eligible:
            participant_scores.append(correct / eligible)
            n_pairs += eligible

    accuracy = (
        float(np.mean(participant_scores))
        if participant_scores else float("nan")
    )
    return accuracy, len(participant_scores), n_pairs


def print_all_metrics(results: dict) -> None:
    """
    Print metrics for all modalities.

    Parameters
    ----------
    results : dict  { modality -> {"df": DataFrame, "id_col": str, "partner_col": str} }
              modality in {"text_only", "audio_only", "late_fusion"}
    """
    for modality, info in results.items():
        df          = info["df"]
        id_col      = info["id_col"]
        partner_col = info.get("partner_col")
        metrics = compute_metrics(df, id_col)

        print(f"\n{'='*55}")
        print(f"  {modality}")
        print(f"{'='*55}")
        print(f"  {'Metric':<20} {'participant-macro':>18}")
        print(f"  {'-'*40}")
        print(f"  {'MAE':<20} {metrics['mae']:>18.4f}")
        print(f"  {'Pearson r':<20} {metrics['pearson_r']:>18.4f}")
        print(f"  {'CCC':<20} {metrics['ccc']:>18.4f}")
        if partner_col:
            acc, n = top1_accuracy(df, id_col, partner_col)
            pw, n_pw, n_pairs = pairwise_accuracy(df, id_col, partner_col)
            print(f"  {'Top-1 Acc':<20} {acc:>16.4f}  (n={n})")
            print(f"  {'Pairwise Acc':<20} {pw:>16.4f}  "
                  f"(participants={n_pw}, pairs={n_pairs})")
