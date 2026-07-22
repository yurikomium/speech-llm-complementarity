#!/usr/bin/env python3
"""Offline end-to-end smoke test for fusion and coupling analysis.

All inputs are generated in a temporary directory. No corpus files, model
weights, network access, or paid APIs are used. The synthetic values are not
intended to reproduce any number reported in the paper.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NWAY_FUSION = ROOT / "src" / "nway_fusion.py"
COUPLING_ANALYSIS = ROOT / "src" / "coupling_analysis.py"
OUTPUT_COLUMNS = {
    "fold_group",
    "pair_key",
    "y_true",
    "female_id",
    "male_id",
    "kaisu",
    "y_pred",
}


def _run(command: list[str]) -> None:
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            "Command failed:\n"
            + " ".join(command)
            + f"\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def _synthetic_predictions(
    session: str,
    direction: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create five folds whose two predictors average exactly to y_true."""
    y_true = np.array([3.0, 4.0, 6.0, 7.0])
    orthogonal_error = np.array([1.0, -1.0, -1.0, 1.0])
    amplitudes = [0.2, 0.5, 0.8, 1.1, 1.4]
    llm_rows: list[dict] = []
    speech_rows: list[dict] = []

    for fold, amplitude in enumerate(amplitudes, start=1):
        for partner_index, (truth, error) in enumerate(
            zip(y_true, orthogonal_error), start=1
        ):
            if direction == "F2M":
                female_id = f"SYNTH_F_RATER_{session}_{fold:02d}"
                male_id = f"SYNTH_M_PARTNER_{partner_index:02d}"
            else:
                female_id = f"SYNTH_F_PARTNER_{partner_index:02d}"
                male_id = f"SYNTH_M_RATER_{session}_{fold:02d}"

            common = {
                "fold_group": fold,
                "pair_key": (
                    f"SYNTH_{session}_{direction}_{fold:02d}_{partner_index:02d}"
                ),
                "y_true": truth,
                "female_id": female_id,
                "male_id": male_id,
                "kaisu": 900 + fold,
            }
            llm_rows.append({
                **common,
                "y_pred": truth + amplitude * error,
            })
            speech_rows.append({
                **common,
                "y_pred": truth - amplitude * error,
            })

    return pd.DataFrame(llm_rows), pd.DataFrame(speech_rows)


def _write_sources(
    results_dir: Path,
    session: str,
    direction: str,
    llm: pd.DataFrame,
    speech: pd.DataFrame,
) -> tuple[Path, Path]:
    llm_path = (
        results_dir / "singles" / session / direction / "llm" / "results_llm.csv"
    )
    speech_path = (
        results_dir
        / "singles"
        / session
        / direction
        / "hubert"
        / "results_hubert.csv"
    )
    llm_path.parent.mkdir(parents=True, exist_ok=True)
    speech_path.parent.mkdir(parents=True, exist_ok=True)
    llm.to_csv(llm_path, index=False)
    speech.to_csv(speech_path, index=False)
    return llm_path, speech_path


def _read_weights(path: Path) -> dict[int, dict[str, float]]:
    pattern = re.compile(r"^\s*(\d+)\s+weighted_avg\s+(\{.*\})$")
    weights: dict[int, dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            params = json.loads(match.group(2))
            weights[int(match.group(1))] = params["weights"]
    if not weights:
        raise AssertionError(f"No weighted_avg parameters found in {path}")
    return weights


def _run_fusion(
    llm_path: Path,
    speech_path: Path,
    output_dir: Path,
    speaker: str,
) -> tuple[pd.DataFrame, dict[int, dict[str, float]]]:
    _run([
        sys.executable,
        str(NWAY_FUSION),
        "--pred-csvs",
        f"llm:{llm_path}",
        f"hubert:{speech_path}",
        "--methods",
        "weighted_avg",
        "--speaker",
        speaker,
        "--smoke",
        "--output-dir",
        str(output_dir),
    ])
    output = pd.read_csv(output_dir / "fusion_weighted_avg.csv")
    weights = _read_weights(output_dir / "fusion_params.txt")
    return output, weights


def _assert_base_fusion(
    source: pd.DataFrame,
    output: pd.DataFrame,
    weights: dict[int, dict[str, float]],
) -> None:
    if set(output.columns) != OUTPUT_COLUMNS:
        raise AssertionError(f"Unexpected fusion columns: {list(output.columns)}")
    if len(output) != len(source):
        raise AssertionError("Fusion output row count differs from the input")
    for column in ["pair_key", "fold_group", "female_id", "male_id", "kaisu"]:
        if output[column].tolist() != source[column].tolist():
            raise AssertionError(f"Fusion output changed {column}")
    if output["y_pred"].isna().any():
        raise AssertionError("Fusion output contains missing predictions")
    np.testing.assert_allclose(output["y_true"], source["y_true"])
    np.testing.assert_allclose(output["y_pred"], source["y_true"], atol=1e-12)
    if sorted(weights) != [1, 2, 3, 4, 5]:
        raise AssertionError(f"Unexpected folds in weight log: {sorted(weights)}")
    for fold_weights in weights.values():
        if fold_weights != {"llm": 0.5, "hubert": 0.5}:
            raise AssertionError(f"Unexpected synthetic weights: {fold_weights}")


def _assert_test_fold_exclusion(
    work_dir: Path,
    base_llm: pd.DataFrame,
    base_speech: pd.DataFrame,
    base_weights: dict[int, dict[str, float]],
) -> None:
    """Perturb fold 1 and verify its own selected weight is unchanged."""
    llm = base_llm.copy()
    speech = base_speech.copy()
    fold_one = llm["fold_group"] == 1
    llm.loc[fold_one, "y_pred"] = llm.loc[fold_one, "y_true"]
    sentinel_error = np.array([2.0, -2.0, -2.0, 2.0])
    speech.loc[fold_one, "y_pred"] = (
        speech.loc[fold_one, "y_true"].to_numpy() + sentinel_error
    )

    source_dir = work_dir / "leakage_sentinel_sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    llm_path = source_dir / "llm.csv"
    speech_path = source_dir / "hubert.csv"
    llm.to_csv(llm_path, index=False)
    speech.to_csv(speech_path, index=False)

    _, perturbed_weights = _run_fusion(
        llm_path,
        speech_path,
        work_dir / "leakage_sentinel_output",
        speaker="female",
    )

    if perturbed_weights[1] != base_weights[1]:
        raise AssertionError(
            "Fold 1 changed its own selected weight; test-fold leakage is possible"
        )
    if not any(
        perturbed_weights[fold] != base_weights[fold]
        for fold in [2, 3, 4, 5]
    ):
        raise AssertionError(
            "Leakage sentinel did not affect any non-held-out selection pool"
        )


def _assert_coupling(results_dir: Path, output_dir: Path) -> None:
    _run([
        sys.executable,
        str(COUPLING_ANALYSIS),
        "--results-dir",
        str(results_dir),
        "--output-dir",
        str(output_dir),
    ])
    coupling = pd.read_csv(output_dir / "coupling.csv")
    expected_columns = {
        "session",
        "direction",
        "n",
        "rho_rSpeech_vs_delta_fusion_minus_llm",
        "CI_lo",
        "CI_hi",
        "p_spearman",
    }
    if set(coupling.columns) != expected_columns:
        raise AssertionError(f"Unexpected coupling columns: {list(coupling.columns)}")
    expected_conditions = {
        ("sess1", "F2M"),
        ("sess1", "M2F"),
        ("sess2", "F2M"),
        ("sess2", "M2F"),
    }
    actual_conditions = set(zip(coupling["session"], coupling["direction"]))
    if actual_conditions != expected_conditions:
        raise AssertionError(f"Unexpected coupling conditions: {actual_conditions}")
    if coupling["n"].tolist() != [5, 5, 5, 5]:
        raise AssertionError(f"Unexpected participant counts: {coupling['n'].tolist()}")
    np.testing.assert_allclose(
        coupling["rho_rSpeech_vs_delta_fusion_minus_llm"],
        [-1.0, -1.0, -1.0, -1.0],
        atol=1e-12,
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="speech_llm_synthetic_") as tmp:
        work_dir = Path(tmp)
        results_dir = work_dir / "results"
        base_f2m_llm: pd.DataFrame | None = None
        base_f2m_speech: pd.DataFrame | None = None
        base_f2m_weights: dict[int, dict[str, float]] | None = None

        for session in ["sess1", "sess2"]:
            for direction in ["F2M", "M2F"]:
                llm, speech = _synthetic_predictions(session, direction)
                llm_path, speech_path = _write_sources(
                    results_dir, session, direction, llm, speech
                )
                output_dir = (
                    results_dir
                    / "fusion_2way"
                    / session
                    / direction
                    / "llm+hubert"
                )
                speaker = "female" if direction == "F2M" else "male"
                output, weights = _run_fusion(
                    llm_path, speech_path, output_dir, speaker
                )
                _assert_base_fusion(llm, output, weights)

                if session == "sess1" and direction == "F2M":
                    base_f2m_llm = llm
                    base_f2m_speech = speech
                    base_f2m_weights = weights

        assert base_f2m_llm is not None
        assert base_f2m_speech is not None
        assert base_f2m_weights is not None
        _assert_test_fold_exclusion(
            work_dir,
            base_f2m_llm,
            base_f2m_speech,
            base_f2m_weights,
        )
        _assert_coupling(results_dir, work_dir / "coupling")

    print("PASS: offline synthetic fusion and coupling smoke test")


if __name__ == "__main__":
    main()
