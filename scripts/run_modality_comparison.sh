#!/usr/bin/env bash
#
# run_modality_comparison.sh  –  Orchestrate the supervised baseline comparison
#
# Usage:
#   bash scripts/run_modality_comparison.sh [--smoke] [--session 2] [--skip-convert]
#
# Phases:
#   1. CSV -> .pt conversion (3 corpus modalities)
#   2. Single-modality training (Sentence-T5, HuBERT, BERT, openSMILE, J-LIWC)
#   3. 2-way fusion among supervised predictors
#
# The paper's primary Claude + HuBERT fusion is run separately with
# src/nway_fusion.py after the LLM prediction CSV is available; see README.md.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
SRC="$PROJECT_ROOT/src"

# ── Parse arguments ───────────────────────────────────────────────────────────

SMOKE=""
SESSION=2
SKIP_CONVERT=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke)    SMOKE="--smoke"; shift ;;
        --session)  SESSION="$2"; shift 2 ;;
        --skip-convert) SKIP_CONVERT=true; shift ;;
        *)          echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SINGLES_DIR="$PROJECT_ROOT/log/modality_comparison/singles"
FUSION2_DIR="$PROJECT_ROOT/log/modality_comparison/fusion_2way"

echo "============================================================"
echo "Supervised Modality Baseline Comparison"
echo "============================================================"
echo "Session      : $SESSION"
echo "Smoke test   : ${SMOKE:-no}"
echo "Skip convert : $SKIP_CONVERT"
echo ""

# ── Phase 1: CSV → .pt conversion ────────────────────────────────────────────

if [ "$SKIP_CONVERT" = false ]; then
    echo "──────────────────────────────────────────────────────────"
    echo "Phase 1: CSV → .pt conversion"
    echo "──────────────────────────────────────────────────────────"

    for MOD in bert opensmile jliwc; do
        echo ""
        echo ">>> Converting $MOD (session $SESSION)..."
        $PYTHON "$SRC/convert_csv_to_pt.py" \
            --modality "$MOD" --session "$SESSION" $SMOKE
    done
else
    echo "Phase 1: Skipped (--skip-convert)"
fi

# ── Phase 2: Single-modality training ─────────────────────────────────────────

echo ""
echo "──────────────────────────────────────────────────────────"
echo "Phase 2: Single-modality training"
echo "──────────────────────────────────────────────────────────"

# Modality definitions: name embed_dim emb_dir embed_key
MODALITIES=(
    "sentt5:768:$PROJECT_ROOT/data/preprocessed:text_embeddings"
    "hubert:1024:$PROJECT_ROOT/data/preprocessed:audio_embeddings"
    "bert:768:$PROJECT_ROOT/data/preprocessed_bert/sess${SESSION}:embeddings"
    "opensmile:338:$PROJECT_ROOT/data/preprocessed_opensmile/sess${SESSION}:embeddings"
    "jliwc:71:$PROJECT_ROOT/data/preprocessed_jliwc/sess${SESSION}:embeddings"
)

for DIRECTION_SPEC in "female:F2M" "male:M2F"; do
    SPEAKER="${DIRECTION_SPEC%%:*}"
    DIR_LABEL="${DIRECTION_SPEC##*:}"

    echo ""
    echo "=== Direction: $DIR_LABEL (speaker=$SPEAKER) ==="

    for MOD_SPEC in "${MODALITIES[@]}"; do
        IFS=':' read -r MOD DIM EMB_DIR EMB_KEY <<< "$MOD_SPEC"
        echo ""
        echo ">>> $MOD (d=$DIM, $DIR_LABEL)..."
        $PYTHON "$SRC/main_single_modal.py" \
            --modality "$MOD" \
            --embed-dim "$DIM" \
            --session "$SESSION" \
            --emb-dir "$EMB_DIR" \
            --embed-key "$EMB_KEY" \
            --speaker "$SPEAKER" \
            --out-dir "$SINGLES_DIR" \
            $SMOKE
    done
done

# ── Phase 3: supervised 2-way fusion ─────────────────────────────────────────

echo ""
echo "──────────────────────────────────────────────────────────"
echo "Phase 3: supervised 2-way fusion"
echo "──────────────────────────────────────────────────────────"

# All supervised modalities with single results
SINGLE_MODS=("sentt5" "hubert" "bert" "opensmile" "jliwc")
SINGLE_CSV_NAMES=(
    "results_sentt5.csv"
    "results_hubert.csv"
    "results_bert.csv"
    "results_opensmile.csv"
    "results_jliwc.csv"
)

for DIRECTION_SPEC in "female:F2M" "male:M2F"; do
    SPEAKER="${DIRECTION_SPEC%%:*}"
    DIR_LABEL="${DIRECTION_SPEC##*:}"

    echo ""
    echo "=== 2-way fusion: $DIR_LABEL ==="

    N=${#SINGLE_MODS[@]}
    for (( i=0; i<N; i++ )); do
        for (( j=i+1; j<N; j++ )); do
            MOD_A="${SINGLE_MODS[$i]}"
            MOD_B="${SINGLE_MODS[$j]}"
            CSV_A="$SINGLES_DIR/sess${SESSION}/${DIR_LABEL}/${MOD_A}/${SINGLE_CSV_NAMES[$i]}"
            CSV_B="$SINGLES_DIR/sess${SESSION}/${DIR_LABEL}/${MOD_B}/${SINGLE_CSV_NAMES[$j]}"

            # Skip if CSV doesn't exist
            if [ ! -f "$CSV_A" ] || [ ! -f "$CSV_B" ]; then
                echo "  [SKIP] $MOD_A + $MOD_B (CSV not found)"
                continue
            fi

            COMBO="${MOD_A}+${MOD_B}"
            OUT="$FUSION2_DIR/sess${SESSION}/${DIR_LABEL}/${COMBO}"

            echo "  >>> $COMBO..."
            $PYTHON "$SRC/nway_fusion.py" \
                --pred-csvs "${MOD_A}:${CSV_A}" "${MOD_B}:${CSV_B}" \
                --speaker "$SPEAKER" \
                --output-dir "$OUT"
        done
    done
done

echo ""
echo "============================================================"
echo "All phases complete."
echo "============================================================"
