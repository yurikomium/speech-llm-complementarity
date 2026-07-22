"""
Preprocessing tools for Like/Love score prediction models.

Subcommands:
  feature_extraction  Extract text/audio embeddings and save as .pt files.
  make_cvfold         Create leave-one-group-out CV split txt files.

--- feature_extraction ---
Supports all 25 folds of the speed-dating corpus (folds 01-25).

Default output:
  data/preprocessed/{kaisu}_{session}_{conv}_{room}_{female_id}_{male_id}_{speaker}.pt

Steps per utterance (merged consecutive same-speaker turns):
  0. Merge consecutive same-speaker utterances in the transcript
  1. Extract audio segment from WAV (start/end timestamps)
  2. VAD trimming (webrtcvad, 30ms frames, 300ms padding)
  3. Text embedding  : sentence-transformers/sentence-t5-large  -> [768]  L2-normalized
  4. Audio embedding : facebook/hubert-large-ll60k              -> [1024] L2-normalized
  5. Save .pt file with the rating-side participant's utterance embeddings

Like scores are taken from sess1/sess2 columns in fold_assignments.csv,
divided by 13.  speaker="female" uses like_F_to_M_sessN,
                 speaker="male"   uses like_M_to_F_sessN.

--- make_cvfold ---
Creates train/val/test split txt files for leave-one-group-out CV.
Generates two sets of splits — one per prediction direction:
  F2M (speaker=female, predicts female→male scores)
  M2F (speaker=male,   predicts male→female scores)

For each test fold (1-25):
  - Fix test fold
  - From the remaining folds that share no participant with the test fold,
    select one validation fold using a seed-controlled random choice
  - Participants in test+validation are excluded from all train pairs
  - No participant appears in more than one of train/val/test
  - Train filter: speaker-side only (< 3 partners removed, single pass)
    F2M → female speakers with < 3 male   partners removed
    M2F → male   speakers with < 3 female partners removed

Output files per fold:
  {SPLIT_DIR}/{Like|Love}/sess{1|2}/F2M/fold{fold_group}_train.txt
  {SPLIT_DIR}/{Like|Love}/sess{1|2}/M2F/fold{fold_group}_train.txt
  (and the corresponding _val.txt and _test.txt files)

Each file contains one pair_key per line.
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
_PROJECT_ROOT     = Path(__file__).resolve().parent.parent
TRANSCRIPT_DIR    = _PROJECT_ROOT / "data" / "transcript"
AUDIO_BASE        = Path(os.environ.get(
    "SD_AUDIO_DIR", str(_PROJECT_ROOT / "data" / "raw_audio")
))
PAIRS_CSV         = _PROJECT_ROOT / "data" / "document" / "fold_assignments_full.csv"
OUTPUT_DIR        = _PROJECT_ROOT / "data" / "preprocessed"
SPLIT_DIR         = _PROJECT_ROOT / "data" / "cv_folds"

# ── Config ─────────────────────────────────────────────────────────────────────
TARGET_FOLDS      = set(range(1, 26))    # folds 01-25
TARGET_SR         = 16000
SCORE_DIVISOR     = 13.0
VAD_AGGRESSIVENESS = 2
VAD_FRAME_MS      = 30     # webrtcvad supports 10/20/30 ms
VAD_PADDING_MS    = 300    # padding around detected speech

# {kaisu}_{session}_{conv}_{room}_{female_id}_{male_id}_concat.csv
FILENAME_RE = re.compile(
    r"^(\d+)_(\d+)_(\d+)_([A-E])_([FM]\d+)_([FM]\d+)_concat\.csv$"
)


# ── Audio helpers ──────────────────────────────────────────────────────────────

def load_wav_mono_16k(wav_path: Path) -> torch.Tensor:
    """Load WAV, convert to mono float32 at 16 kHz. Returns 1-D tensor."""
    import soundfile as sf
    import torchaudio

    data, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T)  # [channels, T]
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    return wav.squeeze(0)  # [T]


def slice_audio(wav: torch.Tensor, start: float, end: float) -> torch.Tensor:
    s = int(start * TARGET_SR)
    e = min(int(end * TARGET_SR), wav.shape[0])
    return wav[s:e]


def vad_trim(wav: torch.Tensor) -> torch.Tensor:
    """
    Remove silence with webrtcvad (30 ms frames, 300 ms padding).
    Falls back to original waveform if no speech is detected.
    """
    import webrtcvad

    if wav.numel() == 0:
        return wav

    frame_samples = TARGET_SR * VAD_FRAME_MS // 1000  # 480 at 16 kHz / 30 ms
    pad_frames    = VAD_PADDING_MS // VAD_FRAME_MS     # 10 frames

    # webrtcvad requires int16 PCM bytes
    wav_int16 = (wav.float() * 32767).clamp(-32768, 32767).short()
    raw_bytes = wav_int16.numpy().tobytes()
    frame_bytes = frame_samples * 2  # 2 bytes per int16 sample

    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

    # Build per-frame speech mask
    speech_mask = []
    for i in range(0, len(raw_bytes), frame_bytes):
        frame = raw_bytes[i: i + frame_bytes]
        if len(frame) < frame_bytes:
            frame = frame + b'\x00' * (frame_bytes - len(frame))
        speech_mask.append(vad.is_speech(frame, TARGET_SR))

    # Expand mask with ±pad_frames padding
    padded = list(speech_mask)
    for i, v in enumerate(speech_mask):
        if v:
            lo = max(0, i - pad_frames)
            hi = min(len(padded), i + pad_frames + 1)
            for j in range(lo, hi):
                padded[j] = True

    # Collect voiced frames
    voiced_chunks = []
    for i, keep in enumerate(padded):
        if keep:
            s = i * frame_samples
            e = s + frame_samples
            voiced_chunks.append(wav[s:e])

    if not voiced_chunks:
        return wav  # fallback: keep original

    return torch.cat(voiced_chunks)


# ── Transcript helpers ─────────────────────────────────────────────────────────

def merge_consecutive(df_all: pd.DataFrame, speaker: str) -> pd.DataFrame:
    """
    STEP 0: Sort utterances by start time, then merge adjacent same-speaker
    runs (i.e. when no other speaker's turn intervenes).
    """
    df_sorted = df_all.sort_values("start").reset_index(drop=True)

    result = []
    cur = None
    last_was_target = False

    for _, row in df_sorted.iterrows():
        if row["speaker"] == speaker:
            if last_was_target and cur is not None:
                # Extend current run
                cur["end"]  = row["end"]
                cur["text"] = cur["text"] + " " + str(row["text"])
            else:
                if cur is not None:
                    result.append(cur)
                cur = {
                    "start": row["start"],
                    "end":   row["end"],
                    "text":  str(row["text"]),
                }
            last_was_target = True
        else:
            last_was_target = False

    if cur is not None:
        result.append(cur)

    return pd.DataFrame(result) if result else pd.DataFrame(
        columns=["start", "end", "text"]
    )


# ── Subcommand: feature_extraction ────────────────────────────────────────────

def cmd_feature_extraction(args):
    from transformers import HubertModel, Wav2Vec2FeatureExtractor
    from sentence_transformers import SentenceTransformer

    transcript_dir = Path(args.transcript_dir)
    audio_base = Path(args.audio_dir)
    pairs_csv = Path(args.pairs_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load like scores indexed by (female_id, male_id)
    pairs_df = pd.read_csv(pairs_csv)
    pairs_df = pairs_df[pairs_df["exclude_reason"].fillna("") != "duplicate_pair"].reset_index(drop=True)
    pairs_index: dict[tuple, pd.Series] = {
        (row["female_id"], row["male_id"]): row
        for _, row in pairs_df.iterrows()
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load models once (no gradient tracking needed)
    print("Loading text model (sentence-t5-large)...")
    text_model = SentenceTransformer(
        "sentence-transformers/sentence-t5-large", device=str(device)
    )
    for p in text_model.parameters():
        p.requires_grad_(False)

    print("Loading audio model (hubert-large-ll60k)...")
    audio_feat_ext = Wav2Vec2FeatureExtractor.from_pretrained(
        "facebook/hubert-large-ll60k"
    )
    audio_model = HubertModel.from_pretrained("facebook/hubert-large-ll60k").to(device)
    for p in audio_model.parameters():
        p.requires_grad_(False)
    audio_model.eval()

    # The first filename field is kaisu, not fold_group.  Fold membership is
    # obtained from the pair-assignment table below.
    transcript_files = [
        f for f in sorted(transcript_dir.glob("*_concat.csv"))
        if FILENAME_RE.match(f.name)
    ]
    print(f"Found {len(transcript_files)} transcript files\n")

    for transcript_path in transcript_files:
        m = FILENAME_RE.match(transcript_path.name)
        kaisu_str, sess_str, conv_str, room, female_id, male_id = m.groups()
        session_base = f"{kaisu_str}_{sess_str}_{conv_str}_{room}_{female_id}_{male_id}"

        # Lookup like scores
        key = (female_id, male_id)
        if key not in pairs_index:
            print(f"[SKIP] No pair info for {female_id}_{male_id}")
            continue
        pair_row   = pairs_index[key]
        pair_key   = str(pair_row["pair_key"])
        fold_group = int(pair_row["fold_group"])
        if fold_group not in TARGET_FOLDS:
            continue
        session    = int(sess_str)
        # Use sess1 or sess2 like scores depending on which session this file is from
        sess_suffix = f"sess{session}"
        like_f2m   = float(pair_row[f"like_F_to_M_{sess_suffix}"]) / SCORE_DIVISOR
        like_m2f   = float(pair_row[f"like_M_to_F_{sess_suffix}"]) / SCORE_DIVISOR
        love_f2m   = float(pair_row[f"love_F_to_M_{sess_suffix}"]) / SCORE_DIVISOR
        love_m2f   = float(pair_row[f"love_M_to_F_{sess_suffix}"]) / SCORE_DIVISOR

        # Load transcript
        df = pd.read_csv(transcript_path)
        if "text" not in df.columns:
            print(f"[WARN] 'text' column missing in {transcript_path.name}, skipping")
            continue

        for speaker in ("female", "male"):
            out_path = output_dir / f"{session_base}_{speaker}.pt"
            if out_path.exists():
                print(f"[SKIP] {out_path.name} already exists")
                continue

            # STEP 0: merge consecutive same-speaker utterances
            utts = merge_consecutive(df, speaker)
            if utts.empty:
                print(f"[WARN] No utterances for {speaker} in {transcript_path.name}")
                continue

            audio_path = audio_base / kaisu_str / room / f"{session_base}_{speaker}.wav"
            if not audio_path.exists():
                print(f"[WARN] Audio not found: {audio_path}")
                continue

            # Load full WAV for this speaker once
            wav = load_wav_mono_16k(audio_path)

            audio_segments = []
            texts          = []

            for _, utt in utts.iterrows():
                texts.append(utt["text"])

                # STEP 1: extract audio segment
                seg = slice_audio(wav, utt["start"], utt["end"])
                if seg.numel() == 0:
                    seg = torch.zeros(TARGET_SR // 10)  # 100 ms silence fallback

                # STEP 2: VAD trimming
                seg = vad_trim(seg)
                audio_segments.append(seg)

            n_utts = len(texts)

            # STEP 3: text embeddings [n_utts, 768]
            with torch.no_grad():
                text_embs = text_model.encode(
                    texts,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                )
                text_embs = F.normalize(text_embs.to(device), p=2, dim=-1)

            # STEP 4: audio embeddings [n_utts, 1024]
            audio_embs = []
            with torch.no_grad():
                for seg in audio_segments:
                    inputs = audio_feat_ext(
                        seg.numpy(),
                        sampling_rate=TARGET_SR,
                        return_tensors="pt",
                        padding=True,
                    )
                    input_values = inputs["input_values"].to(device)
                    outputs = audio_model(input_values)
                    # [1, T, 1024] -> mean pool over time -> [1024]
                    emb = outputs.last_hidden_state.mean(dim=1).squeeze(0)
                    emb = F.normalize(emb, p=2, dim=-1)
                    audio_embs.append(emb.cpu())

            audio_embs_t = torch.stack(audio_embs)   # [n_utts, 1024]
            text_embs_t  = text_embs.cpu()            # [n_utts, 768]

            # STEP 5: save
            # speaker="female" uses *_F_to_M (female rates male)
            # speaker="male"   uses *_M_to_F (male rates female)
            like_score = like_f2m if speaker == "female" else like_m2f
            love_score = love_f2m if speaker == "female" else love_m2f

            payload = {
                "pair_key":         pair_key,
                "female_id":        female_id,
                "male_id":          male_id,
                "speaker":          speaker,
                "fold_group":       fold_group,
                "kaisu":            int(kaisu_str),
                "session":          int(sess_str),
                "text_embeddings":  text_embs_t,   # Tensor[n_utts, 768]
                "audio_embeddings": audio_embs_t,  # Tensor[n_utts, 1024]
                "like_score":       like_score,    # float (score / 13)
                "love_score":       love_score,    # float (score / 13)
            }
            torch.save(payload, out_path)
            print(f"[SAVED] {out_path.name}  ({n_utts} utterances)")

    print("\nDone.")


# ── Subcommand: make_cvfold ────────────────────────────────────────────────────

def _mad_low_variance_speakers(
    train_df: pd.DataFrame,
    id_col: str,
    score_cols: list[str],
) -> set:
    """
    Return the set of speaker IDs whose like-score std is a lower-tail outlier.

    Per-speaker std is computed over all values in score_cols (both sessions).
    Outlier threshold: median(std) - 2.5 * MAD  (lower side only).
    Speakers with std below the threshold are returned for removal.
    """
    stds: dict[str, float] = {}
    for speaker_id, group in train_df.groupby(id_col):
        vals = []
        for col in score_cols:
            vals.extend(group[col].dropna().tolist())
        stds[speaker_id] = float(np.std(vals)) if len(vals) >= 2 else 0.0

    if not stds:
        return set()

    std_arr = np.array(list(stds.values()))
    median_std = float(np.median(std_arr))
    mad = float(np.median(np.abs(std_arr - median_std)))
    threshold = median_std - 2.5 * mad

    return {sid for sid, s in stds.items() if s < threshold}


def cmd_make_cvfold(args):
    import random

    split_dir = Path(args.split_dir)

    df = pd.read_csv(Path(args.pairs_csv))
    df = df[df["exclude_reason"].fillna("") != "duplicate_pair"].reset_index(drop=True)
    all_folds = sorted(df["fold_group"].unique())

    # score column per (target, session, direction)
    score_col_map = {
        ("Like", 1, "F2M"): "like_F_to_M_sess1",
        ("Like", 2, "F2M"): "like_F_to_M_sess2",
        ("Like", 1, "M2F"): "like_M_to_F_sess1",
        ("Like", 2, "M2F"): "like_M_to_F_sess2",
        ("Love", 1, "F2M"): "love_F_to_M_sess1",
        ("Love", 2, "F2M"): "love_F_to_M_sess2",
        ("Love", 1, "M2F"): "love_M_to_F_sess1",
        ("Love", 2, "M2F"): "love_M_to_F_sess2",
    }

    for target in ("Like", "Love"):
        for session in (1, 2):
            for direction in ("F2M", "M2F"):
                out_dir = split_dir / target / f"sess{session}" / direction
                out_dir.mkdir(parents=True, exist_ok=True)

                score_cols = [score_col_map[(target, session, direction)]]
                speaker_col = "female_id" if direction == "F2M" else "male_id"

                # Pre-filter df per fold_group: remove constant-score speakers
                # (≥2 conversations, all scores identical) before any splitting.
                # Each fold_group is checked independently so a speaker with
                # varied scores in another fold_group is unaffected there.
                _filtered: list = []
                for _fg, _grp in df.groupby("fold_group"):
                    _const: set = set()
                    for _sid, _sgrp in _grp.groupby(speaker_col):
                        _s = _sgrp[score_cols[0]].dropna().values
                        if len(_s) >= 2 and float(np.std(_s)) == 0.0:
                            _const.add(_sid)
                    if _const:
                        print(f"  [{target}/sess{session}/{direction}] "
                              f"fold_group {_fg}: constant-score speakers "
                              f"removed: {sorted(_const)}")
                        _grp = _grp[~_grp[speaker_col].isin(_const)]
                    _filtered.append(_grp)
                df_cv = pd.concat(_filtered).reset_index(drop=True)

                # Recompute fold_participants from filtered df
                fold_participants_cv: dict[int, set] = {
                    fg: set(df_cv.loc[df_cv["fold_group"] == fg, "female_id"])
                        | set(df_cv.loc[df_cv["fold_group"] == fg, "male_id"])
                    for fg in all_folds
                }
                # Reset rng for each combination → identical val fold selections
                rng = random.Random(args.seed)

                for test_fold in all_folds:
                    test_participants = fold_participants_cv[test_fold]

                    # Candidate val folds: no participant overlap with test fold
                    valid_val_folds = [
                        fg for fg in all_folds
                        if fg != test_fold
                        and fold_participants_cv[fg].isdisjoint(test_participants)
                    ]

                    if not valid_val_folds:
                        print(f"[WARN] [{target}/sess{session}/{direction}] fold{test_fold:02d}: no valid val fold — skipping")
                        continue

                    val_fold = rng.choice(valid_val_folds)

                    # Train: exclude participants from both test and val
                    excluded = test_participants | fold_participants_cv[val_fold]
                    train_mask = (
                        (df_cv["fold_group"] != test_fold) &
                        (df_cv["fold_group"] != val_fold) &
                        (~df_cv["female_id"].isin(excluded)) &
                        (~df_cv["male_id"].isin(excluded))
                    )
                    train_df = df_cv[train_mask].copy()

                    # Single-pass filters: speaker side only (no same-gender cascade)
                    if direction == "F2M":
                        # 1. < 3 male partners
                        partners = train_df.groupby("female_id")["male_id"].nunique()
                        low = set(partners[partners < 3].index)
                        train_df = train_df[~train_df["female_id"].isin(low)]
                        # 2. MAD lower-tail: low score variance
                        low_var = _mad_low_variance_speakers(
                            train_df, "female_id", score_cols,
                        )
                        train_df = train_df[~train_df["female_id"].isin(low_var)]
                    else:
                        # 1. < 3 female partners
                        partners = train_df.groupby("male_id")["female_id"].nunique()
                        low = set(partners[partners < 3].index)
                        train_df = train_df[~train_df["male_id"].isin(low)]
                        # 2. MAD lower-tail: low score variance
                        low_var = _mad_low_variance_speakers(
                            train_df, "male_id", score_cols,
                        )
                        train_df = train_df[~train_df["male_id"].isin(low_var)]

                    test_keys  = df_cv[df_cv["fold_group"] == test_fold]["pair_key"].tolist()
                    val_keys   = df_cv[df_cv["fold_group"] == val_fold]["pair_key"].tolist()
                    train_keys = train_df["pair_key"].tolist()

                    (out_dir / f"fold{test_fold}_test.txt").write_text("\n".join(test_keys) + "\n")
                    (out_dir / f"fold{test_fold}_val.txt").write_text("\n".join(val_keys) + "\n")
                    (out_dir / f"fold{test_fold}_train.txt").write_text("\n".join(train_keys) + "\n")

                    print(
                        f"[{target}/sess{session}/{direction}] fold{test_fold:02d}: test={len(test_keys):3d}  "
                        f"val={len(val_keys):3d} (fold{val_fold:02d})  "
                        f"train={len(train_keys):3d}"
                    )

    print(f"\nSplit files written to {split_dir}/{{Like,Love}}/sess{{1,2}}/{{F2M,M2F}}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Preprocessing tools for Like/Love score prediction models."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- feature_extraction ---
    feature_parser = subparsers.add_parser(
        "feature_extraction",
        help="Extract text/audio embeddings from transcripts and save as .pt files.",
    )
    feature_parser.add_argument(
        "--transcript-dir", default=str(TRANSCRIPT_DIR),
        help=f"Transcript CSV directory (default: {TRANSCRIPT_DIR})",
    )
    feature_parser.add_argument(
        "--audio-dir", default=str(AUDIO_BASE),
        help=f"Raw-audio base directory (default: {AUDIO_BASE})",
    )
    feature_parser.add_argument(
        "--pairs-csv", default=str(PAIRS_CSV),
        help=f"Fold-assignment CSV (default: {PAIRS_CSV})",
    )
    feature_parser.add_argument(
        "--output-dir", default=str(OUTPUT_DIR),
        help=f"Embedding output directory (default: {OUTPUT_DIR})",
    )

    # --- make_cvfold ---
    cv_parser = subparsers.add_parser(
        "make_cvfold",
        help="Create leave-one-group-out CV split txt files.",
    )
    cv_parser.add_argument(
        "--split-dir",
        default=str(SPLIT_DIR),
        help=f"Directory to write split txt files (default: {SPLIT_DIR})",
    )
    cv_parser.add_argument(
        "--pairs-csv", default=str(PAIRS_CSV),
        help=f"Fold-assignment CSV (default: {PAIRS_CSV})",
    )
    cv_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for val fold selection (default: 42)",
    )

    args = parser.parse_args()

    if args.command == "feature_extraction":
        cmd_feature_extraction(args)
    elif args.command == "make_cvfold":
        cmd_make_cvfold(args)


if __name__ == "__main__":
    main()
