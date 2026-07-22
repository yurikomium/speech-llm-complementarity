#!/usr/bin/env python3
"""
Gemini 2.5 Flash — LLM Direct Liking Prediction.

Same task as predict_liking.py (Claude) run with Gemini 2.5 Flash. Two input
modalities: text-only (T) and text + audio (TA). Used as an MLLM baseline for
the direct-input comparison.

Usage:
  python predict_liking_gemini.py --modality T  --session 1 --direction m_to_f --smoke
  python predict_liking_gemini.py --modality TA --session 2 --direction f_to_m
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from google import genai
from google.genai import types

try:
    from _api_response import (
        ApiResponseError,
        serialize_api_response,
        write_attempt_error,
        write_jsonl_record,
    )
except ModuleNotFoundError:  # support ``python -m prompts.predict_liking_gemini``
    from prompts._api_response import (
        ApiResponseError,
        serialize_api_response,
        write_attempt_error,
        write_jsonl_record,
    )

REPO_ROOT = Path(__file__).resolve().parents[1]

FOLD_CSV = REPO_ROOT / "data/document/fold_assignments_full.csv"
OUT_DIR = REPO_ROOT / "data/interim/llm_keyutts/gemini_direct_liking"
URIS_FILE = OUT_DIR / "audio_uris.json"

MODEL = "gemini-2.5-flash"
API_KEY_ENV = "GOOGLE_API_KEY"
SLEEP_SEC = 0.3
MAX_RETRIES = 2

SESSION_CONFIG = {
    1: {
        "duration_text": "5分間",
        "input_jsonl": REPO_ROOT / "data/interim/llm_keyutts/windows_full_624_sess1.jsonl",
        "constant_rater_col": "constant_rater_sess1",
        "liking_suffix": "sess1",
    },
    2: {
        "duration_text": "10分間",
        "input_jsonl": REPO_ROOT / "data/interim/llm_keyutts/windows_full_624.jsonl",
        "constant_rater_col": "constant_rater_sess2",
        "liking_suffix": "sess2",
    },
}

DIRECTION_CONFIG = {
    "m_to_f": {"evaluator_role": "男性", "target_role": "女性", "evaluator_prefix": "M"},
    "f_to_m": {"evaluator_role": "女性", "target_role": "男性", "evaluator_prefix": "F"},
}

Q_KEYS = [f"Q{i}" for i in range(1, 14)]

LIKING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {q: {"type": "integer"} for q in Q_KEYS},
    "required": Q_KEYS,
}

# --- Prompt templates ---
# These are the canonical executable Japanese templates. The paper's appendix
# (Section "LLM Prompt Details") documents their structure and substitutions.
# T condition: identical to the Claude text-only prompt (keeps "based on the
#              transcript content only")
# TA condition: adds reference to audio input (drops "only")

SYSTEM_PROMPT_T = """\
あなたはスピードデーティングの実験参加者をシミュレートしています。

状況: あなたは{evaluator_role}参加者として、{target_role}の相手と{duration}の対話をしたところです。
以下にその対話の書き起こしが提示されます。

タスク: 対話の書き起こしの内容のみに基づいて、相手の{target_role}に対するあなたの印象を、
以下の13の質問それぞれについて1〜9の整数で回答してください。

回答尺度:
1 = 全くそう思わない
2 = そう思わない
3 = あまりそう思わない
4 = どちらかといえばそう思わない
5 = どちらでもない
6 = どちらかといえばそう思う
7 = ややそう思う
8 = そう思う
9 = 非常にそう思う

質問項目:
Q1: 私は相手の{target_role}と一緒にいる時、ほとんどいつも同じ気分になる
Q2: 相手の{target_role}はとても適応力のある人だと思う
Q3: 相手の{target_role}は責任ある仕事に推薦できる人物だと思う
Q4: 私は相手の{target_role}をとてもよくできた人だと思う
Q5: 相手の{target_role}の判断の良さには全面の信頼をおいている
Q6: 相手の{target_role}と知り合いになれば、すぐに相手の{target_role}を好きになると思う
Q7: 相手の{target_role}と私はお互いにとてもよく似ていると思う
Q8: クラスやグループで選挙があれば私は相手の{target_role}に投票するつもりだ
Q9: 相手の{target_role}はみんなから尊敬されるような人物だと思う
Q10: 相手の{target_role}はとても知的な人だと思う
Q11: 相手の{target_role}は私の知り合いの中で最も好ましい人物だと思う
Q12: 私は相手の{target_role}のような人物になりたいと思う
Q13: 相手の{target_role}は称賛の的になりやすい人物だと思う

すべての質問に必ず回答してください。対話の内容から判断できない場合は、
対話の雰囲気や文脈から最も妥当と思われるスコアをつけてください。"""

SYSTEM_PROMPT_TA = """\
あなたはスピードデーティングの実験参加者をシミュレートしています。

状況: あなたは{evaluator_role}参加者として、{target_role}の相手と{duration}の対話をしたところです。
以下にその対話の書き起こしと音声が提示されます。

タスク: 対話の書き起こしと音声に基づいて、相手の{target_role}に対するあなたの印象を、
以下の13の質問それぞれについて1〜9の整数で回答してください。

回答尺度:
1 = 全くそう思わない
2 = そう思わない
3 = あまりそう思わない
4 = どちらかといえばそう思わない
5 = どちらでもない
6 = どちらかといえばそう思う
7 = ややそう思う
8 = そう思う
9 = 非常にそう思う

質問項目:
Q1: 私は相手の{target_role}と一緒にいる時、ほとんどいつも同じ気分になる
Q2: 相手の{target_role}はとても適応力のある人だと思う
Q3: 相手の{target_role}は責任ある仕事に推薦できる人物だと思う
Q4: 私は相手の{target_role}をとてもよくできた人だと思う
Q5: 相手の{target_role}の判断の良さには全面の信頼をおいている
Q6: 相手の{target_role}と知り合いになれば、すぐに相手の{target_role}を好きになると思う
Q7: 相手の{target_role}と私はお互いにとてもよく似ていると思う
Q8: クラスやグループで選挙があれば私は相手の{target_role}に投票するつもりだ
Q9: 相手の{target_role}はみんなから尊敬されるような人物だと思う
Q10: 相手の{target_role}はとても知的な人だと思う
Q11: 相手の{target_role}は私の知り合いの中で最も好ましい人物だと思う
Q12: 私は相手の{target_role}のような人物になりたいと思う
Q13: 相手の{target_role}は称賛の的になりやすい人物だと思う

すべての質問に必ず回答してください。対話の内容と音声から判断できない場合は、
対話の雰囲気や文脈から最も妥当と思われるスコアをつけてください。"""

USER_PROMPT_T = """\
以下は、あなた（{evaluator_role}）と相手（{target_role}）の対話の書き起こしです。

{transcript}

上記の対話に基づいて、13の質問項目すべてに回答してください。"""

USER_PROMPT_TA = """\
以下は、あなた（{evaluator_role}）と相手（{target_role}）の対話の書き起こしです。
添付の音声ファイルはこの対話の録音です。

{transcript}

上記の対話の書き起こしと音声に基づいて、13の質問項目すべてに回答してください。"""

SAFETY_SETTINGS = [
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
]


def utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def format_transcript(rec: dict[str, Any]) -> str:
    lines = []
    for window in rec.get("windows", []):
        for utt in window.get("utterances", []):
            start = float(utt.get("start", 0))
            mm = int(start) // 60
            ss = int(start) % 60
            speaker = "女性" if utt.get("speaker") == "female" else "男性"
            text = utt.get("text", "")
            lines.append(f"[{mm:02d}:{ss:02d}] {speaker}: {text}")
    return "\n".join(lines)


def build_prompts(
    direction: str, rec: dict[str, Any], session: int, modality: str,
) -> tuple[str, str]:
    dir_cfg = DIRECTION_CONFIG[direction]
    sess_cfg = SESSION_CONFIG[session]
    fmt = {**dir_cfg, "duration": sess_cfg["duration_text"]}

    if modality == "T":
        system_prompt = SYSTEM_PROMPT_T.format(**fmt)
        user_prompt = USER_PROMPT_T.format(
            transcript=format_transcript(rec), **dir_cfg,
        )
    else:
        system_prompt = SYSTEM_PROMPT_TA.format(**fmt)
        user_prompt = USER_PROMPT_TA.format(
            transcript=format_transcript(rec), **dir_cfg,
        )
    return system_prompt, user_prompt


def call_gemini(
    client: genai.Client,
    system_prompt: str,
    user_prompt: str,
    audio_uri: str | None = None,
) -> tuple[dict[str, Any], dict[str, int], Any]:
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0,
        max_output_tokens=500,
        response_mime_type="application/json",
        response_schema=LIKING_SCHEMA,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        safety_settings=SAFETY_SETTINGS,
    )

    if audio_uri:
        contents: Any = [
            types.Part.from_uri(file_uri=audio_uri, mime_type="audio/flac"),
            user_prompt,
        ]
    else:
        contents = user_prompt

    resp = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=config,
    )
    raw_output = serialize_api_response(resp)

    try:
        usage = {
            "prompt_tokens": getattr(resp.usage_metadata, "prompt_token_count", 0) or 0,
            "output_tokens": getattr(resp.usage_metadata, "candidates_token_count", 0) or 0,
            "total_tokens": getattr(resp.usage_metadata, "total_token_count", 0) or 0,
        }

        if not resp.candidates or not resp.candidates[0].content or not resp.candidates[0].content.parts:
            reason = "UNKNOWN"
            if resp.candidates:
                reason = str(getattr(resp.candidates[0], "finish_reason", "UNKNOWN"))
            raise ValueError(f"Empty response. Finish reason: {reason}")

        scores = json.loads(resp.text)
    except Exception as exc:
        raise ApiResponseError(str(exc), raw_output) from exc

    return scores, usage, raw_output


def exclude_constant_raters(
    fold_df: pd.DataFrame, session: int, direction: str,
) -> pd.DataFrame:
    """Same constant-rater exclusion logic as the Claude pipeline."""
    prefix = DIRECTION_CONFIG[direction]["evaluator_prefix"]

    if session == 1:
        col = "constant_rater_sess1"
        mask = fold_df[col].str.startswith(prefix) & (fold_df[col] != "")
    else:
        col = "constant_rater_sess2"
        mask = (
            fold_df["exclude_reason"].eq("constant_rater_sess2")
            & fold_df[col].str.startswith(prefix)
        )
    return fold_df[~mask]


def load_done_keys(out_jsonl: Path) -> set[str]:
    done_keys: set[str] = set()
    if out_jsonl.exists():
        with out_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    pk = str(obj.get("pair_key", ""))
                    if pk:
                        done_keys.add(pk)
                except json.JSONDecodeError:
                    continue
    return done_keys


def validate_scores(scores: dict[str, Any]) -> list[str]:
    errors = []
    for q in Q_KEYS:
        val = scores.get(q)
        if val is None:
            errors.append(f"{q} missing")
        elif not isinstance(val, int) or val < 1 or val > 9:
            errors.append(f"{q}={val} out of range")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description="Gemini Direct Liking Prediction")
    ap.add_argument("--modality", required=True, choices=["T", "TA"],
                    help="T=text-only, TA=text+audio")
    ap.add_argument("--session", required=True, type=int, choices=[1, 2])
    ap.add_argument("--direction", required=True, choices=["m_to_f", "f_to_m"])
    ap.add_argument("--smoke", action="store_true", help="Process only 2 pairs for quick sanity check")
    args = ap.parse_args()

    modality = args.modality
    session = args.session
    direction = args.direction
    is_smoke = args.smoke
    sess_cfg = SESSION_CONFIG[session]
    condition_id = f"{modality}_{direction}_sess{session}"

    t0 = time.time()

    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        print(f"[ERROR] {API_KEY_ENV} is not set", file=sys.stderr)
        return 1

    input_jsonl = sess_cfg["input_jsonl"]
    if not input_jsonl.exists():
        print(f"[ERROR] {input_jsonl} not found", file=sys.stderr)
        return 1

    if not FOLD_CSV.exists():
        print(f"[ERROR] {FOLD_CSV} not found", file=sys.stderr)
        return 1

    audio_uris: dict[str, str] = {}
    if modality == "TA":
        if not URIS_FILE.exists():
            print(f"[ERROR] {URIS_FILE} not found. See README.md Section 3.",
                  file=sys.stderr)
            return 1
        audio_uris = json.loads(URIS_FILE.read_text(encoding="utf-8"))
        print(f"[INFO] Audio URIs loaded: {len(audio_uris)}")

    fold_df = pd.read_csv(FOLD_CSV)
    fold_df["exclude_reason"] = fold_df["exclude_reason"].fillna("")
    fold_df[sess_cfg["constant_rater_col"]] = fold_df[sess_cfg["constant_rater_col"]].fillna("")
    fold_df = fold_df[fold_df["exclude_reason"] != "duplicate_pair"]
    fold_df = exclude_constant_raters(fold_df, session, direction)

    liking_col = f"like_{'M_to_F' if direction == 'm_to_f' else 'F_to_M'}_{sess_cfg['liking_suffix']}"
    valid_pair_keys = set(fold_df["pair_key"].astype(str))
    liking_map = dict(zip(
        fold_df["pair_key"].astype(str),
        fold_df[liking_col].astype(float),
    ))
    print(f"[INFO] Valid pairs: {len(valid_pair_keys)}")

    selected: list[dict[str, Any]] = []
    with input_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line.strip())
            pk = str(rec.get("pair_key", ""))
            if pk in valid_pair_keys:
                selected.append(rec)
    print(f"[INFO] Loaded {len(selected)} transcripts")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    out_jsonl = OUT_DIR / f"predictions_{condition_id}.jsonl"
    errors_path = OUT_DIR / f"errors_{condition_id}.jsonl"

    done_keys = load_done_keys(out_jsonl)
    records_to_process = [
        r for r in selected if str(r.get("pair_key", "")) not in done_keys
    ]
    if is_smoke:
        records_to_process = records_to_process[:2]

    print(f"[INFO] Condition: {condition_id}")
    print(f"[INFO] Modality: {modality}")
    print(f"[INFO] Session: {session} ({sess_cfg['duration_text']})")
    print(f"[INFO] Direction: {direction}")
    print(f"[INFO] Model: {MODEL}")
    print(f"[INFO] To process: {len(records_to_process)} (cached: {len(done_keys)})")
    print(f"[INFO] Mode: {'SMOKE' if is_smoke else 'FULL'}")

    if not records_to_process:
        print("[INFO] All records already processed")
        return 0

    client = genai.Client(api_key=api_key)

    system_prompt, _ = build_prompts(direction, {"windows": []}, session, modality)

    n_ok = 0
    n_errors = 0
    total_usage: dict[str, int] = {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    with out_jsonl.open("a", encoding="utf-8") as fout, \
         errors_path.open("a", encoding="utf-8") as ferr:

        for i, rec in enumerate(records_to_process, 1):
            pk = str(rec.get("pair_key", ""))
            actual_liking = liking_map.get(pk)
            _, user_prompt = build_prompts(direction, rec, session, modality)

            audio_uri = None
            if modality == "TA":
                uri_key = f"{pk}_sess{session}"
                audio_uri = audio_uris.get(uri_key)
                if not audio_uri:
                    error_record = {
                        "pair_key": pk, "direction": direction,
                        "condition": condition_id,
                        "error": f"Audio URI not found for {uri_key}",
                        "raw_output": None,
                    }
                    write_jsonl_record(ferr, error_record)
                    n_errors += 1
                    print(f"[ERR]  [{i}/{len(records_to_process)}] {pk}: audio URI missing")
                    continue

            out_obj = None
            last_err = None

            for attempt in range(MAX_RETRIES + 1):
                attempt_raw = None
                try:
                    scores, usage, attempt_raw = call_gemini(
                        client, system_prompt, user_prompt, audio_uri,
                    )
                    for k in total_usage:
                        total_usage[k] += usage.get(k, 0)

                    fatal = validate_scores(scores)
                    if fatal:
                        raise ValueError(f"Validation failed: {', '.join(fatal)}")

                    liking_total = sum(scores[q] for q in Q_KEYS)
                    out_obj = {
                        "pair_key": pk,
                        "direction": direction,
                        "condition": condition_id,
                        "modality": modality,
                        "session": session,
                        "scores": scores,
                        "liking_total": liking_total,
                        "actual_liking": actual_liking,
                        "usage": usage,
                        "model": MODEL,
                        "raw_output": attempt_raw,
                    }
                    break
                except Exception as exc:
                    last_err = str(exc)
                    context = {
                        "pair_key": pk,
                        "direction": direction,
                        "condition": condition_id,
                    }
                    write_attempt_error(
                        ferr, context, attempt, MAX_RETRIES, exc, attempt_raw,
                    )
                    if attempt < MAX_RETRIES:
                        time.sleep(1.0)

            if out_obj is None:
                n_errors += 1
                print(
                    f"[ERR]  [{i}/{len(records_to_process)}] {pk}: {last_err}",
                    file=sys.stderr,
                )
            else:
                n_ok += 1
                write_jsonl_record(fout, out_obj)
                print(
                    f"[OK]   [{i}/{len(records_to_process)}] {pk}: "
                    f"total={out_obj['liking_total']}, "
                    f"actual={actual_liking}, "
                    f"tokens={usage['output_tokens']}"
                )

            elapsed = time.time() - t0
            if i % 10 == 0 or i == len(records_to_process):
                rate = elapsed / i
                eta = rate * (len(records_to_process) - i)
                print(
                    f"[INFO] {i}/{len(records_to_process)} "
                    f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining, "
                    f"ok={n_ok}, errors={n_errors})"
                )

            time.sleep(SLEEP_SEC)

    elapsed = time.time() - t0
    n_total = n_ok + n_errors

    print(f"\n{'=' * 60}")
    print(f"[DONE] {condition_id}")
    print(f"  ok={n_ok}/{n_total}, errors={n_errors}")
    print(f"  tokens: in={total_usage['prompt_tokens']:,}, "
          f"out={total_usage['output_tokens']:,}")
    print(f"  elapsed: {elapsed:.0f}s")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
