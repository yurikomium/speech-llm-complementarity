#!/usr/bin/env python3
"""
gpt-audio-mini — LLM Direct Liking Prediction.

Same task as predict_liking_gemini.py, run with gpt-audio-mini using text plus
audio (TA). Used as an MLLM baseline for the direct-input comparison.

Usage:
  python predict_liking_gpt_audio.py --session 2 --direction f_to_m
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI

try:
    from _api_response import (
        ApiResponseError,
        serialize_api_response,
        write_attempt_error,
    )
except ModuleNotFoundError:  # support ``python -m prompts.predict_liking_gpt_audio``
    from prompts._api_response import (
        ApiResponseError,
        serialize_api_response,
        write_attempt_error,
    )

REPO_ROOT = Path(__file__).resolve().parents[1]

FOLD_CSV = REPO_ROOT / "data/document/fold_assignments_full.csv"
OUT_DIR = REPO_ROOT / "data/interim/llm_keyutts/gpt_audio_direct_liking"
MP3_DIR = OUT_DIR / "merged_mp3"

MODEL = "gpt-audio-mini"
MODALITY = "TA"
API_KEY_ENV = "OPENAI_API_KEY"
SLEEP_SEC = 0.5
MAX_RETRIES = 2

SESSION_CONFIG = {
    1: {
        "duration_text": "5分間",
        "input_jsonl": REPO_ROOT / "data/interim/llm_keyutts/windows_full_624_sess1.jsonl",
        "liking_suffix": "sess1",
    },
    2: {
        "duration_text": "10分間",
        "input_jsonl": REPO_ROOT / "data/interim/llm_keyutts/windows_full_624.jsonl",
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
    "additionalProperties": False,
}

TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "submit_liking_scores",
        "description": "13の質問項目に対するLikingスコア（各1-9の整数）を提出する",
        "parameters": LIKING_SCHEMA,
        "strict": True,
    },
}

# --- Prompt templates ---
# These are the canonical executable Japanese templates. The paper's appendix
# (Section "LLM Prompt Details") documents their structure and substitutions.
# They are identical to the corresponding Gemini prompts.

SYSTEM_PROMPT = """\
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

USER_PROMPT = """\
以下は、あなた（{evaluator_role}）と相手（{target_role}）の対話の書き起こしです。
添付の音声ファイルはこの対話の録音です。

{transcript}

上記の対話の書き起こしと音声に基づいて、13の質問項目すべてに回答してください。"""


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
    direction: str, rec: dict[str, Any], session: int,
) -> tuple[str, str]:
    dir_cfg = DIRECTION_CONFIG[direction]
    sess_cfg = SESSION_CONFIG[session]
    fmt = {**dir_cfg, "duration": sess_cfg["duration_text"]}

    system_prompt = SYSTEM_PROMPT.format(**fmt)
    user_prompt = USER_PROMPT.format(
        transcript=format_transcript(rec), **dir_cfg,
    )
    return system_prompt, user_prompt


def call_openai(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    audio_path: Path,
) -> tuple[dict[str, Any], dict[str, int], Any]:
    audio_b64 = base64.b64encode(audio_path.read_bytes()).decode("ascii")
    user_content: Any = [
        {"type": "text", "text": user_prompt},
        {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "mp3"}},
    ]

    resp = client.chat.completions.create(
        model=MODEL,
        modalities=["text", "audio"],
        audio={"voice": "alloy", "format": "wav"},
        temperature=0,
        max_tokens=500,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        tools=[TOOL_DEF],
        tool_choice={"type": "function", "function": {"name": "submit_liking_scores"}},
    )
    raw_output = serialize_api_response(resp)
    try:
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
        }
        if not resp.choices or not resp.choices[0].message.tool_calls:
            reason = "UNKNOWN"
            if resp.choices:
                reason = str(resp.choices[0].finish_reason)
            raise ValueError(
                f"No tool call in response. Finish reason: {reason}"
            )
        tool_call = resp.choices[0].message.tool_calls[0]
        scores = json.loads(tool_call.function.arguments)
        return scores, usage, raw_output
    except Exception as exc:
        raise ApiResponseError(str(exc), raw_output) from exc


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


def exclude_constant_rater_groups(
    fold_df: pd.DataFrame, direction: str, liking_col: str,
) -> pd.DataFrame:
    """Exclude constant-score rater groups used outside paper evaluation."""
    rater_col = "female_id" if direction == "f_to_m" else "male_id"
    stats = fold_df.groupby([rater_col, "fold_group"])[liking_col].agg(
        size="size", nunique="nunique",
    )
    constant_groups = set(
        stats.loc[
            stats["size"].ge(2) & stats["nunique"].eq(1)
        ].index
    )
    excluded = [
        (rater, fold_group) in constant_groups
        for rater, fold_group in zip(
            fold_df[rater_col], fold_df["fold_group"],
        )
    ]
    return fold_df.loc[~pd.Series(excluded, index=fold_df.index)]


def main() -> int:
    ap = argparse.ArgumentParser(description="GPT-Audio-Mini Direct Liking Prediction")
    ap.add_argument("--session", required=True, type=int, choices=[1, 2])
    ap.add_argument("--direction", required=True, choices=["m_to_f", "f_to_m"])
    ap.add_argument("--smoke", action="store_true", help="Process only 2 pairs for quick sanity check")
    args = ap.parse_args()

    session = args.session
    direction = args.direction
    is_smoke = args.smoke
    sess_cfg = SESSION_CONFIG[session]
    condition_id = f"{MODALITY}_{direction}_sess{session}"

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

    if not MP3_DIR.exists():
        print(f"[ERROR] {MP3_DIR} not found", file=sys.stderr)
        return 1

    fold_df = pd.read_csv(FOLD_CSV)
    fold_df["exclude_reason"] = fold_df["exclude_reason"].fillna("")
    fold_df = fold_df[fold_df["exclude_reason"] != "duplicate_pair"]

    liking_col = f"like_{'M_to_F' if direction == 'm_to_f' else 'F_to_M'}_{sess_cfg['liking_suffix']}"
    fold_df = exclude_constant_rater_groups(fold_df, direction, liking_col)
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

    missing_mp3 = []
    for rec in selected:
        pk = str(rec.get("pair_key", ""))
        mp3_path = MP3_DIR / f"{pk}_sess{session}.mp3"
        if not mp3_path.exists():
            missing_mp3.append(str(mp3_path))
    if missing_mp3:
        print(f"[ERROR] {len(missing_mp3)} MP3 files missing:", file=sys.stderr)
        for p in missing_mp3[:5]:
            print(f"  {p}", file=sys.stderr)
        return 1
    print(f"[INFO] All {len(selected)} MP3 files confirmed")

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
    print(f"[INFO] Modality: {MODALITY}")
    print(f"[INFO] Session: {session} ({sess_cfg['duration_text']})")
    print(f"[INFO] Direction: {direction}")
    print(f"[INFO] Model: {MODEL}")
    print(f"[INFO] To process: {len(records_to_process)} (cached: {len(done_keys)})")
    print(f"[INFO] Mode: {'SMOKE' if is_smoke else 'FULL'}")

    if not records_to_process:
        print("[INFO] All records already processed")
        return 0

    client = OpenAI(api_key=api_key)

    system_prompt, _ = build_prompts(direction, {"windows": []}, session)

    n_ok = 0
    n_errors = 0
    total_usage: dict[str, int] = {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    with out_jsonl.open("a", encoding="utf-8") as fout, \
         errors_path.open("a", encoding="utf-8") as ferr:

        for i, rec in enumerate(records_to_process, 1):
            pk = str(rec.get("pair_key", ""))
            actual_liking = liking_map.get(pk)
            _, user_prompt = build_prompts(direction, rec, session)

            audio_path = MP3_DIR / f"{pk}_sess{session}.mp3"
            if not audio_path.exists():
                error_record = {
                    "pair_key": pk, "direction": direction,
                    "condition": condition_id,
                    "error": f"MP3 not found: {audio_path}",
                    "raw_output": None,
                }
                ferr.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                ferr.flush()
                n_errors += 1
                print(f"[ERR]  [{i}/{len(records_to_process)}] {pk}: MP3 missing")
                continue

            out_obj = None
            last_err = None

            for attempt in range(MAX_RETRIES + 1):
                attempt_raw = None
                try:
                    scores, usage, attempt_raw = call_openai(
                        client, system_prompt, user_prompt, audio_path,
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
                        "modality": MODALITY,
                        "session": session,
                        "scores": scores,
                        "liking_total": liking_total,
                        "actual_liking": actual_liking,
                        "usage": usage,
                        "model": MODEL,
                        "raw_output": attempt_raw,
                    }
                    last_err = None
                    break
                except Exception as exc:
                    last_err = str(exc)
                    write_attempt_error(
                        ferr,
                        {
                            "pair_key": pk,
                            "direction": direction,
                            "condition": condition_id,
                            "modality": MODALITY,
                            "session": session,
                        },
                        attempt,
                        MAX_RETRIES,
                        exc,
                        attempt_raw,
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
                fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                fout.flush()
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
