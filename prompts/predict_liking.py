#!/usr/bin/env python3
"""
LLM Direct Liking Prediction (Session 2, Claude Sonnet).

Runs zero-shot prediction of the 13-item Rubin Liking Scale from Japanese
speed-dating transcripts. Four conditions: direction (M->F / F->M) x reasoning
(A: OFF, temperature=0, forced tool use / B: ON, adaptive thinking, structured
outputs).

Usage:
  python predict_liking.py --condition A --direction m_to_f --smoke   # smoke test
  python predict_liking.py --condition A --direction m_to_f           # full run
  python predict_liking.py --condition B --direction f_to_m           # full run
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

try:
    from _api_response import (
        ApiResponseError,
        serialize_api_response,
        write_attempt_error,
    )
except ModuleNotFoundError:  # support ``python -m prompts.predict_liking``
    from prompts._api_response import (
        ApiResponseError,
        serialize_api_response,
        write_attempt_error,
    )

REPO_ROOT = Path(__file__).resolve().parents[1]

INPUT_JSONL = REPO_ROOT / "data/interim/llm_keyutts/windows_full_624.jsonl"
FOLD_CSV = REPO_ROOT / "data/document/fold_assignments_full.csv"
OUT_DIR = REPO_ROOT / "data/interim/llm_keyutts/llm_direct_liking"

MODEL = "claude-sonnet-4-6"
API_KEY_ENV = "ANTHROPIC_API_KEY"
SLEEP_SEC = 0.3
MAX_RETRIES = 2

Q_KEYS = [f"Q{i}" for i in range(1, 14)]

JSON_SCHEMA_LIKING: dict[str, Any] = {
    "type": "object",
    "properties": {
        q: {"type": "integer"}
        for q in Q_KEYS
    },
    "required": Q_KEYS,
    "additionalProperties": False,
}

# --- Prompt templates ---
# These are the canonical executable Japanese prompt templates. The paper's
# appendix (Section "LLM Prompt Details") documents their structure and
# placeholder substitutions while abbreviating the questionnaire item list.
SYSTEM_PROMPT_TEMPLATE = """\
あなたはスピードデーティングの実験参加者をシミュレートしています。

状況: あなたは{evaluator_role}参加者として、{target_role}の相手と10分間の対話をしたところです。
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

USER_PROMPT_TEMPLATE = """\
以下は、あなた（{evaluator_role}）と相手（{target_role}）の対話の書き起こしです。

{transcript}

上記の対話に基づいて、13の質問項目すべてに回答してください。"""

DIRECTION_CONFIG = {
    "m_to_f": {"evaluator_role": "男性", "target_role": "女性",
               "liking_col": "like_M_to_F_sess2"},
    "f_to_m": {"evaluator_role": "女性", "target_role": "男性",
               "liking_col": "like_F_to_M_sess2"},
}


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


def build_prompts(direction: str, rec: dict[str, Any]) -> tuple[str, str]:
    cfg = DIRECTION_CONFIG[direction]
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(**cfg)
    transcript = format_transcript(rec)
    user_prompt = USER_PROMPT_TEMPLATE.format(
        transcript=transcript, **cfg,
    )
    return system_prompt, user_prompt


def call_condition_a(
    client: Any, system_prompt: str, user_prompt: str,
) -> tuple[dict[str, Any], dict[str, int], str | None, Any]:
    """Condition A: reasoning OFF, temperature=0, forced tool use."""
    tool = {
        "name": "submit_liking_scores",
        "description": "13の質問項目に対するLikingスコア（各1-9）を提出する",
        "input_schema": JSON_SCHEMA_LIKING,
    }
    resp = client.messages.create(
        model=MODEL,
        max_tokens=500,
        temperature=0,
        system=system_prompt,
        tools=[tool],
        tool_choice={"type": "tool", "name": "submit_liking_scores"},
        messages=[{"role": "user", "content": user_prompt}],
    )
    raw_output = serialize_api_response(resp)
    try:
        usage = {
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
        }
        for block in resp.content:
            if block.type == "tool_use" and block.name == "submit_liking_scores":
                return block.input, usage, None, raw_output
        raise ValueError("No tool_use block found in response")
    except Exception as exc:
        raise ApiResponseError(str(exc), raw_output) from exc


def call_condition_b(
    client: Any, system_prompt: str, user_prompt: str,
) -> tuple[dict[str, Any], dict[str, int], str | None, Any]:
    """Condition B: reasoning ON (adaptive thinking), structured outputs."""
    resp = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=system_prompt,
        output_config={
            "format": {
                "type": "json_schema",
                "schema": JSON_SCHEMA_LIKING,
            }
        },
        messages=[{"role": "user", "content": user_prompt}],
    )
    raw_output = serialize_api_response(resp)
    try:
        usage = {
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
        }
        thinking_text = None
        scores = None
        for block in resp.content:
            if block.type == "thinking":
                thinking_text = block.thinking
            elif block.type == "text":
                scores = json.loads(block.text)
        if scores is None:
            raise ValueError("No text block with JSON found in response")
        return scores, usage, thinking_text, raw_output
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
    """Return list of fatal errors (empty = valid)."""
    errors = []
    for q in Q_KEYS:
        val = scores.get(q)
        if val is None:
            errors.append(f"{q} missing")
        elif not isinstance(val, int) or val < 1 or val > 9:
            errors.append(f"{q}={val} out of range")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description="LLM Direct Liking Prediction (Claude, Session 2)")
    ap.add_argument("--condition", required=True, choices=["A", "B"],
                    help="A=reasoning OFF (temp=0), B=reasoning ON (adaptive)")
    ap.add_argument("--direction", required=True, choices=["m_to_f", "f_to_m"])
    ap.add_argument("--smoke", action="store_true", help="Process only 2 pairs for quick sanity check")
    args = ap.parse_args()

    condition = args.condition
    direction = args.direction
    is_smoke = args.smoke
    condition_id = f"{condition}{'1' if direction == 'm_to_f' else '2'}"
    dir_cfg = DIRECTION_CONFIG[direction]

    t0 = time.time()

    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        print(f"[ERROR] {API_KEY_ENV} is not set", file=sys.stderr)
        return 1

    if not INPUT_JSONL.exists():
        print(f"[ERROR] {INPUT_JSONL} not found", file=sys.stderr)
        return 1

    if not FOLD_CSV.exists():
        print(f"[ERROR] {FOLD_CSV} not found", file=sys.stderr)
        return 1

    fold_df = pd.read_csv(FOLD_CSV)
    fold_df["exclude_reason"] = fold_df["exclude_reason"].fillna("")
    fold_df["constant_rater_sess2"] = fold_df["constant_rater_sess2"].fillna("")
    fold_df = fold_df[fold_df["exclude_reason"] != "duplicate_pair"]
    if direction == "m_to_f":
        # Exclude male constant raters from the M-to-F direction.
        fold_df = fold_df[
            ~(fold_df["exclude_reason"].eq("constant_rater_sess2")
              & fold_df["constant_rater_sess2"].str.startswith("M"))
        ]
    else:
        # Exclude female constant raters from the F-to-M direction.
        fold_df = fold_df[
            ~(fold_df["exclude_reason"].eq("constant_rater_sess2")
              & fold_df["constant_rater_sess2"].str.startswith("F"))
        ]
    valid_pair_keys = set(fold_df["pair_key"].astype(str))
    liking_map = dict(zip(
        fold_df["pair_key"].astype(str),
        fold_df[dir_cfg["liking_col"]].astype(float),
    ))
    print(f"[INFO] Valid pairs: {len(valid_pair_keys)}")

    selected: list[dict[str, Any]] = []
    with INPUT_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line.strip())
            pk = str(rec.get("pair_key", ""))
            if pk in valid_pair_keys:
                selected.append(rec)
    print(f"[INFO] Loaded {len(selected)} transcripts")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    out_jsonl = OUT_DIR / f"predictions_{condition_id}_{direction}.jsonl"
    errors_path = OUT_DIR / f"errors_{condition_id}_{direction}.jsonl"

    done_keys = load_done_keys(out_jsonl)
    records_to_process = [
        r for r in selected if str(r.get("pair_key", "")) not in done_keys
    ]
    if is_smoke:
        records_to_process = records_to_process[:2]

    print(f"[INFO] Condition: {condition_id} ({condition})")
    print(f"[INFO] Direction: {direction}")
    print(f"[INFO] Model: {MODEL}")
    print(f"[INFO] To process: {len(records_to_process)} (cached: {len(done_keys)})")
    print(f"[INFO] Mode: {'SMOKE' if is_smoke else 'FULL'}")

    if not records_to_process:
        print("[INFO] All records already processed")
        return 0

    from anthropic import Anthropic
    client = Anthropic(api_key=api_key)

    call_fn = call_condition_a if condition == "A" else call_condition_b

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(**dir_cfg)

    n_ok = 0
    n_errors = 0
    total_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}

    with out_jsonl.open("a", encoding="utf-8") as fout, \
         errors_path.open("a", encoding="utf-8") as ferr:

        for i, rec in enumerate(records_to_process, 1):
            pk = str(rec.get("pair_key", ""))
            actual_liking = liking_map.get(pk)
            _, user_prompt = build_prompts(direction, rec)

            out_obj = None
            last_err = None

            for attempt in range(MAX_RETRIES + 1):
                attempt_raw = None
                try:
                    scores, usage, thinking_text, attempt_raw = call_fn(
                        client, system_prompt, user_prompt,
                    )
                    total_usage["input_tokens"] += usage["input_tokens"]
                    total_usage["output_tokens"] += usage["output_tokens"]

                    fatal = validate_scores(scores)
                    if fatal:
                        raise ValueError(f"Validation failed: {', '.join(fatal)}")

                    liking_total = sum(scores[q] for q in Q_KEYS)
                    out_obj = {
                        "pair_key": pk,
                        "direction": direction,
                        "condition": condition_id,
                        "scores": scores,
                        "liking_total": liking_total,
                        "actual_liking": actual_liking,
                        "thinking_text": thinking_text,
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

    in_cost = total_usage["input_tokens"] / 1_000_000 * 3
    out_cost = total_usage["output_tokens"] / 1_000_000 * 15
    print(f"\n{'=' * 60}")
    print(f"[DONE] {condition_id} {direction}")
    print(f"  ok={n_ok}/{n_total}, errors={n_errors}")
    print(f"  tokens: in={total_usage['input_tokens']:,}, "
          f"out={total_usage['output_tokens']:,}")
    print(f"  cost: ${in_cost + out_cost:.2f} "
          f"(in=${in_cost:.2f} + out=${out_cost:.2f})")
    print(f"  elapsed: {elapsed:.0f}s")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
