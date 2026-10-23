"""
이미 끝난 Batch의 결과를 다시 받아, 기존 피험자 CSV에 응답 원문과 토큰 수만 채워 넣음 (API 재호출 없음, 비용 없음).

- 점수(desire/belief)는 바꾸지 않음. 채우는 열: response_text, n_judgements, output_tokens, input_tokens,
  reasoning_tokens, thinking_chars, stop_reason, internal_thinking(있으면)
- batch_state.json에 기록된 모든 batch를 순서대로 읽고, 같은 요청(custom_id)이 여러 batch에 있으면
  나중 batch의 결과를 사용 (재제출된 요청이 CSV에 최종 반영된 것이므로)
- 결과 보관 기간: Claude 29일, OpenAI 30일, Gemini 6주. 기간이 지나면 다시 받을 수 없음

사용 예:
  python -m src.fetch_batch_responses --dir results/claude-opus-4-8/vanilla/prefixstep
  python -m src.fetch_batch_responses --dir ... --dry_run      # 몇 행이 채워질지만 확인
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import batch_api  # noqa: E402

FILL_COLS = ["output_tokens", "input_tokens", "reasoning_tokens", "thinking_chars", "stop_reason"]


def parse_custom_id(cid):
    subj = int(cid[1:3])
    sc = int(cid.split("_sc")[1].split("_")[0])
    t = int(cid.split("_t")[1])
    return subj, sc, (t if t else None)


def collect(state):
    """{(subj, sc, t): parsed response}  — 나중 batch가 앞의 것을 덮어씀"""
    out = {}
    for b in state.get("batches", []):
        n_ok = n_bad = 0
        try:
            for cid, kind, payload in batch_api.results(b):
                if kind != "succeeded":
                    n_bad += 1
                    continue
                out[parse_custom_id(cid)] = payload
                n_ok += 1
        except Exception as e:
            print(f"⚠️ {b.get('id')}: 결과를 받지 못했습니다 ({e}). 보관 기간이 지났을 수 있습니다.")
            continue
        print(f"📥 {b.get('provider', 'claude')} {b['id']}: 성공 {n_ok}건, 실패 {n_bad}건")
    return out


def fill_subject_csv(path, subj, responses, stepwise_prefix, dry_run=False):
    df = pd.read_csv(path)
    for c in ["response_text", "n_judgements", "internal_thinking"] + FILL_COLS:
        if c not in df.columns:
            df[c] = None
    df["response_text"] = df["response_text"].astype(object)
    df["internal_thinking"] = df["internal_thinking"].astype(object)
    df["stop_reason"] = df["stop_reason"].astype(object)

    # 한 응답 = 한 작업 단위: prefixstep은 (scenario, time_step), 나머지는 scenario
    if stepwise_prefix:
        keys = [(int(r.scenario_id), int(r.time_step) if pd.notna(r.time_step) else None) for r in df.itertuples()]
    else:
        keys = [(int(r.scenario_id), None) for r in df.itertuples()]
    df["_key"] = keys
    filled = 0
    for key, idx in df.groupby("_key", sort=False).groups.items():
        resp = responses.get((subj,) + tuple(key))
        if resp is None:
            continue
        idx = list(idx)
        if df.loc[idx, "error"].notna().any() if "error" in df.columns else False:
            continue  # 오류 행은 건너뜀 (재실행 시 다시 호출되므로)
        meta = resp.get("meta") or {}
        df.loc[idx, "n_judgements"] = len(idx)
        df.loc[idx[0], "response_text"] = resp.get("text", "")
        if resp.get("thinking") and pd.isna(df.loc[idx[0], "internal_thinking"]):
            df.loc[idx[0], "internal_thinking"] = resp["thinking"]
        for c in FILL_COLS:
            if meta.get(c) is not None:
                df.loc[idx, c] = meta[c]
        filled += 1
    df = df.drop(columns="_key")
    if filled and not dry_run:
        df.to_csv(path, index=False)
    return filled


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="batch_state.json과 subject_XX.csv가 있는 결과 폴더")
    ap.add_argument("--dry_run", action="store_true", help="CSV를 수정하지 않고 채워질 개수만 출력")
    args = ap.parse_args()

    state_path = os.path.join(args.dir, "batch_state.json")
    if not os.path.exists(state_path):
        sys.exit(f"{state_path}가 없습니다. --batch로 실행한 결과 폴더인지 확인하세요.")
    state = json.load(open(state_path))
    responses = collect(state)
    prefix = os.path.basename(os.path.normpath(args.dir)).startswith("prefixstep") or \
        any(t is not None for (_, _, t) in responses)

    total = 0
    for subj in sorted({k[0] for k in responses}):
        path = os.path.join(args.dir, f"subject_{subj:02d}.csv")
        if not os.path.exists(path):
            print(f"⚠️ {path} 없음: 건너뜀")
            continue
        n = fill_subject_csv(path, subj, responses, prefix, args.dry_run)
        total += n
        print(f"   subject_{subj:02d}: 응답 {n}개 채움")
    print(("(dry run) " if args.dry_run else "") + f"✅ 총 {total}개 응답의 원문/토큰을 채웠습니다. 점수는 바꾸지 않았습니다.")


if __name__ == "__main__":
    main()
