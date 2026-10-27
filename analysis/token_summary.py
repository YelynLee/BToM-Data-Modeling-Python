"""
조건별 생성 토큰 요약: 호출당 / 판단당 / 시나리오당.

전체 생성 토큰(gen_tokens)은 제공사마다 정의가 달라 맞춰서 계산함
  - Claude, OpenAI, DeepSeek: output_tokens가 추론 토큰을 포함 -> gen = output_tokens
  - Gemini: output_tokens(candidates)에 추론(thoughts)이 빠져 있음 -> gen = output_tokens + reasoning_tokens
판단당 = gen / n_judgements (Every-step은 한 응답이 T개 스텝에 답함, Prefix / singlestep은 1)
시나리오당 = 그 시나리오에 쓴 호출들의 gen 합 (피험자 평균)

사용 예)
  python analysis/token_summary.py \\
      --input results/claude-opus-4-6/vanilla/everystep/rerun results/claude-opus-4-6/vanilla/singlestep \\
              results/claude-opus-4-6/vanilla/prefixstep \\
      --labels "Every-step" "Every-step-single" "Prefix-step" --out results/claude-opus-4-6/vanilla/compare/tokens.csv
"""
import os
import glob
import argparse
import numpy as np
import pandas as pd

PER_STEP = ("prefixstep", "singlestep")


def load_calls(path):
    """폴더의 subject_*.csv -> 호출 1회당 1행 (subject, scenario_id, time_step, tokens, n_judgements)"""
    frames = []
    for f in sorted(glob.glob(os.path.join(path, "subject_*.csv"))):
        d = pd.read_csv(f)
        d["subject"] = int(os.path.basename(f)[8:10])
        frames.append(d)
    if not frames:
        raise FileNotFoundError(f"No subject_*.csv in {path}")
    df = pd.concat(frames, ignore_index=True)
    if "error" in df.columns:
        df = df[df["error"].isna()]
    if "output_tokens" not in df.columns or df["output_tokens"].isna().all():
        raise ValueError(f"{path}: 토큰 기록이 없습니다 (토큰 기록 이전 실행이면 fetch_batch_responses.py로 채우거나 다시 실행).")
    mode = str(df["mode"].dropna().iloc[0]) if "mode" in df.columns and df["mode"].notna().any() else ""
    keys = ["subject", "scenario_id"] + (["time_step"] if mode in PER_STEP else [])
    n_j = df.groupby(keys).size().rename("n_rows")
    calls = df.groupby(keys).first()
    calls["n_judgements"] = calls["n_judgements"].fillna(n_j) if "n_judgements" in calls else n_j
    model = str(df["model"].dropna().iloc[0]).lower() if "model" in df.columns else ""
    out = calls["output_tokens"].astype(float)
    rsn = calls["reasoning_tokens"].astype(float) if "reasoning_tokens" in calls else pd.Series(0.0, index=calls.index)
    calls["gen_tokens"] = out + rsn.fillna(0) if "gemini" in model else out
    calls["gen_per_judgement"] = calls["gen_tokens"] / calls["n_judgements"]
    return calls.reset_index(), mode, model


def summarize(calls, label, mode, model):
    per_sc = calls.groupby(["subject", "scenario_id"])["gen_tokens"].sum().groupby("scenario_id").mean()
    q = lambda s: (s.mean(), s.median(), s.quantile(0.1), s.quantile(0.9))
    rows = []
    items = [("gen_per_call", calls["gen_tokens"]), ("gen_per_judgement", calls["gen_per_judgement"]),
             ("gen_per_scenario", per_sc), ("output_tokens_per_call", calls["output_tokens"].astype(float))]
    if "reasoning_tokens" in calls and calls["reasoning_tokens"].notna().any():
        items.append(("reasoning_tokens_per_call", calls["reasoning_tokens"].astype(float)))
    for name, s in items:
        m, med, lo, hi = q(s.dropna())
        rows.append(dict(condition=label, model=model, mode=mode, metric=name, mean=m, median=med, p10=lo, p90=hi,
                         n=int(s.notna().sum())))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    assert len(args.input) == len(args.labels)
    rows = []
    for p, lab in zip(args.input, args.labels):
        calls, mode, model = load_calls(p)
        rows += summarize(calls, lab, mode, model)
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(res.pivot(index="metric", columns="condition", values="mean").round(1)[args.labels].to_string())
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        res.to_csv(args.out, index=False)
        print(f"✅ saved: {args.out}")


if __name__ == "__main__":
    main()
