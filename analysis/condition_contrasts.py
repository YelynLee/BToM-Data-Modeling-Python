"""
조건 비교 대비(contrast)와 Holm 보정. compare_stepwise_bootstrap / truth_leak_explore 출력만 읽음 (새 데이터 계산 없음).

대비 (모델마다, 있는 조건만):
  output format         = Every-step-single - Every-step
  information           = Single-trunc - Every-step-single
  wording               = Prefix-step - Single-trunc
  information + wording = Prefix-step - Every-step-single   (Single-trunc가 없을 때의 대비; 있으면 보조로 함께 보고)
  total                 = Prefix-step - Every-step
지표: hindsight_index, abs_err_belief_vs_btom (stepwise_scenario_level.csv), pull pre/post/never (truth_leak_steps.csv가 있으면)
검정: 시나리오 단위 paired sign-flip 순열 검정 (n_perm), 95% 시나리오 bootstrap CI (층화 없음, n_boot)
Holm: (모델 x 지표 x 구간) 안에서, 위 대비 전체를 한 묶음으로 보정 (계획된 대비, 모델 간에는 보정하지 않음)

사용 예)
  python analysis/condition_contrasts.py --models "Claude-4.6-Opus" "Gemini-2.5-Pro" "GPT-5.4" \\
     --compare results/claude-opus-4-6/vanilla/compare_4cond results/gemini-2.5-pro/vanilla/compare_4cond \\
               results/gpt-5.4/vanilla/compare_2x2 --out results/stats/condition_contrasts.csv
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from stats_utils import holm

CONTRASTS = [("output format", "Every-step-single", "Every-step"),
             ("information", "Single-trunc", "Every-step-single"),
             ("wording", "Prefix-step", "Single-trunc"),
             ("information + wording", "Prefix-step", "Every-step-single"),
             ("total", "Prefix-step", "Every-step")]


def scenario_table(d, metric):
    x = pd.read_csv(os.path.join(d, "stepwise_scenario_level.csv"))
    x = x[x.metric == metric]
    conds = [c for c in ["Every-step", "Every-step-single", "Single-trunc", "Prefix-step"] if c in x.columns]
    return pd.DataFrame({c: x.dropna(subset=[c]).groupby("scenario_id")[c].first() for c in conds})


def pull_tables(d):
    f = os.path.join(d, "truth_leak_steps.csv")
    if not os.path.exists(f):
        return {}
    L = pd.read_csv(f)
    g = L.groupby(["cond", "segment", "sc"])[["num", "den"]].sum()
    g = g[g.den > 0]
    P = (g.num / g.den).rename("pull").reset_index()
    return {f"pull_{seg}": P[P.segment == seg].pivot(index="sc", columns="cond", values="pull")
            for seg in ["pre", "post", "never"]}


def test(d, rng, n_boot, n_perm):
    d = d[np.isfinite(d)]
    n = len(d)
    bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    null = (rng.choice([-1, 1], (n_perm, n)) * d).mean(1)
    p = (np.sum(np.abs(null) >= abs(d.mean())) + 1) / (n_perm + 1)
    return d.mean(), *np.percentile(bs, [2.5, 97.5]), p, n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--compare", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--n_perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    rows = []
    for m, d in zip(a.models, a.compare):
        tabs = {k: scenario_table(d, k) for k in ["hindsight_index", "abs_err_belief_vs_btom"]}
        tabs.update(pull_tables(d))
        for metric, W in tabs.items():
            block = []
            for name, x, y in CONTRASTS:
                if x not in W or y not in W:
                    continue
                if name == "information + wording" and "Single-trunc" in W:
                    name = "information + wording (Single-trunc skipped)"
                est, lo, hi, p, n = test((W[x] - W[y]).dropna().values, rng, a.n_boot, a.n_perm)
                block.append(dict(model=m, metric=metric, contrast=name, comparison=f"{x} - {y}", estimate=est,
                                  ci_low=lo, ci_high=hi, p_perm=p, n_scenarios=n))
            for r, ph in zip(block, holm([r["p_perm"] for r in block])):
                r["p_holm"] = ph
                r["holm_family"] = f"{m} | {metric} ({len(block)} contrasts)"
            rows += block
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out.to_csv(a.out, index=False)
    pd.set_option("display.width", 250)
    print(out.drop(columns="holm_family").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
