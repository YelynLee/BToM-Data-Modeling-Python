"""
조건별 단계별 VPA 그림 (논문 VPA 디자인, analysis/vpa_plot.py). compare_stepwise_bootstrap.py의 요약 csv를 읽어 그림만 그림.

  1) 모델별 그림: 행 = 조건(Every-step, Every-step-single, Single-trunc, Prefix-step), 열 = desire / belief
       vpa_conditions_<모델>.png
  2) 요약 그림 (belief만): 행 = 모델, 열 = 조건
       vpa_conditions_belief_summary.png
  + 그림에 쓴 값: vpa_conditions.csv
CI는 compare_stepwise_bootstrap 결과를 그대로 씀 (기본: 시나리오 bootstrap. 요약 csv의 bootstrap 열로 확인).

사용 예)
  python analysis/plot_vpa_conditions.py \\
     --models "Claude-4.6-Opus" "Gemini-2.5-Pro" "GPT-5.4" \\
     --compare results/claude-opus-4-6/vanilla/compare_4cond results/gemini-2.5-pro/vanilla/compare_4cond \\
               results/gpt-5.4/vanilla/compare_2x2 \\
     --out_dir results/stats/vpa
"""
import os
import sys
import argparse
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from analysis.vpa_plot import draw_vpa, rows_from_summary, results_to_frame

def note(scheme):
    return (f"bar = unique adjusted R² (95% CI, {scheme} bootstrap); "
            "β = standardized coefficient in the full model")

CONDS = ["Every-step", "Every-step-single", "Single-trunc", "Prefix-step"]
SHORT = {"Every-step": "Every-step", "Every-step-single": "Every-step-single", "Single-trunc": "Single-trunc",
         "Prefix-step": "Prefix-step"}
PRED = {"desire": ["btom", "truebelief", "nocost", "motionheuristic"],
        "belief": ["btom", "truebelief", "nocost", "motionheuristic", "hindsight"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, help="그림에 쓸 모델 이름")
    ap.add_argument("--compare", nargs="+", required=True, help="각 모델의 compare 폴더 (stepwise_bootstrap_summary.csv)")
    ap.add_argument("--conditions", nargs="+", default=CONDS)
    ap.add_argument("--out_dir", default="results/stats/vpa")
    a = ap.parse_args()
    assert len(a.models) == len(a.compare)
    os.makedirs(a.out_dir, exist_ok=True)

    per_model, frames, schemes = {}, [], set()
    for m, d in zip(a.models, a.compare):
        s = pd.read_csv(os.path.join(d, "stepwise_bootstrap_summary.csv"))
        # bootstrap 열이 없는 예전 요약은 2단계 기본값으로 만든 것
        scheme = s["bootstrap"].iloc[0] if "bootstrap" in s else "two-level (scenarios, then runs)"
        schemes.add(scheme)
        conds = [c for c in a.conditions if c in set(s.condition)]
        rows = rows_from_summary(s, conds)
        per_model[m] = dict(rows)
        draw_vpa(rows, os.path.join(a.out_dir, f"vpa_conditions_{m.replace(' ', '_')}.png"),
                 predictors={"Desire": PRED["desire"], "Belief": PRED["belief"]}, title=m, note=note(scheme))
        f = results_to_frame(rows)
        f.insert(0, "model", m)
        frames.append(f.rename(columns={"row": "condition"}))

    # 요약: 행 = 모델, 열 = 조건, belief만
    conds = [c for c in a.conditions if any(c in per_model[m] for m in a.models)]
    rows = [(m, {SHORT[c]: (per_model[m].get(c) or {}).get("Belief") for c in conds}) for m in a.models]
    draw_vpa(rows, os.path.join(a.out_dir, "vpa_conditions_belief_summary.png"),
             columns=[SHORT[c] for c in conds], predictors={SHORT[c]: PRED["belief"] for c in conds},
             title="Belief: variance partitioning over all steps, by condition", col_fontsize=14,
             note=note(schemes.pop() if len(schemes) == 1 else "mixed: " + " / ".join(sorted(schemes))))
    pd.concat(frames).to_csv(os.path.join(a.out_dir, "vpa_conditions.csv"), index=False)
    print(f"💾 {os.path.join(a.out_dir, 'vpa_conditions.csv')}")


if __name__ == "__main__":
    main()
