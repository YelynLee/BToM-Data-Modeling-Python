"""
부록 Figure 9: Every-step(논문 16회) belief의 hindsight index, 유효 출력이 있는 모든 모델.
막대 색 = Better / Worse 그룹, 큰 값부터 정렬. 0 = BToM, 1 = Hindsight(λ = 2).

입력: compare_stepwise_bootstrap.py --no_paired 로 모델들을 한 번에 돌린 stepwise_bootstrap_summary.csv
  python analysis/compare_stepwise_bootstrap.py --no_paired \\
     --input results/claude-opus-4-6/vanilla/everystep/everystep_valid_only.csv \\
             results/gemini-2.5-pro/vanilla/everystep/everystep_valid_only.csv ... \\
     --labels "Claude-4.6-Opus" "Gemini-2.5-Pro" ... --out_dir results/stats/everystep_hi
  python analysis/plot_hindsight_index_all.py --summary results/stats/everystep_hi/stepwise_bootstrap_summary.csv \\
     --better "Claude-4.6-Opus" "Gemini-2.5-Pro" "DeepSeek-R1" "DeepSeek-V4-Pro" --out results/stats/figureS6_hindsight_index.png
"""
import os
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_BETTER, C_WORSE = "#60A917", "#D80073"   # 논문 Figure 3의 Better / Worse 색


def current_belief_index():
    """agent의 현재 belief(Spot 2가 보인 뒤 참에 확신, 그 전에는 BToM)를 평정으로 넣었을 때의 hindsight index.
    Spot 2가 보이기 전에는 Hindsight = BToM이라 지수에서 빠지므로 TrueBelief 레퍼런스로 계산한 값과 같음."""
    import sys, io, contextlib
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for q in (root, os.path.join(root, "analysis"), os.path.join(root, "src")):
        if q not in sys.path:
            sys.path.insert(0, q)
    import numpy as np
    with contextlib.redirect_stdout(io.StringIO()):
        import compare_stepwise_bootstrap as C
        allowed = C.get_group_indices(include_irrational=False)
        scen = sorted(s for g in range(1, 8) for s in allowed[g - 1])
        rows = C.row_table(scen)
        refs = C.ref_arrays(rows, {r: C.load_reference_everystep(r) for r in C.ALL_REFS})
        eng = C.Engine(rows, refs)
    return float(np.nanmean(eng.hindsight_by_scenario(refs["truebelief"], np.ones(len(rows), bool))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--better", nargs="+", required=True, help="Better 그룹 모델 이름 (summary의 condition 값)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_current_belief", action="store_true", help="현재 belief 기준선을 그리지 않음")
    a = ap.parse_args()
    cb = None if a.no_current_belief else current_belief_index()
    s = pd.read_csv(a.summary)
    h = s[(s.metric == "hindsight_index") & ~s.condition.str.contains(" - ")].drop_duplicates("condition")
    h = h.sort_values("estimate", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(0.55 * len(h) + 1.4, 2.3))
    for i, r in h.iterrows():
        col = C_BETTER if r.condition in a.better else C_WORSE
        ax.bar(i, r.estimate, width=0.62, color=col)
        ax.errorbar(i, r.estimate, [[r.estimate - r.ci_low], [r.ci_high - r.estimate]], color="#333", lw=0.8, capsize=2)
    ax.axhline(0, color="#333", lw=0.6)
    ax.axhline(1, color="#E63946", lw=0.9, ls="--")
    n = len(h)
    ax.text(n - 0.45, 1, "Hindsight\nmodel", color="#E63946", fontsize=6.5, ha="left", va="center", clip_on=False)
    ax.text(n - 0.45, 0, "BToM", color="#333", fontsize=6.5, ha="left", va="center", clip_on=False)
    if cb is not None:
        ax.axhline(cb, color="#555555", lw=0.9, ls=":")
        ax.text(n - 0.45, cb, f"Current\nbelief ({cb:.2f})", color="#555555", fontsize=6.5, ha="left", va="center",
                clip_on=False)
        ax.set_ylim(-0.1, max(3.45, h.ci_high.max() + 0.1))
    ax.set_xticks(range(n))
    ax.set_xticklabels([c.replace("-", "-\n", 1) for c in h.condition], fontsize=6)
    ax.set_xlim(-0.6, n - 0.5)
    ax.tick_params(axis="y", labelsize=6.5, length=2)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Hindsight index (belief)", fontsize=7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", lw=0.5)
    ax.set_axisbelow(True)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C_BETTER, label="Better-performing"), Patch(color=C_WORSE, label="Lower-performing")],
              fontsize=6.5, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=300, bbox_inches="tight")
    h = h.assign(group=["better" if c in a.better else "lower" for c in h.condition], current_belief_index=cb)
    h[["condition", "group", "estimate", "ci_low", "ci_high", "current_belief_index"]].to_csv(os.path.splitext(a.out)[0] + ".csv", index=False)
    print("saved", a.out)


if __name__ == "__main__":
    main()
