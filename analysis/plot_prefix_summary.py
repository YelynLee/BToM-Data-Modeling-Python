"""
논문 Figure 6: Every-step vs Prefix-step 요약 (모델 3개, belief).

  (a) 참 Spot-2 내용 쪽으로의 pull, agent가 Spot 2를 보기 전   -> TrueBelief(관찰자 지식 투사)의 표지
  (b) 같은 pull, agent가 Spot 2를 본 뒤                         -> Hindsight / 현재 belief 귀속의 표지
  (c) 마지막 스텝 belief와 사람 평정의 r (Every, Prefix, End-step 마커, BToM–human 점선)
  (d) 스텝 간 belief 변화의 BToM 일치도 r(ΔLLM, ΔBToM)

입력
  --compare  모델별 폴더 (final_step.csv, truth_leak_index.csv)
  --update   update_agreement.csv
사용 예)
  python analysis/plot_prefix_summary.py --models "Claude-4.6-Opus" "Gemini-2.5-Pro" "GPT-5.4" \\
     --update_names "Claude Opus 4.6" "Gemini 2.5 Pro" "GPT-5.4" \\
     --compare results/claude-opus-4-6/vanilla/compare_4cond results/gemini-2.5-pro/vanilla/compare_4cond \\
               results/gpt-5.4/vanilla/compare_2x2 \\
     --update results/stats/update/update_agreement.csv --out results/stats/figure6_prefix.png
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EVERY, PREFIX = "Every-step", "Prefix-step"
C_PREFIX, C_EVERY = "#1D9E75", "#888780"
TICK = {"Claude-4.6-Opus": "Claude\nOpus", "Gemini-2.5-Pro": "Gemini\nPro", "GPT-5.4": "GPT\n5.4"}


def bars(ax, x, vals, w=0.36):
    """vals: {cond: (est, lo, hi)}; Every = 빈 막대 + 빗금, Prefix = 채운 막대."""
    for k, (cond, off) in enumerate([(EVERY, -w / 2), (PREFIX, w / 2)]):
        if cond not in vals or vals[cond] is None:
            continue
        e, lo, hi = vals[cond]
        kw = dict(facecolor="white", edgecolor=C_EVERY, hatch="////", lw=0.8) if cond == EVERY else \
            dict(facecolor=C_PREFIX, edgecolor=C_PREFIX, lw=0.8)
        ax.bar(x + off, e, width=w - 0.04, **kw, label=cond if x == 0 else None)
        ax.errorbar(x + off, e, [[e - lo], [hi - e]], color="#333333", lw=0.8, capsize=1.8)


def style(ax, title, models, ylabel=None, ylim=None):
    ax.set_title(title, fontsize=7.5, loc="left")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([TICK.get(m, m) for m in models], fontsize=6)
    ax.set_xlim(-0.6, len(models) - 0.35)
    ax.tick_params(axis="y", labelsize=6.5, length=2)
    ax.tick_params(axis="x", length=0)
    ax.axhline(0, color="#333333", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", lw=0.5)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=7)
    if ylim:
        ax.set_ylim(*ylim)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--update_names", nargs="+", required=True, help="update_agreement.csv의 model 이름 (같은 순서)")
    ap.add_argument("--compare", nargs="+", required=True)
    ap.add_argument("--update", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    U = pd.read_csv(a.update)
    U = U[(U.reference == "BToM") & (U.phase_idx.astype(str) == "all")]

    fig, axs = plt.subplots(1, 4, figsize=(7.0, 2.05), gridspec_kw=dict(wspace=0.5))
    rows = []
    for i, (m, um, d) in enumerate(zip(a.models, a.update_names, a.compare)):
        tl = pd.read_csv(os.path.join(d, "truth_leak_index.csv"))
        fs = pd.read_csv(os.path.join(d, "final_step.csv"))
        fs = fs[(fs.dimension == "belief") & (fs.versus == "Human")]
        for j, seg in enumerate(["pre", "post"]):
            v = {c: tuple(tl[(tl.segment == seg) & (tl.condition == c)][["estimate", "ci_low", "ci_high"]].iloc[0])
                 for c in (EVERY, PREFIX) if ((tl.segment == seg) & (tl.condition == c)).any()}
            bars(axs[j], i, v)
            rows += [dict(model=m, panel=f"pull_{seg}", condition=c, estimate=x[0], ci_low=x[1], ci_high=x[2])
                     for c, x in v.items()]
        v = {c: tuple(fs[fs.condition == c][["r", "ci_low", "ci_high"]].iloc[0]) for c in (EVERY, PREFIX)}
        bars(axs[2], i, v)
        end = fs[fs.condition == "End-step"][["r", "ci_low", "ci_high"]].iloc[0]
        axs[2].plot(i + 0.47, end.r, marker="D", ms=3.5, color="#222222", zorder=5, label="End-step" if i == 0 else None)
        axs[2].errorbar(i + 0.47, end.r, [[end.r - end.ci_low], [end.ci_high - end.r]], color="#222222", lw=0.7)
        u = U[(U.model == um) & (U.dimension == "belief")]
        v = {c: tuple(u[u.condition == c][["r", "ci_low", "ci_high"]].iloc[0]) for c in (EVERY, PREFIX)}
        bars(axs[3], i, v)
        rows += [dict(model=m, panel=p, condition=c, estimate=x[0], ci_low=x[1], ci_high=x[2])
                 for p, vv in [("final_r_human", {c: tuple(fs[fs.condition == c][["r", "ci_low", "ci_high"]].iloc[0])
                                                  for c in (EVERY, PREFIX, "End-step")}), ("update_r_belief", v)]
                 for c, x in vv.items()]
    bt = fs[fs.condition == "BToM"].r.iloc[0]
    axs[2].axhline(bt, color="#555555", lw=0.7, ls="--")
    axs[2].text(2.75, bt - 0.09, "BToM–human", fontsize=5.5, ha="right", color="#555555")
    style(axs[0], "(a) Pull, before Spot 2 seen", a.models, "Pull toward true\nSpot-2 content", (-0.05, 1.05))
    style(axs[1], "(b) Pull, after Spot 2 seen", a.models, None, (-0.05, 1.05))
    style(axs[2], "(c) Final step vs. humans", a.models, "r (belief, t = T)", (-0.3, 1.0))
    style(axs[3], "(d) Belief updates vs. BToM", a.models, r"r($\Delta$LLM, $\Delta$BToM)", (-0.1, 0.5))
    h, l = axs[2].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 1.07))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=300, bbox_inches="tight")
    pd.DataFrame(rows).to_csv(os.path.splitext(a.out)[0] + ".csv", index=False)
    print("saved", a.out)


if __name__ == "__main__":
    main()
