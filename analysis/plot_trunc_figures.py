"""
리뷰어 응답용 그림 2개 (Single-trunc를 포함한 조건 분해)

  R0 : 조건 설계 도식 (End-step, Every-step, Every-step-single, Single-trunc, Prefix-step). 데이터 입력 없음
  R2 : 한 모델의 네 조건(Every-step, Every-step-single, Single-trunc, Prefix-step), 1x4
       A hindsight 지수   B 참 상태 쪽 pull (Spot 2 보이기 전 / 보인 뒤 / 끝내 안 보임)
       C 마지막 스텝 r(같은 모델 End-step)   D 마지막 스텝 r(사람)
  R3 : 모델별 hindsight 지수 감소(Every-step -> Prefix-step)의 단계별 분해
       출력 형식(Every - Single), 정보(Single - Single-trunc), 문구(Single-trunc - Prefix).
       Single-trunc가 없는 모델은 정보 + 문구(Single - Prefix)로 합쳐 표시. 시나리오 bootstrap CI는 csv로 저장

입력
  --r2_dir   compare_stepwise_bootstrap(4조건), final_step_compare, truth_leak_explore 출력이 모두 있는 폴더
  --r2_model 그림 제목에 쓸 모델 이름
  --models / --compare   R3용. 모델 이름과 각 모델의 compare 폴더(stepwise_scenario_level.csv)

사용 예)
  python analysis/plot_trunc_figures.py \\
     --r2_dir results/claude-opus-4-6/vanilla/compare_4cond --r2_model "Claude Opus 4.6 (thinking off)" \\
     --models "Claude Opus 4.6" "Gemini 2.5 Pro" "GPT-5.4" \\
     --compare results/claude-opus-4-6/vanilla/compare_4cond results/gemini-2.5-pro/vanilla/compare_2x2 \\
               results/gpt-5.4/vanilla/compare_2x2 \\
     --out_dir results/reviewer
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

CONDS = ["Every-step", "Every-step-single", "Single-trunc", "Prefix-step"]
TICK = ["Every", "Single", "Single-\ntrunc", "Prefix"]
COL = {"Every-step": "#888780", "Every-step-single": "#D85A30", "Single-trunc": "#BA7517", "Prefix-step": "#1D9E75"}
SEG = [("pre", "before Spot 2\nvisible"), ("post", "after Spot 2\nvisible"), ("never", "Spot 2 never\nvisible (control)")]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def _bars(ax, est, lo, hi):
    for i, c in enumerate(CONDS):
        ax.bar(i, est[i], color=COL[c], width=0.72)
        ax.errorbar(i, est[i], [[est[i] - lo[i]], [hi[i] - est[i]]], color="k", lw=1, capsize=3)
    ax.set_xticks(range(len(CONDS)), TICK)
    ax.axhline(0, color="k", lw=0.6)


def fig_r0(out, T=8, tq=4):
    """질문 시점 t = tq를 예시로, 각 조건이 맥락에 넣는 로그와 질문을 그림."""
    rows = [("End-step", "full", [T], "End", "1 (t = T)"),
            ("Every-step", "full", list(range(1, T + 1)), "Every", "T (all steps)"),
            ("Every-step-single", "full", [tq], "Every", "1 (step t)"),
            ("Single-trunc", "trunc", [tq], "Every", "1 (step t)"),
            ("Prefix-step", "trunc", [tq], "End", "1 (step t)")]
    wc = {"End": "#1D9E75", "Every": "#D85A30"}
    fig, ax = plt.subplots(figsize=(12.5, 4.2))
    for i, (name, log, qs, word, nj) in enumerate(rows):
        y = len(rows) - 1 - i
        for t in range(1, T + 1):
            shown = log == "full" or t <= tq
            ax.add_patch(plt.Rectangle((t - 0.45, y - 0.3), 0.9, 0.6, facecolor="#cfd8e3" if shown else "white",
                                       edgecolor="#8a96a3" if shown else "#c9c9c9", ls="-" if shown else "--", lw=0.8))
        for q in qs:
            ax.plot(q, y + 0.42, marker="v", ms=8, color=wc[word], clip_on=False)
        ax.text(0.2, y, name, ha="right", va="center", fontsize=10, fontweight="bold")
        ax.text(T + 0.9, y, "full (1..T)" if log == "full" else "1..t", va="center", fontsize=9)
        ax.text(T + 2.6, y, nj, va="center", fontsize=9)
        ax.text(T + 4.6, y, word, va="center", fontsize=9, color=wc[word])
    for x, h in [(T + 0.9, "Log in context"), (T + 2.6, "Judgments per call"), (T + 4.6, "Question wording")]:
        ax.text(x, len(rows) - 0.25, h, fontsize=9, fontweight="bold")
    ax.set_xticks(range(1, T + 1), [{1: "1", tq: "t", tq + 1: "t+1", T: "T"}.get(t, "") for t in range(1, T + 1)])
    ax.set_xlabel("time step (example: question about step t)")
    pairs = [(3, 2, "output format"), (2, 1, "information"), (1, 0, "wording")]
    for a_, b_, lab in pairs:
        ax.annotate("", xy=(-2.75, b_), xytext=(-2.75, a_), arrowprops=dict(arrowstyle="<->", lw=0.8))
        ax.text(-2.85, (a_ + b_) / 2, lab, ha="right", va="center", fontsize=8.5, style="italic")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Patch(facecolor="#cfd8e3", edgecolor="#8a96a3", label="step in context"),
                       Patch(facecolor="white", edgecolor="#c9c9c9", ls="--", label="step not shown"),
                       Line2D([], [], marker="v", ls="", color=wc["End"], label="question, End-step wording"),
                       Line2D([], [], marker="v", ls="", color=wc["Every"], label="question, Every-step wording")],
              loc="upper center", bbox_to_anchor=(0.45, -0.18), ncol=4, frameon=False, fontsize=8.5)
    ax.set_xlim(-5.2, T + 6.6)
    ax.set_ylim(-0.6, len(rows) - 0.05)
    ax.set_yticks([])
    for sp in ("left", "top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ saved: {out}")


def fig_r2(d, model, out):
    s = pd.read_csv(os.path.join(d, "stepwise_bootstrap_summary.csv"))
    f = pd.read_csv(os.path.join(d, "final_step.csv"))
    pull = pd.read_csv(os.path.join(d, "truth_leak_index.csv"))
    fig, ax = plt.subplots(1, 4, figsize=(18.5, 4.3))

    hi_rows = [s[(s.condition == c) & (s.metric == "hindsight_index")].iloc[0] for c in CONDS]
    est = [r.estimate for r in hi_rows]
    _bars(ax[0], est, [r.ci_low for r in hi_rows], [r.ci_high for r in hi_rows])
    ax[0].axhline(1, ls="--", c="gray", lw=0.8)
    top = max(r.ci_high for r in hi_rows)
    ax[0].set_ylim(0, top * 1.13)
    for a_, b_, lab in [(0, 1, "format"), (1, 2, "information"), (2, 3, "wording")]:
        y = max(hi_rows[a_].ci_high, hi_rows[b_].ci_high) + top * 0.035
        ax[0].annotate("", xy=(b_, y), xytext=(a_, y), arrowprops=dict(arrowstyle="->", lw=0.8))
        ax[0].text((a_ + b_) / 2, y + top * 0.015, lab, ha="center", fontsize=8)
    ax[0].set_ylabel("Hindsight index (belief, t ≥ t_vis)")
    ax[0].set_title("A. Hindsight index\n0 = BToM, 1 = Hindsight (λ = 2)", fontsize=10)

    w = 0.2
    for j, c in enumerate(CONDS):
        for i, (seg, _) in enumerate(SEG):
            r = pull[(pull.segment == seg) & (pull.condition == c)].iloc[0]
            x = i + (j - 1.5) * w
            ax[1].bar(x, r.estimate, w, color=COL[c], label=TICK[j].replace("\n", "") if i == 0 else None)
            ax[1].errorbar(x, r.estimate, [[r.estimate - r.ci_low], [r.ci_high - r.estimate]], color="k", lw=0.8, capsize=2)
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_xticks(range(3), [l for _, l in SEG], fontsize=9)
    ax[1].legend(fontsize=8, frameon=False)
    ax[1].set_ylabel("Pull toward true Spot-2 content (belief)")
    ax[1].set_title("B. Pull toward the true Spot-2 content\n0 = BToM, 1 = certain of truth (= TrueBelief)", fontsize=10)

    fb = f[f.dimension == "belief"]
    for k, (versus, title, ylab) in enumerate([
            ("End-step", "C. Agreement with own End-step ratings", "r with own End-step (belief, t = T)"),
            ("Human", "D. Agreement with human ratings", "r with human (belief, t = T)")]):
        g = fb[fb.versus == versus].set_index("condition")
        a = ax[2 + k]
        _bars(a, [g.loc[c, "r"] for c in CONDS], [g.loc[c, "ci_low"] for c in CONDS], [g.loc[c, "ci_high"] for c in CONDS])
        a.set_ylim(-0.5, 1.0)
        a.set_ylabel(ylab)
        a.set_title(title, fontsize=10)
    ax[2].text(3.45, -0.47, "Prefix at t = T uses the End-step prompt\n(run-to-run reliability)", fontsize=8,
               color="gray", ha="right", va="bottom")
    gh = fb[fb.versus == "Human"].set_index("condition")
    if "End-step" in gh.index:
        v = gh.loc["End-step", "r"]
        ax[3].axhline(v, ls="--", c=COL["Prefix-step"], lw=0.9)
        ax[3].text(-0.45, v - 0.07, f"End-step {v:.2f}", fontsize=8, color=COL["Prefix-step"])
    if "BToM" in gh.index:
        v = gh.loc["BToM", "r"]
        ax[3].axhline(v, ls=":", c="gray", lw=0.9)
        ax[3].text(-0.45, v + 0.03, f"BToM–human {v:.2f}", fontsize=8, color="gray")
    fig.suptitle(f"{model}: belief ratings, five runs per condition", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ saved: {out}")


def scenario_hi(d):
    x = pd.read_csv(os.path.join(d, "stepwise_scenario_level.csv"))
    x = x[x.metric == "hindsight_index"]
    cols = {}
    for c in CONDS:
        if c in x.columns:
            cols[c] = x.dropna(subset=[c]).groupby("scenario_id")[c].first()
    return pd.DataFrame(cols).dropna()


def fig_r3(models, dirs, out, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for m, d in zip(models, dirs):
        df = scenario_hi(d)
        if "Single-trunc" in df:
            steps = [("output format", "Every-step", "Every-step-single"), ("information", "Every-step-single", "Single-trunc"),
                     ("wording", "Single-trunc", "Prefix-step")]
        else:
            steps = [("output format", "Every-step", "Every-step-single"),
                     ("information + wording", "Every-step-single", "Prefix-step")]

        def calc(x):
            tot = x["Every-step"].mean() - x["Prefix-step"].mean()
            return {k: (x[a].mean() - x[b].mean(), (x[a].mean() - x[b].mean()) / tot) for k, a, b in steps} | {"total": (tot, 1.0)}
        est = calc(df)
        bs = [calc(df.iloc[rng.integers(0, len(df), len(df))]) for _ in range(n_boot)]
        for k in est:
            v = np.array([b[k] for b in bs])
            rows.append(dict(model=m, term=k, estimate=est[k][0], ci_low=np.percentile(v[:, 0], 2.5),
                             ci_high=np.percentile(v[:, 0], 97.5), share=est[k][1],
                             share_ci_low=np.percentile(v[:, 1], 2.5), share_ci_high=np.percentile(v[:, 1], 97.5),
                             n_scenarios=len(df)))
    R = pd.DataFrame(rows)
    R.to_csv(out.replace(".png", ".csv"), index=False)

    cc = {"output format": "#D85A30", "information": "#BA7517", "wording": "#1D9E75", "information + wording": "#5F8F7A"}
    fig, ax = plt.subplots(figsize=(8.8, 0.85 * len(models) + 0.9))
    for i, m in enumerate(models):
        pos = neg = 0.0
        for _, r in R[(R.model == m) & (R.term != "total")].iterrows():
            if r.estimate >= 0:
                ax.barh(i, r.estimate, left=pos, color=cc[r.term], edgecolor="white")
                ax.text(pos + r.estimate / 2, i, f"{r.estimate:.2f}\n({r.share:.0%})", ha="center", va="center",
                        fontsize=7.5, color="white")
                pos += r.estimate
            else:
                ax.barh(i, r.estimate, left=neg, color=cc[r.term], edgecolor="white", hatch="//")
                neg += r.estimate
    ax.set_yticks(range(len(models)), models)
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("Reduction in hindsight index, Every-step → Prefix-step (belief)")
    used = [k for k in cc if k in set(R.term)]
    ax.legend(handles=[Patch(color=cc[k], label=k) for k in used], fontsize=8, frameon=False, ncol=len(used),
              loc="upper center", bbox_to_anchor=(0.5, 1.25))
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ saved: {out} (+ .csv)")
    print(R.round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--r2_dir")
    ap.add_argument("--r2_model", default="")
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--compare", nargs="+")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--design", action="store_true", help="R0 조건 설계 도식도 저장")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    if a.design:
        fig_r0(os.path.join(a.out_dir, "R0_condition_design.png"))
    if a.r2_dir:
        fig_r2(a.r2_dir, a.r2_model, os.path.join(a.out_dir, "R2_opus_four_conditions.png"))
    if a.models:
        fig_r3(a.models, a.compare, os.path.join(a.out_dir, "R3_factor_decomposition.png"))


if __name__ == "__main__":
    main()
