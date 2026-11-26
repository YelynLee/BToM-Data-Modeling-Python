"""
λ 민감도 그림 (Every-step 16회, 8개 모델 + 사람 End-step).
(a) r(평정, Hindsight_λ) 곡선, (b) 모델별 r: λ = 0 (BToM), λ = 2 (논문 Hindsight), λ̂.

입력: hindsight_lambda.py 출력 폴더 (lambda_curves.csv, lambda_fit.csv)
  python analysis/plot_lambda_curves.py --dir results/stats/lambda_everystep \\
     --better "Claude-4.6-Opus" "Gemini-2.5-Pro" "DeepSeek-R1" "DeepSeek-V4-Pro"
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

C_BETTER, C_WORSE, C_HUMAN = "#60A917", "#D80073", "#222222"
INF_X = 4096


def xpos(l):
    return np.where(np.isinf(l), INF_X, l) + 0.1


def lam_txt(l):
    return "∞" if np.isinf(l) else f"{l:g}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--better", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--only_a", action="store_true", help="곡선 패널 (a)만 그림 (논문 부록용)")
    a = ap.parse_args()
    C = pd.read_csv(os.path.join(a.dir, "lambda_curves.csv"))
    F = pd.read_csv(os.path.join(a.dir, "lambda_fit.csv")).sort_values("r_at_hat", ascending=False)
    markers = ["o", "s", "^", "D"]
    if a.only_a:
        fig, ax = plt.subplots(figsize=(5.4, 3.0))
        bx = None
    else:
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 3.1), gridspec_kw=dict(width_ratios=[1.25, 1]))

    # (a) 곡선
    ib = iw = 0
    for name, g in C.groupby("run", sort=False):
        x = xpos(g.lam.to_numpy())
        if name.startswith("Human"):
            ax.plot(x, g.r, color=C_HUMAN, lw=1.8, ls="--", zorder=5)
            ax.text(x[0] * 1.05, g.r.iloc[0] + 0.04, "Human (End-step)", fontsize=6.5, color=C_HUMAN)
            continue
        better = name in a.better
        col = C_BETTER if better else C_WORSE
        mk = markers[(ib if better else iw) % 4]
        ib, iw = ib + better, iw + (not better)
        ax.plot(x, g.r, color=col, lw=1.1, marker=mk, ms=2.6, alpha=0.9, label=name)
    ax.axvline(2.1, color="#888", ls=":", lw=0.9)
    ax.text(2.3, -0.33, "λ = 2\n(paper)", fontsize=6, color="#555", va="bottom")
    ax.axhline(0, color="#bbb", lw=0.5)
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.6, 2.1, 8.1, 32.1, 128.1, 1024.1, INF_X + 0.1])
    ax.set_xticklabels(["0", "0.5", "2", "8", "32", "128", "1024", "∞"], fontsize=7)
    ax.set_xlabel("λ (Hindsight weight on the revealed world)", fontsize=8)
    ax.set_ylabel("r(ratings, Hindsight$_\\lambda$), belief", fontsize=8)
    ax.set_ylim(-0.35, 1.02)
    ax.tick_params(labelsize=7)
    if not a.only_a:
        ax.set_title("(a) Fit across λ", fontsize=8.5, loc="left")
    ax.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4)

    # (b) 모델별 요약
    if a.only_a:
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(color="#eeeeee", lw=0.5)
        ax.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig(a.out or os.path.join(a.dir, "lambda_curves_a.png"), dpi=300, bbox_inches="tight")
        print("saved")
        return
    y = np.arange(len(F))[::-1]
    for yi, (_, r) in zip(y, F.iterrows()):
        col = C_BETTER if r.run in a.better else C_WORSE
        ax_ = bx
        ax_.plot([r.r_at_lambda0, r.r_at_hat], [yi, yi], color="#d0d0d0", lw=1, zorder=1)
        ax_.scatter(r.r_at_lambda0, yi, marker="o", s=22, facecolor="white", edgecolor="#555", lw=0.9, zorder=3)
        ax_.scatter(r.r_at_lambda2, yi, marker="|", s=70, color="#555", lw=1.4, zorder=3)
        ax_.scatter(r.r_at_hat, yi, marker="o", s=26, color=col, zorder=4)
        lo, hi = lam_txt(r.ci_low), lam_txt(r.ci_high)
        sat = "" if r.lambda_hat < 128 else " (≈∞)"
        ax_.text(1.03, yi, f"best λ = {lam_txt(r.lambda_hat)}{sat} [{lo}, {hi}]", fontsize=6, va="center")
    bx.set_yticks(y)
    bx.set_yticklabels(F.run, fontsize=6.5)
    bx.set_xlim(-0.35, 1.0)
    bx.axvline(0, color="#bbb", lw=0.5)
    bx.set_xlabel("r(ratings, model), belief", fontsize=8)
    bx.tick_params(labelsize=7)
    bx.set_title("(b) BToM (λ = 0), Hindsight (λ = 2), best λ", fontsize=8.5, loc="left")
    bx.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec="#555", label="λ = 0 (BToM)"),
                       Line2D([], [], marker="|", ls="", color="#555", ms=8, mew=1.4, label="λ = 2"),
                       Line2D([], [], marker="o", ls="", color="#777", label="best λ [95% CI]")],
              fontsize=6, frameon=False, loc="lower left", bbox_to_anchor=(0, -0.4), ncol=3)
    for z in (ax, bx):
        z.grid(color="#eeeeee", lw=0.5)
        z.set_axisbelow(True)
        for sp in ("top", "right"):
            z.spines[sp].set_visible(False)
    fig.tight_layout()
    out = a.out or os.path.join(a.dir, "lambda_everystep.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
