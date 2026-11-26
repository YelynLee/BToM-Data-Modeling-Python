"""
사람 End-step 평정에 대한 β 민감도 (원 논문 correlation_analysis.m 기준과 동일).

기준: 제외 시나리오 [11 12 22 71 72] 뺀 73개, pooled r(belief) + r(desire) 최대인 β.
  - BToM / TrueBelief / NoCost: data/<model>_results_complete.mat (β 0.5..10, 20개)
  - Hindsight: data/hindsight/results_b<β>_l<λ>.mat 격자 (있는 칸만)
β̂ 신뢰구간은 시나리오 bootstrap (리샘플마다 r_b + r_d 최대인 β).

  python analysis/beta_sensitivity_human.py --matlab_root ../BToM-MATLAB-Resource --out_dir results/stats/beta_sensitivity
출력: beta_curves.csv, beta_fit.csv, hindsight_beta_lambda_grid.csv, beta_sensitivity.png
"""
import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import scipy.io as sio

EXCL = [11, 12, 22, 71, 72]
INCL = np.setdiff1d(np.arange(1, 79), EXCL) - 1
COLORS = {"BToM": "#1f4e9c", "TrueBelief": "#e08a00", "NoCost": "#7a7a7a", "Hindsight (λ = 2)": "#E63946"}


def r_pair(B, D, hb, hd, cols):
    rb = np.corrcoef(B[:, cols].ravel(), hb[:, cols].ravel())[0, 1]
    rd = np.corrcoef(D[:, cols].ravel(), hd[:, cols].ravel())[0, 1]
    return rb, rd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matlab_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    d = os.path.join(a.matlab_root, "data")
    H = sio.loadmat(os.path.join(d, "human_data.mat"))
    hb, hd = H["bel_inf_mean_norm"], H["des_inf_mean"]

    # 모델별 β 곡선: dict name -> (betas, B[β](3x78), D[β])
    curves = {}
    for m, name in [("btom", "BToM"), ("truebelief", "TrueBelief"), ("nocost", "NoCost")]:
        R = sio.loadmat(os.path.join(d, f"{m}_results_complete.mat"))
        betas = R["beta_score_values"].ravel()
        curves[name] = (betas, [R["belief_model"][:, :, i] for i in range(len(betas))],
                        [R["desire_model"][:, :, i] for i in range(len(betas))])
    grid = {}
    for f in glob.glob(os.path.join(d, "hindsight", "results_b*_l*.mat")):
        b, l = map(float, re.findall(r"results_b([\d.]+)_l([\d.]+)\.mat", f)[0])
        R = sio.loadmat(f)
        grid[(b, l)] = (R["belief_model"], R["desire_model"])
    hb2 = sorted(b for (b, l) in grid if l == 2.0)
    curves["Hindsight (λ = 2)"] = (np.array(hb2), [grid[(b, 2.0)][0] for b in hb2], [grid[(b, 2.0)][1] for b in hb2])

    rng = np.random.default_rng(a.seed)
    boot = [rng.choice(INCL, len(INCL)) for _ in range(a.n_boot)]
    rows, fits = [], []
    for name, (betas, Bs, Ds) in curves.items():
        rr = np.array([r_pair(B, D, hb, hd, INCL) for B, D in zip(Bs, Ds)])
        for be, (rb, rd) in zip(betas, rr):
            rows.append(dict(model=name, beta=be, r_belief=rb, r_desire=rd, r_sum=rb + rd))
        k = int(np.argmax(rr.sum(1)))
        bh = []
        for ix in boot:
            bh.append(betas[int(np.argmax([sum(r_pair(B, D, hb, hd, ix)) for B, D in zip(Bs, Ds)]))])
        bh = np.array(bh)
        fits.append(dict(model=name, beta_hat=betas[k], r_belief=rr[k, 0], r_desire=rr[k, 1],
                         beta_hat_belief_only=betas[int(np.argmax(rr[:, 0]))],
                         beta_hat_desire_only=betas[int(np.argmax(rr[:, 1]))],
                         ci_low=np.percentile(bh, 2.5), ci_high=np.percentile(bh, 97.5),
                         r_sum_range_within_ci=f"{rr[(betas >= np.percentile(bh, 2.5)) & (betas <= np.percentile(bh, 97.5)), :].sum(1).min():.3f}"
                                               f"-{rr[k].sum():.3f}"))
        print(f"{name}: β̂ = {betas[k]} [{np.percentile(bh, 2.5)}, {np.percentile(bh, 97.5)}], "
              f"r_b {rr[k, 0]:.3f}, r_d {rr[k, 1]:.3f}")
    C = pd.DataFrame(rows)
    C.to_csv(os.path.join(a.out_dir, "beta_curves.csv"), index=False)
    pd.DataFrame(fits).to_csv(os.path.join(a.out_dir, "beta_fit.csv"), index=False)
    G = pd.DataFrame([dict(beta=b, lam=l, **dict(zip(["r_belief", "r_desire"], r_pair(B, D, hb, hd, INCL))))
                      for (b, l), (B, D) in sorted(grid.items())])
    G["r_sum"] = G.r_belief + G.r_desire
    G.to_csv(os.path.join(a.out_dir, "hindsight_beta_lambda_grid.csv"), index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(10, 2.8))
    for ax, col, ttl in zip(axs, ["r_belief", "r_desire", "r_sum"],
                            ["Belief: r(model, human)", "Desire: r(model, human)", "Fit criterion: r$_b$ + r$_d$"]):
        for name, g in C.groupby("model", sort=False):
            ls = (0, (3, 2)) if (name == "BToM" and col == "r_desire") else "-"
            ax.plot(g.beta, g[col], marker="o", ms=2.5, lw=1.3, color=COLORS[name], label=name, ls=ls,
                    zorder=4 if name == "BToM" else 3)
            if col == "r_sum":
                k = g[col].idxmax()
                ax.plot(g.beta[k], g[col][k], marker="*", ms=9, color=COLORS[name], mec="white", mew=0.6)
        ax.set_title(ttl, fontsize=8)
        ax.set_xlabel("β (agent rationality)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(color="#e6e6e6", lw=0.5)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axs[0].legend(fontsize=6.5, frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.36))
    axs[1].text(0.97, 0.97, "Hindsight desire = BToM desire\n(λ affects only beliefs)", transform=axs[1].transAxes,
                fontsize=6, ha="right", va="top", color="#555")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, "beta_sensitivity.png"), dpi=250, bbox_inches="tight")
    print("saved", a.out_dir)


if __name__ == "__main__":
    main()
