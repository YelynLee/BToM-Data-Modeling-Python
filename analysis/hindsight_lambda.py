"""
Hindsight λ 적합과 민감도 곡선.

원리: MATLAB score_momdp_path.m에서 Hindsight는 T >= t_vis일 때 초기 belief 후보 b의 smoothing 로그 점수에
lambda_HB * log b(참 세계)를 더한다. 따라서 어떤 λ0로 계산한 초기 belief 사후확률 P_λ0(b | x_1:T)가 있으면
    P_λ(b | x_1:T) ∝ P_λ0(b | x_1:T) * b(참)^(λ - λ0)      (T >= t_vis, T > 1)
로 임의의 λ를 정확히 얻는다. β = 2.5 작업공간(λ0 = 5)에서 λ = 0, 0.5, 1, 2와 BToM every-step 파일을
최대 차이 1e-5(belief 격자 반올림 수준)로 재현함.

입력 (둘 중 하나)
  - matlab/export_belief_posterior.m 출력: data/hindsight/posterior_b<β>.mat (logp, points, wi, t_vis, lambda_used)
  - btom_results 실행 직후 저장한 MATLAB 작업공간 .mat (smooth_joint_post, belief_sub, scenario, t_vis_array,
    beta_score, lambda_HB). 78개 시나리오가 모두 채워진 것만 사용 가능

사용 예)
  python analysis/hindsight_lambda.py --posterior ../BToM-MATLAB-Resource/hindsight_야매용_저장.mat \\
     --runs "Opus Every|results/claude-opus-4-6/vanilla/everystep/everystep_valid_only.csv" \\
            "Opus Prefix|results/claude-opus-4-6/vanilla/prefixstep/thinking_off" \\
     --out_dir results/stats/lambda
출력: lambda_curves.csv (λ별 pooled r), lambda_fit.csv (λ̂과 시나리오 bootstrap CI), lambda_curves.png
"""
import os
import sys
import io
import argparse
import contextlib
import pickle
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.special import logsumexp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for q in (ROOT, os.path.join(ROOT, "analysis"), os.path.join(ROOT, "src")):
    if q not in sys.path:
        sys.path.insert(0, q)

LAMS = np.r_[0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
BEL = ["belief_L", "belief_M", "belief_Empty"]


# ---------------------------------------------------------------- 사후확률 읽기
def load_posterior(path):
    """Returns (scenarios, beta, lambda0). scenarios[ns] = dict(pts, wi, tvis, logp (T x n_point))"""
    names = sio.whosmat(path)
    keys = {n for n, *_ in names}
    if "logp" in keys:
        m = sio.loadmat(path, squeeze_me=True)
        out = []
        for ns in range(len(m["logp"])):
            lp = np.atleast_2d(m["logp"][ns])
            pts = np.atleast_2d(m["points"][ns])
            if lp.shape[0] != pts.shape[0]:
                lp = lp.T
            out.append(dict(pts=pts, wi=int(m["wi"][ns]) - 1, tvis=int(m["t_vis"][ns]), logp=lp.T))
        return out, float(m["beta_score"]), float(m["lambda_used"])
    m = sio.loadmat(path, squeeze_me=True, variable_names=["smooth_joint_post", "belief_sub", "t_vis_array",
                                                           "scenario", "beta_score", "lambda_HB"])
    out = []
    for ns in range(78):
        sc = m["scenario"][ns]
        T = len(np.asarray(sc["path"].item()))
        if np.size(m["belief_sub"][ns]) == 0:
            raise ValueError(f"scenario {ns + 1} is empty in {path} (interrupted run)")
        wi = int(np.asarray(sc["condition"].item())[1]) - 1
        cells = [m["smooth_joint_post"][ns]] if T == 1 else list(m["smooth_joint_post"][ns])
        pts = np.atleast_2d(m["belief_sub"][ns] if T == 1 else m["belief_sub"][ns][0])
        lp = []
        for c in cells:
            c1 = c if (isinstance(c, np.ndarray) and c.dtype != object) else (c[0] if c.ndim else c.item())
            c1 = np.atleast_2d(c1)
            if c1.shape[1] != pts.shape[0] and c1.shape[0] == pts.shape[0]:
                c1 = c1.T
            lp.append(logsumexp(c1, axis=0))
        out.append(dict(pts=pts, wi=wi, tvis=int(m["t_vis_array"][ns]), logp=np.array(lp)))
    return out, float(m["beta_score"]), float(m["lambda_HB"])


def belief_marg(post, lam0, lam):
    """λ에 대한 시나리오별 belief 주변분포 (n_world x T)."""
    res = []
    for d in post:
        lt = np.log(d["pts"][:, d["wi"]] + 1e-10)
        B = []
        for t, lp in enumerate(d["logp"]):
            if t + 1 >= d["tvis"] and t + 1 > 1:
                lp = lp + (lam - lam0) * lt
            p = np.exp(lp - logsumexp(lp))
            B.append(p @ d["pts"])
        res.append(np.array(B).T)
    return res


# ---------------------------------------------------------------- 단계별 예측과 LLM 정렬
class Aligner:
    def __init__(self):
        with contextlib.redirect_stdout(io.StringIO()):
            import compare_stepwise_bootstrap as C
            from src.prepare_everystep import everystep_column
        self.C, self.col = C, everystep_column
        allowed = C.get_group_indices(include_irrational=False)
        self.scen = sorted(s for g in range(1, 8) for s in allowed[g - 1])
        self.rows = C.row_table(self.scen)
        self.base = C.load_reference_everystep("btom")
        self.Tmax = C.df_btom.groupby("scenario_id").time_step.max().to_dict()

    def predict(self, marg):
        r = self.base.copy()
        r[BEL] = np.array([marg[s - 1][:, self.col(t, self.Tmax[s], marg[s - 1].shape[1])]
                           for s, t in zip(r.scenario_id, r.time_step)])
        return self.C.ref_arrays(self.rows, {"h": r})["h"][:, self.C.BEL]

    def ratings(self, path):
        arr, _ = self.C.subject_array(self.C.load_rows(path), self.rows)
        return self.C.aggregate(arr)[:, self.C.BEL]


def pooled_r(a, p, idx=None):
    ok = ~np.isnan(a).any(1) & ~np.isnan(p).any(1)
    if idx is not None:
        ok = ok & idx
    return np.corrcoef(a[ok].ravel(), p[ok].ravel())[0, 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--posterior", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help='"이름|경로" (subject 폴더 또는 everystep_valid_only.csv)')
    ap.add_argument("--human", action="store_true", help="사람 End-step belief 곡선도 계산")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    post, beta, lam0 = load_posterior(a.posterior)
    print(f"posterior: beta = {beta}, computed at lambda = {lam0}")
    al = Aligner()
    preds = {l: al.predict(belief_marg(post, lam0, l)) for l in LAMS}
    sid = al.rows.scenario_id.to_numpy()
    uniq = np.unique(sid)
    rng = np.random.default_rng(a.seed)
    boot = [rng.choice(uniq, len(uniq)) for _ in range(a.n_boot)]
    curves, fits = [], []
    for spec in a.runs:
        name, path = spec.split("|", 1)
        y = al.ratings(path)
        rs = np.array([pooled_r(y, preds[l]) for l in LAMS])
        curves += [dict(run=name, beta=beta, lam=l, r=r) for l, r in zip(LAMS, rs)]
        # 시나리오 bootstrap: 리샘플마다 r이 최대인 λ (행을 시나리오 단위로 복제)
        pos = {s: np.where(sid == s)[0] for s in uniq}
        lam_b = []
        for b in boot:
            ix = np.concatenate([pos[s] for s in b])
            yy = y[ix]
            lam_b.append(LAMS[int(np.argmax([pooled_r(yy, preds[l][ix]) for l in LAMS]))])
        lam_b = np.array(lam_b)
        best = LAMS[int(np.argmax(rs))]
        fits.append(dict(run=name, beta=beta, lambda_hat=best, r_at_hat=rs.max(), r_at_lambda2=rs[list(LAMS).index(2)],
                         r_at_lambda0=rs[0], ci_low=np.percentile(lam_b, 2.5), ci_high=np.percentile(lam_b, 97.5),
                         share_at_upper_bound=np.mean(lam_b == LAMS[-1])))
        print(f"{name}: λ̂ = {best} (r = {rs.max():.3f}; λ=2: {rs[list(LAMS).index(2)]:.3f}), "
              f"95% CI [{np.percentile(lam_b, 2.5)}, {np.percentile(lam_b, 97.5)}]")
    if a.human:
        from src.config import HUMAN_PKL_PATH
        with open(HUMAN_PKL_PATH, "rb") as f:
            H = pickle.load(f)["bel_inf_mean_norm"]
        ix = np.array(al.scen) - 1
        for l in LAMS:
            E = np.array([x[:, -1] for x in belief_marg(post, lam0, l)]).T
            curves.append(dict(run="Human (End-step)", beta=beta, lam=l,
                               r=np.corrcoef(E[:, ix].ravel(), H[:, ix].ravel())[0, 1]))
    C = pd.DataFrame(curves)
    C.to_csv(os.path.join(a.out_dir, "lambda_curves.csv"), index=False)
    pd.DataFrame(fits).to_csv(os.path.join(a.out_dir, "lambda_fit.csv"), index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for name, g in C.groupby("run", sort=False):
        ax.plot(g.lam + 0.1, g.r, marker="o", ms=3, lw=1.3, label=name)
    ax.axvline(2.1, color="#888", ls="--", lw=0.8)
    ax.text(2.2, ax.get_ylim()[0] + 0.02, "λ = 2 (paper)", fontsize=7, color="#555")
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.35, 0.6, 1.1, 2.1, 4.1, 8.1, 16.1, 32.1, 64.1])
    ax.set_xticklabels(["0", "0.25", "0.5", "1", "2", "4", "8", "16", "32", "64"], fontsize=7)
    ax.set_xlabel("λ (Hindsight weight on the revealed state)", fontsize=8)
    ax.set_ylabel("r(ratings, Hindsight$_\\lambda$), belief", fontsize=8)
    ax.tick_params(labelsize=7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=6.5, frameon=False, loc="best")
    fig.savefig(os.path.join(a.out_dir, "lambda_curves.png"), dpi=250, bbox_inches="tight")
    print("saved", a.out_dir)


if __name__ == "__main__":
    main()
