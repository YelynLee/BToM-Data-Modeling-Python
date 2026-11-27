"""
증거 민감도 (evidence sensitivity): LLM의 스텝 간 변화가 BToM이 말하는 증거의 크기에 맞춰 달라지는가.

단위: 스텝 전환(t−1 -> t) x 옵션. ΔBToM = BToM의 변화 (증거의 크기와 방향), ΔLLM = LLM의 변화 (실행 평균).
1) 증거 강도 5분위: |ΔBToM|을 같은 개수의 5구간으로 나눠, 구간마다
     signed response = 평균 sign(ΔBToM)·ΔLLM   (BToM 방향으로 움직인 양, BToM 자신은 = 구간 평균 |ΔBToM|)
     movement        = 평균 |ΔLLM|             (방향과 무관한 움직임)
2) 요약 두 숫자 (모든 전환, 원래 단위)
     sensitivity     = signed response를 |ΔBToM|에 회귀한 기울기 (1 = BToM만큼 증거에 비례해 반응, 0 = 증거와 무관)
     baseline        = |ΔLLM|을 |ΔBToM|에 회귀한 절편 (증거가 없을 때도 움직이는 양)
3) 사건 구간별 반응 (같은 기준을 네 모델에 적용, 전환의 도착 phase로 구분)
     event response  = Σ sign(ΔBToM)·ΔLLM / Σ|ΔBToM|  (그 구간 전환 전체; 1 = BToM만큼)
4) relative closure (서술용) R = (Better − Worse)/(BToM − Worse), BToM 기준값 = 1 (sensitivity, event response)
CI: 시나리오 bootstrap (네 모델 같은 리샘플).

사용 예)
  python analysis/evidence_sensitivity.py --out_dir results/stats/prefix_2x2 \\
     --runs "Opus|Claude|better|results/claude-opus-4-6/vanilla/prefixstep" ...
"""
import os, sys, argparse, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "src"), os.path.join(ROOT, "analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)
warnings.filterwarnings("ignore")
from analysis.compare_stepwise_bootstrap import load_rows, row_table, ref_arrays, subject_array, aggregate
from analysis.prefix_2x2 import resolve, C_BETTER, C_WORSE
from analysis.phase_utils import get_group_phase_labels
from src.prepare_everystep import load_reference_everystep
from src.config import get_group_indices

SL = {"desire": range(0, 3), "belief": range(3, 6)}
EVENTS = ["Appr G1", "Pass G1", "See G2", "Stop", "Return G1", "Stop Btw", "Appr G2", "Selected"]
NB = 5


def wmean(x, w):
    return np.sum(w * x) / np.sum(w)


def wfit(x, y, w):
    X = np.c_[np.ones_like(x), x] * np.sqrt(w)[:, None]
    return np.linalg.lstsq(X, y * np.sqrt(w), rcond=None)[0]       # (intercept, slope)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help='"label|family|better/worse|path"')
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--groups", default="1,2,3,4,5,6,7", help="사용할 행동 그룹 (예: 2,3,4,5,6,7 = 그룹 1 제외 검증)")
    ap.add_argument("--tag", default="", help="출력 파일 접미사")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    gl = get_group_indices(include_irrational=False)
    keep_g = {int(g) for g in args.groups.split(",")}
    gid_of = {s: g for g, ss in enumerate(gl, 1) for s in ss if g in keep_g}
    models, paths = [], []
    for spec in args.runs:
        lab, fam, tier, path = spec.split("|")
        models.append((lab, fam, tier)); paths.append(resolve(path))
    rows = row_table(sorted(gid_of))
    btom = ref_arrays(rows, {"btom": load_reference_everystep("btom")})["btom"]
    A = {lab: aggregate(subject_array(load_rows(p), rows)[0]) for (lab, _, _), p in zip(models, paths)}

    sid = rows["scenario_id"].values
    ph = rows["phase_idx"].values
    idx = np.where(np.r_[False, sid[1:] == sid[:-1]])[0]
    s_tr = sid[idx]
    ev_lab = np.array([get_group_phase_labels(gid_of[s]).get(int(ph[t]), "?") for s, t in zip(s_tr, idx)])
    us = np.unique(s_tr)
    pos = np.searchsorted(us, s_tr)

    data = {}
    for dim, ks in SL.items():
        db = np.stack([btom[idx, k] - btom[idx - 1, k] for k in ks], 1).ravel()
        ad = np.abs(db)
        bins = np.digitize(ad, np.quantile(ad, np.linspace(0, 1, NB + 1)[1:-1]))
        ev = np.repeat(ev_lab, 3)
        sc = np.repeat(pos, 3)
        dys = {lab: np.stack([A[lab][idx, k] - A[lab][idx - 1, k] for k in ks], 1).ravel() for lab in A}
        data[dim] = dict(db=db, ad=ad, bins=bins, ev=ev, sc=sc, dys=dys)

    def compute(w_s):
        out = {}
        for dim, D in data.items():
            w = w_s[D["sc"]]
            for b in range(NB):
                m = D["bins"] == b
                out[(dim, "btom", f"bin{b}_movement")] = wmean(D["ad"][m], w[m])
            for e in EVENTS:
                out[(dim, "(n)", f"event_{e}")] = np.sum(w[D["ev"] == e]) / 3
            for lab, dy in D["dys"].items():
                sr = np.sign(D["db"]) * dy
                for b in range(NB):
                    m = D["bins"] == b
                    out[(dim, lab, f"bin{b}_signed")] = wmean(sr[m], w[m])
                    out[(dim, lab, f"bin{b}_movement")] = wmean(np.abs(dy[m]), w[m])
                out[(dim, lab, "sensitivity")] = wfit(D["ad"], sr, w)[1]
                out[(dim, lab, "baseline")] = wfit(D["ad"], np.abs(dy), w)[0]
                for e in EVENTS:
                    m = D["ev"] == e
                    out[(dim, lab, f"event_{e}")] = (np.sum(w[m] * sr[m]) / np.sum(w[m] * D["ad"][m])
                                                     if m.any() else np.nan)
        return out

    est = compute(np.ones(len(us)))
    boots = pd.DataFrame([compute(np.bincount(rng.integers(0, len(us), len(us)), minlength=len(us)).astype(float))
                          for _ in range(args.n_boot)])

    fam = {}
    for lab, f, t in models:
        fam.setdefault(f, {})[t] = lab
    rec = []
    for (dim, lab, met), v in est.items():
        lo, hi = np.nanpercentile(boots[(dim, lab, met)], [2.5, 97.5])
        rec.append(dict(dim=dim, model=lab, metric=met, value=v, ci_low=lo, ci_high=hi))
    # 대비: Better − Worse, relative closure (BToM 기준 = 1)
    for dim in SL:
        for met in ["sensitivity", "baseline"] + [f"event_{e}" for e in EVENTS]:
            for f, d in fam.items():
                b_, w_ = d["better"], d["worse"]
                g = est[(dim, b_, met)] - est[(dim, w_, met)]
                gb = boots[(dim, b_, met)] - boots[(dim, w_, met)]
                lo, hi = np.nanpercentile(gb, [2.5, 97.5])
                rec.append(dict(dim=dim, model=f"gap_{f}", metric=met, value=g, ci_low=lo, ci_high=hi))
                if met != "baseline":
                    R = g / (1 - est[(dim, w_, met)])
                    Rb = gb / (1 - boots[(dim, w_, met)])
                    lo, hi = np.nanpercentile(Rb, [2.5, 97.5])
                    rec.append(dict(dim=dim, model=f"closure_{f}", metric=met, value=R, ci_low=lo, ci_high=hi))
    d = pd.DataFrame(rec)
    os.makedirs(args.out_dir, exist_ok=True)
    d.to_csv(os.path.join(args.out_dir, f"evidence_sensitivity{args.tag}.csv"), index=False)

    pd.set_option("display.width", 250)
    t = d.assign(s=lambda x: x.value.round(3).astype(str) + " [" + x.ci_low.round(3).astype(str) + ", "
                 + x.ci_high.round(3).astype(str) + "]")
    cols = [m[0] for m in models] + [f"gap_{f}" for f in fam] + [f"closure_{f}" for f in fam]
    for dim in SL:
        tt = t[(t.dim == dim) & t.metric.isin(["sensitivity", "baseline"] + [f"event_{e}" for e in EVENTS])]
        print(f"== {dim}")
        print(tt.pivot_table(index="metric", columns="model", values="s", aggfunc="first")
              .reindex(columns=[c for c in cols if c in set(tt.model)] + ["(n)"]).to_string())
    plot_curves(d, models, os.path.join(args.out_dir, f"evidence_sensitivity{args.tag}.png"))


def plot_curves(d, models, out):
    fams = list(dict.fromkeys(f for _, f, _ in models))
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.6))
    for r, dim in enumerate(("desire", "belief")):
        x = np.array([d[(d.dim == dim) & (d.model == "btom") & (d.metric == f"bin{b}_movement")].value.iloc[0]
                      for b in range(NB)])
        xs = np.arange(NB)
        for c, (kind, title) in enumerate((("movement", "How much it moves\n(mean |Δ LLM|)"),
                                          ("signed", "How much it moves with the evidence\n(mean sign(Δ BToM)·Δ LLM)"))):
            ax = axes[r, c]
            ax.plot(xs, x, color="#555555", ls=(0, (4, 2)), lw=1.6, marker="s", ms=5, label="BToM", zorder=2)
            for lab, f, tier in models:
                col = C_BETTER if tier == "better" else C_WORSE
                ls = "-" if f == fams[0] else (0, (1.5, 1.2))
                mk = "o" if f == fams[0] else "D"
                sub = d[(d.dim == dim) & (d.model == lab)]
                y = np.array([sub[sub.metric == f"bin{b}_{kind}"].value.iloc[0] for b in range(NB)])
                lo = np.array([sub[sub.metric == f"bin{b}_{kind}"].ci_low.iloc[0] for b in range(NB)])
                hi = np.array([sub[sub.metric == f"bin{b}_{kind}"].ci_high.iloc[0] for b in range(NB)])
                ax.plot(xs, y, color=col, ls=ls, lw=2, marker=mk, ms=6, mec="white", mew=0.8, zorder=3)
                ax.fill_between(xs, lo, hi, color=col, alpha=0.08, lw=0)
            ax.axhline(0, color="#333333", lw=0.7)
            ax.set_xticks(xs)
            ax.set_xticklabels([f"Q{b + 1}\n{x[b]:.3f}" if dim == "belief" else f"Q{b + 1}\n{x[b]:.2f}"
                                for b in range(NB)], fontsize=8.5)
            if r == 1:
                ax.set_xlabel("Evidence strength: quintile of |Δ BToM| (bin mean)", fontsize=9.5)
            if r == 0:
                ax.set_title(title, fontsize=11.5, fontweight="bold")
            ax.grid(axis="y", color="#EEEEEE", lw=0.7)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
            ax.tick_params(axis="y", labelsize=8.5)
        axes[r, 0].set_ylabel(("Desire (rating units)" if dim == "desire" else "Belief (probability units)"),
                              fontsize=10)
        axes[r, 0].annotate(dim.capitalize(), xy=(0, 0.5), xycoords="axes fraction", xytext=(-58, 0),
                            textcoords="offset points", ha="right", va="center", rotation=90, fontsize=14,
                            fontweight="bold")
    h = [mlines.Line2D([], [], color="#555555", ls=(0, (4, 2)), marker="s", ms=5, label="BToM")]
    for lab, f, tier in models:
        h.append(mlines.Line2D([], [], color=C_BETTER if tier == "better" else C_WORSE,
                               ls="-" if f == fams[0] else (0, (1.5, 1.2)), marker="o" if f == fams[0] else "D",
                               ms=6, lw=2, label=lab))
    fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0.03, 0.06, 1, 1))
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"✅ saved: {out}")


if __name__ == "__main__":
    main()
