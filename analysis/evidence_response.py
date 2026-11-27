"""
증거 스텝 반응 (evidence-step response): BToM이 크게 갱신하는 순간에 LLM이 얼마나 따라 움직이는가.

정의 (시나리오 x 옵션마다)
  증거 스텝 = |ΔBToM|이 가장 큰 스텝 전환 (t−1 -> t)
  evidence_response   = Σ sign(ΔBToM)·ΔLLM / Σ |ΔBToM|   (증거 스텝만; 1 = BToM만큼 같은 방향으로 갱신, 0 = 반응 없음)
  offevidence_movement = Σ |ΔLLM| / Σ |ΔBToM|           (증거 스텝이 아닌 전환; 1보다 크면 BToM보다 많이 움직임)
  update_r, sd_ratio, gain: 모든 전환. 항등식 gain ≈ update_r × sd_ratio (방향 일치 × 움직임 크기)
구간 (증거 스텝이 속한 인지적 구간)
  early = 출발 직후 첫 진행 방향 (Start에서 나가는 전환, Appr G1 / Pass G1)
  late  = 확인 후 / 되돌아올 때 (See G2, Stop, Stop Btw, Return, Appr G2 등)
CI: 시나리오 bootstrap 1000회 (paired 아님, 모델별)

사용 예)
  python analysis/evidence_response.py --out_dir results/stats/prefix_2x2 \\
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
INTERVALS = ["early", "late"]
INTERVAL_NAMES = {"early": "Early: initial heading\n(leaving Start, Appr / Pass G1)",
                  "late": "Late: after checking /\nturning back (See G2, Stop, Return…)"}


def interval_of(lbl_from, lbl_to):
    return "early" if (lbl_from.startswith("Start") or lbl_to in ("Appr G1", "Pass G1")) else "late"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help='"label|family|better/worse|path"')
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    gl = get_group_indices(include_irrational=False)
    gid_of = {s: g for g, ss in enumerate(gl, 1) for s in ss}
    models, paths = [], []
    for spec in args.runs:
        lab, fam, tier, path = spec.split("|")
        models.append((lab, fam, tier)); paths.append(resolve(path))
    rows = row_table(sorted(gid_of))
    btom = ref_arrays(rows, {"btom": load_reference_everystep("btom")})["btom"]
    A = {lab: aggregate(subject_array(load_rows(p), rows)[0]) for (lab, _, _), p in zip(models, paths)}

    sid = rows["scenario_id"].values
    ph = rows["phase_idx"].values
    idx = np.where(np.r_[False, sid[1:] == sid[:-1]])[0]       # 전환 t−1 -> t의 t
    s_tr = sid[idx]
    cat = np.array([interval_of(get_group_phase_labels(gid_of[s_tr[i]]).get(int(ph[t - 1]), "?"),
                                get_group_phase_labels(gid_of[s_tr[i]]).get(int(ph[t]), "?"))
                    for i, t in enumerate(idx)])
    us = np.unique(s_tr)

    rec = []
    for dim, ks in SL.items():
        db = np.stack([btom[idx, k] - btom[idx - 1, k] for k in ks], 1)
        ev = np.zeros_like(db, bool)
        for s in us:
            m = np.where(s_tr == s)[0]
            for j in range(3):
                ev[m[np.argmax(np.abs(db[m, j]))], j] = True
        for lab, fam, tier in models:
            a = A[lab]
            dy = np.stack([a[idx, k] - a[idx - 1, k] for k in ks], 1)

            def stats(sel):
                w = np.zeros(len(us))                       # 시나리오 리샘플 횟수 = 가중치
                np.add.at(w, np.searchsorted(us, sel), 1)
                wt = w[np.searchsorted(us, s_tr)][:, None] * np.ones((1, 3))
                out = {}
                e, o = ev, ~ev
                out["evidence_response"] = np.sum(wt[e] * np.sign(db[e]) * dy[e]) / np.sum(wt[e] * np.abs(db[e]))
                out["offevidence_movement"] = np.sum(wt[o] * np.abs(dy[o])) / np.sum(wt[o] * np.abs(db[o]))
                for c in INTERVALS:
                    mm = e & (cat == c)[:, None]
                    out[f"evidence_response_{c}"] = np.sum(wt[mm] * np.sign(db[mm]) * dy[mm]) / np.sum(wt[mm] * np.abs(db[mm]))
                rep = np.repeat(np.arange(dy.size), wt.ravel().astype(int))
                y, x = dy.ravel()[rep], db.ravel()[rep]
                out["update_r"] = np.corrcoef(y, x)[0, 1]
                out["sd_ratio"] = y.std() / x.std()
                out["gain"] = np.sum(y * x) / np.sum(x ** 2)
                return out

            est = stats(us)
            boots = pd.DataFrame([stats(rng.choice(us, len(us))) for _ in range(args.n_boot)])
            for k, v in est.items():
                lo, hi = np.nanpercentile(boots[k], [2.5, 97.5])
                rec.append(dict(dim=dim, model=lab, family=fam, tier=tier, metric=k, value=v, ci_low=lo, ci_high=hi))
        for c in INTERVALS:
            rec.append(dict(dim=dim, model="(count)", family="", tier="", metric=f"n_evidence_{c}",
                            value=int((ev & (cat == c)[:, None]).sum()), ci_low=np.nan, ci_high=np.nan))
    d = pd.DataFrame(rec)
    os.makedirs(args.out_dir, exist_ok=True)
    d.to_csv(os.path.join(args.out_dir, "evidence_response.csv"), index=False)
    t = d[d.model != "(count)"].assign(s=lambda x: x.value.round(2).astype(str) + " [" + x.ci_low.round(2).astype(str)
                                       + ", " + x.ci_high.round(2).astype(str) + "]")
    pd.set_option("display.width", 250)
    print(t.pivot_table(index=["dim", "metric"], columns="model", values="s", aggfunc="first")[[m[0] for m in models]].to_string())
    print(d[d.model == "(count)"][["dim", "metric", "value"]].to_string(index=False))
    plot(d, models, os.path.join(args.out_dir, "evidence_response.png"))


def plot(d, models, out):
    fams = list(dict.fromkeys(f for _, f, _ in models))
    mk = {fams[0]: "o", fams[1]: "D"} if len(fams) > 1 else {fams[0]: "o"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    offs = {lab: o for (lab, _, _), o in zip(models, (-0.21, -0.07, 0.07, 0.21))}
    for ax, dim in zip(axes, ("desire", "belief")):
        ax.axhline(1, color="#555555", ls=(0, (4, 2)), lw=1.2)
        ax.axhline(0, color="#333333", lw=0.8)
        ax.text(1.42, 1.0, "BToM", fontsize=9, va="bottom", ha="right", color="#555555")
        for i, c in enumerate(INTERVALS):
            for lab, fam, tier in models:
                r = d[(d.dim == dim) & (d.model == lab) & (d.metric == f"evidence_response_{c}")].iloc[0]
                col = C_BETTER if tier == "better" else C_WORSE
                x = i + offs[lab]
                ax.errorbar(x, r.value, yerr=[[r.value - r.ci_low], [r.ci_high - r.value]], fmt=mk[fam], color=col,
                            ms=8.5, capsize=3, lw=1.7, mec="white", mew=1, zorder=3)
                ax.text(x, r.ci_high + 0.04, f"{r.value:.2f}", fontsize=7.8, ha="center", va="bottom", color="#333333")
            n = int(d[(d.dim == dim) & (d.metric == f"n_evidence_{c}")].value.iloc[0])
            ax.text(i, -0.42, f"n = {n} evidence steps", ha="center", fontsize=8, color="#666666")
        ax.set_xticks(range(len(INTERVALS)))
        ax.set_xticklabels([INTERVAL_NAMES[c] for c in INTERVALS], fontsize=9.5)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.75)
        ax.set_title(dim.capitalize(), fontsize=13, fontweight="bold")
        ax.grid(axis="y", color="#EEEEEE", lw=0.7, zorder=0)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    axes[0].set_ylabel("Evidence-step response\n(share of BToM's update, 95% CI)", fontsize=10)
    h = [mlines.Line2D([], [], color=(C_BETTER if t == "better" else C_WORSE), marker=mk[f], ls="None", ms=8, label=lab)
         for lab, f, t in models]
    fig.legend(handles=h, loc="lower center", ncol=len(models), frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"✅ saved: {out}")


if __name__ == "__main__":
    main()
