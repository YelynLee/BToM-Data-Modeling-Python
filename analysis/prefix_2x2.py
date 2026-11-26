"""
Prefix-step 2x2 비교 (계열 x 성능): Claude Opus / Sonnet, Gemini 2.5 Pro / Flash.

Hindsight 모델 없이 (교수님 결정) BToM과 대안 레퍼런스 3개(TrueBelief, NoCost, MotionHeuristic)만 사용.
모든 모델에 같은 시나리오 리샘플을 적용 (paired scenario bootstrap, 층화 없음) -> 모델 간 차이와 2x2 대비의 CI.

지표 (모델별; belief = 실행 평균 후 확률 정규화, desire = 원점수)
  level_r      : 단계별 판단(시나리오 x 스텝 x 옵션 pooled)과 BToM의 Pearson r
  rm_r         : 시나리오 내 평균을 뺀 뒤의 r (궤적 안에서의 변화가 BToM과 같은 모양인지), 옵션 평균
  update_r     : 연속 스텝 변화 Δ_t의 r (언제, 어느 방향으로 판단을 바꾸는지)
  vpa_unique_* : BToM / TrueBelief / NoCost / MotionHeuristic 고유 adjusted R² (0으로 자르지 않음), vpa_total
  mae          : |LLM − BToM| 평균 (시나리오 평균 후 평균)
  final_human_r: 마지막 스텝(전체 궤적)의 판단과 사람 End-step 판단의 r (pooled 3 옵션)
  final_btom_r : 마지막 스텝과 BToM End-step의 r
대비 (better − worse)
  gap_Claude = Opus − Sonnet,  gap_Gemini = Pro − Flash,  gap_mean = 두 gap 평균 (성능 주효과)
  family     = (Opus + Sonnet)/2 − (Pro + Flash)/2,  interaction = gap_Claude − gap_Gemini

사용 예)
  python analysis/prefix_2x2.py --out_dir results/stats/prefix_2x2 \\
     --runs "Opus|Claude|better|results/claude-opus-4-6/vanilla/prefixstep/thinking_off" \\
            "Sonnet|Claude|worse|results/claude-sonnet-4-6/vanilla/prefixstep/thinking_off" \\
            "Gemini-Pro|Gemini|better|results/gemini-2.5-pro/vanilla/prefixstep" \\
            "Gemini-Flash|Gemini|worse|results/gemini-2.5-flash/vanilla/prefixstep"
"""
import os
import sys
import argparse
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "src"), os.path.join(ROOT, "analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)
warnings.filterwarnings("ignore")

from analysis.compare_stepwise_bootstrap import (load_rows, row_table, ref_arrays, subject_array, aggregate,
                                                 _adj_r2, COLS)
from src.prepare_everystep import load_reference_everystep
from src.config import get_group_indices, HUMAN_PKL_PATH

REFS = ["btom", "truebelief", "nocost", "motionheuristic"]
REF_NAMES = {"btom": "BToM", "truebelief": "TrueBelief", "nocost": "NoCost", "motionheuristic": "MotionHeuristic"}
SL = {"desire": slice(0, 3), "belief": slice(3, 6)}
C_BETTER, C_WORSE = "#60A917", "#D80073"


def resolve(path):
    """subject_*.csv가 하위 폴더 하나(thinking_off 등)에만 있으면 그 폴더."""
    import glob
    if os.path.isdir(path) and not glob.glob(os.path.join(path, "subject_*.csv")):
        subs = [d for d in sorted(os.listdir(path)) if glob.glob(os.path.join(path, d, "subject_*.csv"))]
        if len(subs) == 1:
            return os.path.join(path, subs[0])
    return path


class Data:
    def __init__(self, rows, refs, human):
        self.rows = rows
        self.refs = refs
        sc = rows["scenario_id"].to_numpy()
        self.scen_ids = sorted(np.unique(sc))
        pos = {s: i for i, s in enumerate(self.scen_ids)}
        self.sidx = np.array([pos[s] for s in sc])
        self.n_s = len(self.scen_ids)
        ts = rows["time_step"].to_numpy()
        # 이전 스텝 행 (같은 시나리오), 마지막 스텝 행
        self.prev = np.full(len(rows), -1)
        same = np.r_[False, sc[1:] == sc[:-1]]
        self.prev[same] = np.where(same)[0] - 1
        self.last = np.r_[sc[1:] != sc[:-1], True]
        # 사람 End-step (시나리오 순서대로, 행 = 시나리오, 열 = COLS)
        h = np.column_stack([human["des_inf_mean"].T, human["bel_inf_mean_norm"].T])   # (78, 6)
        self.human = h[np.array(self.scen_ids) - 1]
        self.btom_last = refs["btom"][self.last]

    def _center(self, x, w):
        cnt = np.bincount(self.sidx, minlength=self.n_s)
        mean = np.bincount(self.sidx, weights=x, minlength=self.n_s) / np.maximum(cnt, 1)
        return x - mean[self.sidx]

    @staticmethod
    def _wr(x, y, w):
        ok = ~np.isnan(x) & ~np.isnan(y) & (w > 0)
        x, y, w = x[ok], y[ok], w[ok]
        mx, my = np.average(x, weights=w), np.average(y, weights=w)
        cov = np.sum(w * (x - mx) * (y - my))
        return cov / np.sqrt(np.sum(w * (x - mx) ** 2) * np.sum(w * (y - my) ** 2))

    def metrics(self, agg, sw):
        rw = sw[self.sidx].astype(float)
        out = {}
        btom = self.refs["btom"]
        has_prev = self.prev >= 0
        for score, sl in SL.items():
            y, b = agg[:, sl], btom[:, sl]
            W = np.repeat(rw[:, None], 3, axis=1)
            out[f"{score}_level_r"] = self._wr(y.ravel(), b.ravel(), W.ravel())
            rms = [self._wr(self._center(y[:, j], rw), self._center(b[:, j], rw), rw) for j in range(3)]
            out[f"{score}_rm_r"] = float(np.mean(rms))
            dy = y[has_prev] - y[self.prev[has_prev]]
            db = b[has_prev] - b[self.prev[has_prev]]
            out[f"{score}_update_r"] = self._wr(dy.ravel(), db.ravel(), W[has_prev].ravel())
            err = np.abs(y - b).mean(axis=1)
            e_sc = np.bincount(self.sidx, weights=err, minlength=self.n_s) / np.bincount(self.sidx, minlength=self.n_s)
            out[f"{score}_mae"] = np.average(e_sc, weights=sw) if sw.sum() > 0 else np.nan
            # VPA (행 복제 = 가중치)
            rep = np.repeat(np.arange(len(agg)), rw.astype(int))
            yy = agg[rep][:, sl].ravel(order="F")
            X = np.column_stack([self.refs[r][rep][:, sl].ravel(order="F") for r in REFS])
            ok = ~np.isnan(yy) & ~np.isnan(X).any(axis=1)
            yy, X = yy[ok], X[ok]
            keep = X.std(axis=0) > 1e-9
            names = [r for r, k in zip(REFS, keep) if k]
            yz = (yy - yy.mean()) / yy.std()
            Xz = (X[:, keep] - X[:, keep].mean(axis=0)) / X[:, keep].std(axis=0)
            full = _adj_r2(yz, Xz)
            out[f"{score}_vpa_total"] = full
            for i, r in enumerate(names):
                out[f"{score}_vpa_unique_{r}"] = full - _adj_r2(yz, np.delete(Xz, i, axis=1))
            # 마지막 스텝 vs 사람 / BToM End-step
            fl = agg[self.last][:, sl]
            Ws = np.repeat(sw[:, None].astype(float), 3, axis=1).ravel()
            out[f"{score}_final_human_r"] = self._wr(fl.ravel(), self.human[:, sl].ravel(), Ws)
            out[f"{score}_final_btom_r"] = self._wr(fl.ravel(), self.btom_last[:, sl].ravel(), Ws)
        return out


def contrasts(m, models):
    """m: {label: metrics dict}. models: [(label, family, tier)]"""
    fam = {}
    for lab, f, t in models:
        fam.setdefault(f, {})[t] = lab
    out = {}
    keys = m[models[0][0]].keys()
    fams = list(fam)
    for k in keys:
        gaps = {}
        for f in fams:
            gaps[f] = m[fam[f]["better"]][k] - m[fam[f]["worse"]][k]
            out[(k, f"gap_{f}")] = gaps[f]
        out[(k, "gap_mean")] = np.mean(list(gaps.values()))
        if len(fams) == 2:
            a, b = fams
            out[(k, f"family_{a}-{b}")] = (np.mean([m[fam[a][t]][k] for t in ("better", "worse")])
                                           - np.mean([m[fam[b][t]][k] for t in ("better", "worse")]))
            out[(k, "interaction")] = gaps[a] - gaps[b]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help='"label|family|better/worse|path"')
    ap.add_argument("--groups", default="1,2,3,4,5,6,7")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    models, paths = [], []
    for spec in args.runs:
        lab, fam, tier, path = spec.split("|")
        models.append((lab, fam, tier))
        paths.append(resolve(path))
    groups = [int(g) for g in args.groups.split(",")]
    gl = get_group_indices(include_irrational=False)          # 그룹 1..7, 1-based 시나리오 id
    scen = sorted(s for gi, g in enumerate(gl, start=1) if gi in groups for s in g)

    dfs = {lab: load_rows(p) for (lab, _, _), p in zip(models, paths)}
    common = set(scen)
    for lab, d in dfs.items():
        common &= set(d["scenario_id"].unique())
    scen = sorted(common)
    rows = row_table(scen)
    refs = ref_arrays(rows, {r: load_reference_everystep(r) for r in REFS})
    human = pickle.load(open(os.path.join(ROOT, HUMAN_PKL_PATH), "rb"))
    data = Data(rows, refs, human)
    aggs, nsub = {}, {}
    for lab, d in dfs.items():
        arr, subs = subject_array(d, rows)
        aggs[lab] = aggregate(arr)
        nsub[lab] = len(subs)
    print(f"scenarios: {data.n_s}, rows: {len(rows)}, runs: {nsub}")

    ones = np.ones(data.n_s)
    point = {lab: data.metrics(aggs[lab], ones) for lab in aggs}
    pc = contrasts(point, models)
    rng = np.random.default_rng(args.seed)
    boots, bc = {lab: [] for lab in aggs}, []
    for b in range(args.n_boot):
        sw = np.bincount(rng.integers(0, data.n_s, data.n_s), minlength=data.n_s).astype(float)
        mb = {lab: data.metrics(aggs[lab], sw) for lab in aggs}
        for lab in aggs:
            boots[lab].append(mb[lab])
        bc.append(contrasts(mb, models))

    rec = []
    for lab, fam, tier in models:
        bdf = pd.DataFrame(boots[lab])
        for k, v in point[lab].items():
            lo, hi = np.nanpercentile(bdf[k], [2.5, 97.5])
            rec.append(dict(model=lab, family=fam, tier=tier, n_runs=nsub[lab], metric=k, value=v, ci_low=lo, ci_high=hi))
    per_model = pd.DataFrame(rec)
    per_model.to_csv(os.path.join(args.out_dir, "prefix_2x2_metrics.csv"), index=False)

    crec = []
    bcdf = pd.DataFrame(bc)
    for (k, c), v in pc.items():
        col = bcdf[(k, c)]
        lo, hi = np.nanpercentile(col, [2.5, 97.5])
        p = 2 * min((col <= 0).mean(), (col >= 0).mean())
        crec.append(dict(metric=k, contrast=c, value=v, ci_low=lo, ci_high=hi, p_boot=min(p, 1.0)))
    con = pd.DataFrame(crec)
    con.to_csv(os.path.join(args.out_dir, "prefix_2x2_contrasts.csv"), index=False)

    # 요약 출력
    def fmt(r):
        return f"{r.value:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}]"
    show = [k for k in point[models[0][0]]]
    tab = per_model.assign(s=per_model.apply(fmt, axis=1)).pivot(index="metric", columns="model", values="s")
    tab = tab.loc[show, [m[0] for m in models]]
    print(tab.to_string())
    ctab = con.assign(s=con.apply(lambda r: fmt(r) + ("*" if r.p_boot < 0.05 else ""), axis=1)).pivot(
        index="metric", columns="contrast", values="s").loc[show]
    print(ctab.to_string())
    tab.to_csv(os.path.join(args.out_dir, "prefix_2x2_metrics_table.csv"))
    ctab.to_csv(os.path.join(args.out_dir, "prefix_2x2_contrasts_table.csv"))

    plot_summary(per_model, models, os.path.join(args.out_dir, "prefix_2x2_summary.png"))
    plot_vpa(per_model, models, os.path.join(args.out_dir, "prefix_2x2_vpa.png"))


def _dot_panel(ax, per_model, models, metric, title, ylim=None):
    fams = list(dict.fromkeys(f for _, f, _ in models))
    for i, f in enumerate(fams):
        for tier, dx in (("better", -0.12), ("worse", 0.12)):
            lab = [m for m, ff, t in models if ff == f and t == tier][0]
            r = per_model[(per_model.model == lab) & (per_model.metric == metric)].iloc[0]
            c = C_BETTER if tier == "better" else C_WORSE
            ax.errorbar(i + dx, r.value, yerr=[[r.value - r.ci_low], [r.ci_high - r.value]], fmt="o", color=c,
                        ms=7, capsize=3, lw=1.5, mec="white", mew=0.8)
            ax.text(i + dx + 0.06, r.value, lab, fontsize=7.5, va="center", color="#444444")
    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels(fams)
    ax.set_xlim(-0.5, len(fams) - 0.3)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", color="#EEEEEE", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_summary(per_model, models, out):
    panels = [("final_human_r", "Final step vs Human"), ("level_r", "Step-wise vs BToM (level)"),
              ("rm_r", "Within-trajectory vs BToM"), ("update_r", "Step-to-step update vs BToM")]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharey="row")
    for r, score in enumerate(("belief", "desire")):
        for c, (k, t) in enumerate(panels):
            _dot_panel(axes[r, c], per_model, models, f"{score}_{k}", f"{score.capitalize()}: {t}", (0, 1))
        axes[r, 0].set_ylabel("Pearson r (95% CI)")
    import matplotlib.lines as mlines
    h = [mlines.Line2D([], [], color=C_BETTER, marker="o", ls="None", label="Better"),
         mlines.Line2D([], [], color=C_WORSE, marker="o", ls="None", label="Worse")]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Prefix-step 2x2: family x capability (scenario bootstrap, 73 scenarios)", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"✅ saved: {out}")


def plot_vpa(per_model, models, out):
    colors = {"btom": "#E0A93B", "truebelief": "#6AA9C9", "nocost": "#8FBF7F", "motionheuristic": "#B8969A"}
    labs = [m[0] for m in models]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    w = 0.2
    for ax, score in zip(axes, ("belief", "desire")):
        for j, r in enumerate(REFS):
            for i, lab in enumerate(labs):
                row = per_model[(per_model.model == lab) & (per_model.metric == f"{score}_vpa_unique_{r}")]
                if row.empty:
                    continue
                row = row.iloc[0]
                x = j + (i - 1.5) * w
                tier = models[i][2]
                ax.bar(x, row.value, w * 0.9, color=colors[r], edgecolor=C_BETTER if tier == "better" else C_WORSE,
                       lw=1.6, hatch=None if models[i][1] == models[0][1] else "///")
                ax.errorbar(x, row.value, yerr=[[row.value - row.ci_low], [row.ci_high - row.value]], color="#333333",
                            lw=1, capsize=2)
        ax.set_xticks(range(len(REFS)))
        ax.set_xticklabels([REF_NAMES[r] for r in REFS])
        ax.axhline(0, color="#333333", lw=0.8)
        ax.set_title(f"{score.capitalize()}: unique adjusted R²", fontsize=10)
        ax.grid(axis="y", color="#EEEEEE", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Unique adj. R² (95% CI)")
    import matplotlib.patches as mpatches
    f0 = models[0][1]
    f1 = [f for _, f, _ in models if f != f0][0]
    h = [mpatches.Patch(facecolor="#CCCCCC", edgecolor="#555555", label=f"{f0} (solid)"),
         mpatches.Patch(facecolor="#CCCCCC", edgecolor="#555555", hatch="///", label=f"{f1} (hatched)")]
    h += [mpatches.Patch(facecolor="white", edgecolor=C_BETTER, lw=1.6, label="Better (outline)"),
          mpatches.Patch(facecolor="white", edgecolor=C_WORSE, lw=1.6, label="Worse (outline)")]
    fig.legend(handles=h, loc="lower center", ncol=6, frameon=False, fontsize=8.5)
    fig.suptitle("Prefix-step VPA (bars in order: " + ", ".join(labs) + ")", fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"✅ saved: {out}")


if __name__ == "__main__":
    main()
