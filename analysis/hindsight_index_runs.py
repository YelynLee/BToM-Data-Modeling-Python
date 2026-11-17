"""
실행(run) 단위 hindsight index 분포: 혼합(run이 BToM 또는 현재 belief 중 하나) vs 점진(모든 run이 중간값).

각 run x 시나리오에서 Spot 2가 보인 뒤 스텝으로 HI를 구하고, 같은 시나리오의 현재 belief HI로 나눈
share(0 = BToM, 1 = 현재 belief)를 계산. 모델별 run 평균 HI와 run x 시나리오 share 분포를 출력/그림.
  python analysis/hindsight_index_runs.py --runs "Opus|.../everystep_valid_only.csv" ... --out_dir ...
"""
import os, sys, io, argparse, contextlib
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for q in (ROOT, os.path.join(ROOT, "analysis"), os.path.join(ROOT, "src")):
    if q not in sys.path:
        sys.path.insert(0, q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        import compare_stepwise_bootstrap as C
        allowed = C.get_group_indices(include_irrational=False)
        scen = sorted(s for g in range(1, 8) for s in allowed[g - 1])
        rows = C.row_table(scen)
        refs = C.ref_arrays(rows, {r: C.load_reference_everystep(r) for r in C.ALL_REFS})
        eng = C.Engine(rows, refs)
    ok = np.ones(len(rows), bool)
    cb = eng.hindsight_by_scenario(refs["truebelief"], ok)          # 시나리오별 현재 belief HI
    recs = []
    for spec in a.runs:
        name, path = spec.split("|", 1)
        arr, subs = C.subject_array(C.load_rows(path), rows)
        pooled = np.nanmean(eng.hindsight_by_scenario(C.aggregate(arr), ok))
        for i, s in enumerate(subs):
            h = eng.hindsight_by_scenario(C.aggregate(arr[i:i + 1]), ok)
            for k, sid in enumerate(eng.scen_ids):
                if np.isfinite(h[k]) and np.isfinite(cb[k]):
                    recs.append(dict(model=name, run=s, scenario=sid, hi=h[k], share=h[k] / cb[k], pooled=pooled))
    D = pd.DataFrame(recs)
    D.to_csv(os.path.join(a.out_dir, "hi_run_scenario.csv"), index=False)
    out = []
    for name, g in D.groupby("model", sort=False):
        run_mean = g.groupby("run").hi.mean()
        sh = g.share.clip(-0.5, 1.5)
        # 분산 분해: run, 시나리오, 잔차 (2원 가법, 균형 근사)
        gm = g.share.mean()
        rv = g.groupby("run").share.mean().var()
        sv = g.groupby("scenario").share.mean().var()
        tv = g.share.var()
        out.append(dict(model=name, pooled_HI=g.pooled.iloc[0], n_runs=g.run.nunique(),
                        run_HI_min=run_mean.min(), run_HI_median=run_mean.median(), run_HI_max=run_mean.max(),
                        share_lt_0_25=np.mean(sh < 0.25), share_0_25_to_0_75=np.mean((sh >= 0.25) & (sh <= 0.75)),
                        share_gt_0_75=np.mean(sh > 0.75), var_runs=rv / tv, var_scen=sv / tv))
    S = pd.DataFrame(out).sort_values("pooled_HI", ascending=False)
    S.to_csv(os.path.join(a.out_dir, "hi_run_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print(S.round(3).to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = list(S.model)
    fig, axs = plt.subplots(2, len(order), figsize=(1.55 * len(order), 3.6), sharey="row")
    for j, name in enumerate(order):
        g = D[D.model == name]
        ax = axs[0, j]
        ax.hist(g.share.clip(-0.5, 1.5), bins=np.linspace(-0.5, 1.5, 21), color="#5f6b78")
        ax.axvline(0, color="#1f4e9c", lw=0.8); ax.axvline(1, color="#555", lw=0.8, ls=":")
        ax.set_title(name.replace("-", "-\n", 1), fontsize=6.5)
        ax.tick_params(labelsize=6)
        rm = g.groupby("run").share.mean()
        bx = axs[1, j]
        bx.scatter(np.random.default_rng(0).uniform(-0.2, 0.2, len(rm)), rm, s=10, color="#5f6b78")
        bx.axhline(0, color="#1f4e9c", lw=0.8); bx.axhline(1, color="#555", lw=0.8, ls=":")
        bx.set_xlim(-0.6, 0.6); bx.set_xticks([]); bx.tick_params(labelsize=6); bx.set_ylim(-0.3, 1.3)
        for z in (ax, bx):
            for sp in ("top", "right"):
                z.spines[sp].set_visible(False)
    axs[0, 0].set_ylabel("run × scenario\ncount", fontsize=7)
    axs[1, 0].set_ylabel("run mean share", fontsize=7)
    fig.text(0.5, 0.455, "share = HI / current-belief HI  (0 = BToM, 1 = current belief)", ha="center", fontsize=7)
    fig.tight_layout(h_pad=3.2)
    fig.savefig(os.path.join(a.out_dir, "hi_runs.png"), dpi=250, bbox_inches="tight")


if __name__ == "__main__":
    main()
