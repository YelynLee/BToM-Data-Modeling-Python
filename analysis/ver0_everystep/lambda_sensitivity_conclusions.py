"""
λ 민감도: Hindsight 레퍼런스의 λ를 바꿔도 Every-step 결론이 유지되는지.
  - hindsight index (Hindsight_λ 방향/단위)의 모델 순위
  - belief VPA (5개 레퍼런스, Hindsight만 Hindsight_λ로 교체): Hindsight 고유 adj. R², BToM β
λ = 2 (β = 3.5, 논문 레퍼런스 파일)도 함께 계산해 재현 확인.

  python analysis/lambda_sensitivity_conclusions.py --posterior <posterior.mat> --runs "이름|csv" ... --out_dir ...
"""
import os, sys, io, argparse, contextlib
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for q in (ROOT, os.path.join(ROOT, "analysis"), os.path.join(ROOT, "src")):
    if q not in sys.path:
        sys.path.insert(0, q)

LAMS = [0.5, 1, 2, 4, 8, 32, np.inf]
BELC = ["belief_L", "belief_M", "belief_Empty"]
REFS = ["btom", "truebelief", "nocost", "motionheuristic", "hindsight"]


def llm_mean(path, keep):
    from analysis.phase_utils import get_phase_index
    df = pd.read_csv(path)
    m = df.groupby(["scenario_id", "group_id", "phase", "time_step"])[BELC].mean().reset_index()
    m["phase_idx"] = [get_phase_index(g, p) for g, p in zip(m.group_id, m.phase)]
    m = m[(m.phase_idx != 9) & m.scenario_id.isin(keep)]
    b = np.maximum(m[BELC] - 1, 0)
    m[BELC] = b.div(b.sum(axis=1).replace(0, 1.0), axis=0)
    return m


def vpa_arrays(m, ref_dfs):
    y = m.melt(id_vars=["scenario_id", "time_step"], value_vars=BELC, var_name="opt", value_name="LLM")
    for r in REFS:
        y = y.merge(ref_dfs[r].melt(id_vars=["scenario_id", "time_step"], value_vars=BELC, var_name="opt", value_name=r),
                    on=["scenario_id", "time_step", "opt"], how="left")
    y = y.dropna()
    return y.LLM.to_numpy(float), y[REFS].to_numpy(float), y.scenario_id.to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posterior", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_boot", type=int, default=500)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    import hindsight_lambda as HL
    from vpa_plot import vpa_with_ci
    with contextlib.redirect_stdout(io.StringIO()):
        al = HL.Aligner()
        C = al.C
        from src.prepare_everystep import load_reference_everystep
        base_refs = {r: load_reference_everystep(r) for r in REFS}
    post, beta, lam0 = HL.load_posterior(a.posterior)
    keep = set(al.scen)
    runs = [s.split("|", 1) for s in a.runs]
    means = {n: llm_mean(p, keep) for n, p in runs}
    ratings = {n: C.aggregate(C.subject_array(C.load_rows(p), al.rows)[0]) for n, p in runs}

    def hind_df(lam):
        if lam == "paper":
            return base_refs["hindsight"]
        marg = HL.belief_marg(post, lam0, lam)
        r = al.base.copy()
        r[BELC] = np.array([marg[s - 1][:, al.col(t, al.Tmax[s], marg[s - 1].shape[1])]
                            for s, t in zip(r.scenario_id, r.time_step)])
        return r

    out = []
    for lam in ["paper"] + LAMS:
        H = hind_df(lam)
        refs_d = dict(base_refs, hindsight=H)
        with contextlib.redirect_stdout(io.StringIO()):
            refs_a = C.ref_arrays(al.rows, refs_d)
            eng = C.Engine(al.rows, refs_a)
        ok = np.ones(len(al.rows), bool)
        cb = float(np.nanmean(eng.hindsight_by_scenario(refs_a["truebelief"], ok)))
        lab = "2 (paper, β=3.5)" if lam == "paper" else ("∞" if np.isinf(lam) else f"{lam:g}")
        for n, _ in runs:
            hi = float(np.nanmean(eng.hindsight_by_scenario(ratings[n], ok)))
            y, X, sc = vpa_arrays(means[n], refs_d)
            v = vpa_with_ci(y, X, REFS, sc, n_boot=a.n_boot, seed=0)
            out.append(dict(lam=lab, model=n, hindsight_index=hi, current_belief_index=cb, n_scen_index=int(np.isfinite(eng.hindsight_by_scenario(refs_a["truebelief"], ok)).sum()),
                            uniq_hind=v["unique"]["hindsight"][0], uniq_hind_lo=v["unique"]["hindsight"][1], uniq_hind_hi=v["unique"]["hindsight"][2],
                            beta_btom=v["beta"]["btom"][0], beta_btom_lo=v["beta"]["btom"][1], beta_btom_hi=v["beta"]["btom"][2],
                            beta_hind=v["beta"]["hindsight"][0], uniq_btom=v["unique"]["btom"][0], total=v["total"][0]))
        print("λ =", lab, "done")
    D = pd.DataFrame(out)
    D.to_csv(os.path.join(a.out_dir, "lambda_sensitivity_conclusions.csv"), index=False)
    pd.set_option("display.width", 250)
    for col in ["hindsight_index", "uniq_hind", "beta_btom", "beta_hind"]:
        print(col); print(D.pivot(index="model", columns="lam", values=col)[["2 (paper, β=3.5)"] + [l for l in D.lam.unique() if not l.startswith("2 (")]].round(2).to_string())
    print(D.drop_duplicates("lam")[["lam", "current_belief_index", "n_scen_index"]].to_string(index=False))


if __name__ == "__main__":
    main()
