"""
End-step 통계 (리뷰어 대응용). 모든 분석은 같은 시나리오 집합(기본: irrational 제외 73개)에서 계산.

[1] 사람 데이터 신뢰도: split-half (Spearman-Brown), leave-one-subject-out
[2] 사람 vs 레퍼런스 모델(BToM 등): 같은 파이프라인의 pooled r (논문의 Baker CV 값 대신)
[3] LLM별 r(LLM, 사람), r(LLM, BToM)
[4] LLM vs BToM 기준선: Δr = r(LLM, 사람) - r(BToM, 사람)
      - paired bootstrap CI / p (주 검정)
      - Williams t (보조; 옵션 3개를 펼친 점들이 완전히 독립은 아님)
      - LLM 10개에 대해 Holm 보정
[5] Better vs Worse 그룹 평균 r 차이 (paired bootstrap)
      ※ 그룹을 같은 데이터(사람과의 상관)로 정의했으므로 차이가 큰 것은 정의상 당연함 -> 기술 통계로만 사용
[6] End-step vs Every-step 최종 스텝 (Every-step pkl이 있는 모델)
      - 모델별 r(End, Every)
      - Better 모델 합산 두 방식: pooled(모델을 쌓아서) / model-mean(모델 평균을 먼저)
      - 각각 r(End, 사람), r(Every, 사람)과 그 차이
      - mean_of_model_r: 모델별 r을 먼저 구한 뒤 평균 (모델마다 따로 본 정합도의 평균)
[7] Worse 그룹 레퍼런스 모델 비교 (논문 §4.2, Fig 4a, Fig 7)
      - 그룹: Worse 모델 평균 평정 vs 레퍼런스 5개 r (논문 Fig 4a와 같은 방식) + 가장 높은 모델과 나머지의 paired 차이
      - 모델별: 각 Worse 모델 vs 레퍼런스 r + 1위와 2위 레퍼런스의 paired 차이 (Fig 7의 "best" 판정이 유의한지)
[8] End-step VPA (논문 §4.3, App F, Fig 4b, Fig 8): 모델별 고유 adjusted R², 표준화 β, total R²
      - p4_*: 예측변수 4개(BToM, TrueBelief, NoCost, MotionHeuristic) = 논문 End-step VPA(endstep_vpa_analysis.py)와 같음
              (Opus 고유 BToM desire 0.174 / belief 0.135 재현). ※ 논문 §3.3 본문은 "five models"라고 되어 있어 수정 필요
      - p5_*: Hindsight까지 5개 (belief만; desire는 Hindsight와 BToM의 상관이 ~1이라 BToM 고유분이 정의상 0에 가까워짐)
      - 모두 z-score
[9] desire와 belief의 모델 순위 일치 (논문 App E.1의 Spearman 0.65)
      - 모델 10개의 r(LLM, 사람) desire 순위 vs belief 순위의 Spearman ρ, 시나리오 bootstrap CI
      - ※ CI는 시나리오 표본에 따른 불확실성만 반영 (모델이 10개뿐이라는 점은 반영하지 않음)
      - 논문은 음수를 0으로 잘랐으나(clip) 여기서는 CI를 위해 자르지 않음. 점추정은 'estimate_clipped'에 함께 기록

CI: 시나리오 단위 bootstrap (기본: 층화 없음 = 논문 Table 4 방식, --stratified면 행동 그룹별 층화; 옵션 3개는 함께), 모든 지표에 같은 리샘플 사용.

사용 예)
  python analysis/endstep_stats.py --condition vanilla --out_dir results/stats
  python analysis/endstep_stats.py --exclude_partial      # Check-Partial 제외 (49개)
"""
import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "src"), os.path.join(ROOT, "analysis")):
    if p not in sys.path:
        sys.path.append(p)

from src.config import BASE_RESULTS_DIR, REFERENCE_PKL_DIR, HUMAN_PKL_PATH, get_group_indices
from stats_utils import (pooled_r, williams_t, boot_indices, set_stratified, percentile_ci, boot_p_two_sided,
                         holm, split_half_reliability, loso_r)

BETTER = ['gemini-2.5-pro', 'deepseek-reasoner', 'deepseek-v4-pro', 'claude-opus-4-6']
WORSE = ['gpt-4o', 'gpt-5.4', 'o4-mini', 'gemini-2.5-flash', 'deepseek-chat', 'claude-sonnet-4-6']
REFS = ['btom', 'truebelief', 'nocost', 'motionheuristic', 'hindsight']
KEY = {'desire': 'des_inf_mean', 'belief': 'bel_inf_mean_norm'}
RAW = {'desire': 'des_inf', 'belief': 'bel_inf'}


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def scenario_set(exclude_partial=False):
    groups = get_group_indices(include_irrational=False)
    use = groups[:5] if exclude_partial else groups
    ids = sorted(s for g in use for s in g)
    group_of = {s: gi for gi, g in enumerate(groups) for s in g}
    return ids, group_of


def vpa(y, X, names):
    """z-score 회귀. Returns: dict(total, unique_<r>, beta_<r>) (분산 0인 예측변수는 제외)"""
    ok = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
    y, X = y[ok], X[ok]
    keep = X.std(axis=0) > 1e-12
    X, names = X[:, keep], [n for n, k in zip(names, keep) if k]
    z = lambda a: (a - a.mean(axis=0)) / a.std(axis=0)
    yz, Xz = z(y), z(X)

    def adj(Xs):
        A = np.column_stack([np.ones(len(yz)), Xs])
        beta, *_ = np.linalg.lstsq(A, yz, rcond=None)
        resid = yz - A @ beta
        r2 = 1 - resid @ resid / (yz @ yz)
        n, k = Xs.shape
        return 1 - (1 - r2) * (n - 1) / (n - k - 1), beta[1:]

    full, beta = adj(Xz)
    out = {"total": full}
    for i, nm in enumerate(names):
        out[f"unique_{nm}"] = full - adj(np.delete(Xz, i, axis=1))[0]
        out[f"beta_{nm}"] = beta[i]
    return out


def mat(d, dim, cols):
    return np.asarray(d[KEY[dim]], float)[:, cols]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="vanilla")
    ap.add_argument("--exclude_partial", action="store_true")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=os.path.join(BASE_RESULTS_DIR, "stats"))
    ap.add_argument("--stratified", action="store_true",
                    help="행동 그룹별 층화 bootstrap (기본: 층화 없음 = 논문 Table 4 방식). 결과 파일명에 _stratified")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ids, group_of = scenario_set(args.exclude_partial)
    cols = np.array(ids) - 1
    n_points = 3 * len(ids)
    set_stratified(args.stratified)
    B = boot_indices(ids, group_of, args.n_boot, args.seed)  # (n_boot, n_scen) 위치
    print(f"[info] scenarios = {len(ids)} ({'Check-Partial 제외' if args.exclude_partial else 'irrational만 제외'}), "
          f"points per dimension = {n_points}, bootstrap = {args.n_boot}")

    human = load(HUMAN_PKL_PATH)
    refs = {r: load(os.path.join(REFERENCE_PKL_DIR, r, f"{r}_data.pkl")) for r in REFS}
    llm = {}
    for m in BETTER + WORSE:
        p = os.path.join(BASE_RESULTS_DIR, m, args.condition, "model_data.pkl")
        if os.path.exists(p):
            llm[m] = load(p)
        else:
            print(f"[warn] missing End-step: {m}")
    every = {}
    for m in BETTER + WORSE:
        p = os.path.join(BASE_RESULTS_DIR, m, args.condition, "everystep", "model_data.pkl")
        if os.path.exists(p):
            every[m] = load(p)

    rows = []
    boot_llm_by_dim, est_llm_by_dim = {}, {}

    def add(section, name, dim, metric, est, boots=None, p=None, **extra):
        lo, hi = percentile_ci(boots) if boots is not None else (np.nan, np.nan)
        rows.append(dict(section=section, name=name, dim=dim, metric=metric, estimate=est,
                         ci_low=lo, ci_high=hi, p=p, **extra))

    def boot_r(a, b):
        """a, b: (3, n_scen) -> 리샘플별 pooled r"""
        return np.array([pooled_r(a[:, ix], b[:, ix]) for ix in B])

    for dim in ('desire', 'belief'):
        H = mat(human, dim, cols)

        # [1] 사람 신뢰도
        sh = split_half_reliability(human[RAW[dim]], cols, normalize_belief=(dim == 'belief'), seed=args.seed)
        lo = loso_r(human[RAW[dim]], cols, normalize_belief=(dim == 'belief'))
        rows.append(dict(section="1_human_reliability", name="human", dim=dim, metric="split_half_spearman_brown",
                         estimate=sh['r_sb'], ci_low=sh['ci_low'], ci_high=sh['ci_high'],
                         note=f"r_half={sh['r_half']:.3f}; noise ceiling sqrt(r_sb)={sh['noise_ceiling']:.3f}"))
        rows.append(dict(section="1_human_reliability", name="human", dim=dim, metric="leave_one_subject_out_r",
                         estimate=lo['mean'], ci_low=lo['min'], ci_high=lo['max'], note="ci = min-max over subjects"))

        # [2] 사람 vs 레퍼런스
        boot_ref = {}
        for r, d in refs.items():
            R = mat(d, dim, cols)
            boot_ref[r] = boot_r(R, H)
            add("2_reference_vs_human", r, dim, "r_with_human", pooled_r(R, H), boot_ref[r])

        # [3][4] LLM
        Bt = mat(refs['btom'], dim, cols)
        r_btom_h = pooled_r(Bt, H)
        sec4 = []
        boot_llm_h = {}
        boot_llm_by_dim[dim] = boot_llm_h
        est_llm_by_dim[dim] = {}
        for m, d in llm.items():
            L = mat(d, dim, cols)
            r_lh, r_lb = pooled_r(L, H), pooled_r(L, Bt)
            b_lh = boot_r(L, H)
            boot_llm_h[m] = b_lh
            est_llm_by_dim[dim][m] = r_lh
            add("3_llm", m, dim, "r_with_human", r_lh, b_lh, group="better" if m in BETTER else "worse")
            add("3_llm", m, dim, "r_with_btom", r_lb, boot_r(L, Bt), group="better" if m in BETTER else "worse")
            diff = b_lh - boot_ref['btom']
            t, df, p_w = williams_t(r_lh, r_btom_h, r_lb, n_points)
            sec4.append(dict(section="4_llm_minus_btom_benchmark", name=m, dim=dim,
                             metric="delta_r_with_human (LLM - BToM)", estimate=r_lh - r_btom_h,
                             ci_low=percentile_ci(diff)[0], ci_high=percentile_ci(diff)[1],
                             p=boot_p_two_sided(diff), williams_t=t, williams_df=df, williams_p=p_w,
                             group="better" if m in BETTER else "worse"))
        if sec4:
            ph = holm([r['p'] for r in sec4])
            pw = holm([r['williams_p'] for r in sec4])
            for r, a, b in zip(sec4, ph, pw):
                r['p_holm'], r['williams_p_holm'] = a, b
            rows.extend(sec4)

        # [5] Better vs Worse
        bet = [m for m in BETTER if m in llm]
        wor = [m for m in WORSE if m in llm]
        if bet and wor:
            est = np.mean([pooled_r(mat(llm[m], dim, cols), H) for m in bet]) - \
                  np.mean([pooled_r(mat(llm[m], dim, cols), H) for m in wor])
            bd = np.mean([boot_llm_h[m] for m in bet], axis=0) - np.mean([boot_llm_h[m] for m in wor], axis=0)
            add("5_better_minus_worse", "mean r_with_human", dim, "better - worse", est, bd, boot_p_two_sided(bd))

        # [6] End vs Every
        both = [m for m in BETTER if m in llm and m in every]
        for m in [m for m in BETTER + WORSE if m in llm and m in every]:
            E, V = mat(llm[m], dim, cols), mat(every[m], dim, cols)
            add("6_end_vs_every", m, dim, "r(End, Every final)", pooled_r(E, V), boot_r(E, V))
        if both:
            Es = [mat(llm[m], dim, cols) for m in both]
            Vs = [mat(every[m], dim, cols) for m in both]
            # 모델별 r을 먼저 구해 평균 (mean_of_model_r)
            full = np.arange(len(ids))
            for what, X, Y in (("r(End, Every final)", Es, Vs), ("r(End, human)", Es, None), ("r(Every final, human)", Vs, None)):
                f = lambda ix: np.mean([pooled_r(a[:, ix], (b if Y is not None else H)[:, ix])
                                        for a, b in zip(X, Y if Y is not None else X)])
                add("6_end_vs_every", f"better{len(both)}_mean_of_model_r", dim, what, f(full), np.array([f(ix) for ix in B]))
            for how in ("pooled", "model_mean"):
                if how == "pooled":
                    def agg(arrs, ix):  # 모델을 옆으로 쌓아서 (3, n_scen * n_models)
                        return np.concatenate([a[:, ix] for a in arrs], axis=1)
                    Hx = lambda ix: np.concatenate([H[:, ix]] * len(both), axis=1)
                else:
                    def agg(arrs, ix):  # 모델 평균을 먼저
                        return np.nanmean([a[:, ix] for a in arrs], axis=0)
                    Hx = lambda ix: H[:, ix]
                full = np.arange(len(ids))
                b_ev = np.array([pooled_r(agg(Es, ix), agg(Vs, ix)) for ix in B])
                b_eh = np.array([pooled_r(agg(Es, ix), Hx(ix)) for ix in B])
                b_vh = np.array([pooled_r(agg(Vs, ix), Hx(ix)) for ix in B])
                tag = f"better{len(both)}_{how}"
                add("6_end_vs_every", tag, dim, "r(End, Every final)", pooled_r(agg(Es, full), agg(Vs, full)), b_ev)
                add("6_end_vs_every", tag, dim, "r(End, human)", pooled_r(agg(Es, full), Hx(full)), b_eh)
                add("6_end_vs_every", tag, dim, "r(Every final, human)", pooled_r(agg(Vs, full), Hx(full)), b_vh)
                d = b_vh - b_eh
                add("6_end_vs_every", tag, dim, "r(Every,human) - r(End,human)",
                    pooled_r(agg(Vs, full), Hx(full)) - pooled_r(agg(Es, full), Hx(full)), d, boot_p_two_sided(d))

        # [7] Worse 그룹 레퍼런스 비교
        wor = [m for m in WORSE if m in llm]
        Rm = {r: mat(d, dim, cols) for r, d in refs.items()}
        if wor:
            G = np.nanmean([mat(llm[m], dim, cols) for m in wor], axis=0)
            targets = [("worse_group_mean", G)] + [(m, mat(llm[m], dim, cols)) for m in wor]
            for name, Y in targets:
                est = {r: pooled_r(Y, R) for r, R in Rm.items()}
                bts = {r: boot_r(Y, R) for r, R in Rm.items()}
                for r in Rm:
                    add("7_worse_vs_references", name, dim, f"r_with_{r}", est[r], bts[r])
                order = sorted(est, key=est.get, reverse=True)
                others = order[1:] if name == "worse_group_mean" else order[1:2]
                for r in others:
                    d = bts[order[0]] - bts[r]
                    add("7_worse_vs_references", name, dim, f"r({order[0]}) - r({r})", est[order[0]] - est[r], d,
                        boot_p_two_sided(d), note="best minus other" if name == "worse_group_mean" else "best minus 2nd")

        # [8] End-step VPA (모델별)
        sets = [("p4", [r for r in REFS if r != "hindsight"])] + ([("p5", REFS)] if dim == "belief" else [])
        for tag, prs in sets:
            for m, d in llm.items():
                Y = mat(d, dim, cols)
                pt = vpa(Y.ravel(order="F"), np.column_stack([Rm[r].ravel(order="F") for r in prs]), prs)
                bts = pd.DataFrame([vpa(Y[:, ix].ravel(order="F"),
                                        np.column_stack([Rm[r][:, ix].ravel(order="F") for r in prs]), prs) for ix in B])
                for k, v in pt.items():
                    extra = {"estimate_clipped": max(v, 0.0)} if k.startswith("unique_") or k == "total" else {}
                    add("8_endstep_vpa", m, dim, f"{tag}_{k}", v, bts[k].to_numpy() if k in bts else None,
                        group="better" if m in BETTER else "worse",
                        note="paper predictors (4)" if tag == "p4" else "with Hindsight (5)", **extra)
                # 고유 R² 1위와 2위의 paired 차이 ("best explained by" 판정이 유의한지)
                uq = sorted([k for k in pt if k.startswith("unique_")], key=pt.get, reverse=True)
                if len(uq) >= 2:
                    dd = (bts[uq[0]] - bts[uq[1]]).to_numpy()
                    add("8_endstep_vpa", m, dim, f"{tag}_{uq[0]} - {uq[1]}", pt[uq[0]] - pt[uq[1]], dd,
                        boot_p_two_sided(dd), group="better" if m in BETTER else "worse", note="top minus 2nd")

    # [9] desire / belief 모델 순위 일치
    from scipy.stats import spearmanr
    ms = [m for m in est_llm_by_dim.get("desire", {}) if m in est_llm_by_dim.get("belief", {})]
    if len(ms) >= 3:
        rho = spearmanr([est_llm_by_dim["desire"][m] for m in ms], [est_llm_by_dim["belief"][m] for m in ms])[0]
        bd = np.array([boot_llm_by_dim["desire"][m] for m in ms])  # (n_model, n_boot)
        bb = np.array([boot_llm_by_dim["belief"][m] for m in ms])
        rb = np.array([spearmanr(bd[:, i], bb[:, i])[0] for i in range(bd.shape[1])])
        add("9_rank_agreement", f"{len(ms)} models", "desire_vs_belief", "spearman(r_desire, r_belief)", rho, rb,
            note="scenario bootstrap; does not reflect sampling of models")

    res = pd.DataFrame(rows)
    res.insert(0, "n_scenarios", len(ids))
    out = os.path.join(args.out_dir, f"endstep_stats_{args.condition}{'_nopartial' if args.exclude_partial else ''}{'_stratified' if args.stratified else ''}.csv")
    res.to_csv(out, index=False)
    print_summary(res)
    print(f"\n✅ saved: {out}")


def print_summary(res):
    fmt = lambda r: f"{r.estimate:+.2f} [{r.ci_low:+.2f}, {r.ci_high:+.2f}]"
    for dim in ('desire', 'belief'):
        d = res[res.dim == dim]
        print(f"\n===== {dim.upper()} =====")
        for r in d[d.section == "1_human_reliability"].itertuples():
            print(f"  human {r.metric:28} {fmt(r)}  {r.note}")
        for r in d[d.section == "2_reference_vs_human"].itertuples():
            print(f"  {r.name:16} r with human     {fmt(r)}")
        s4 = d[d.section == "4_llm_minus_btom_benchmark"].set_index('name')
        for r in d[(d.section == "3_llm") & (d.metric == "r_with_human")].itertuples():
            x = s4.loc[r.name]
            print(f"  {r.name:18} r(LLM,human) {fmt(r)} | minus BToM {x.estimate:+.2f} "
                  f"[{x.ci_low:+.2f},{x.ci_high:+.2f}] p_holm={x.p_holm:.3f} | Williams p_holm={x.williams_p_holm:.3f}")
        for r in d[d.section.isin(["5_better_minus_worse", "6_end_vs_every"])].itertuples():
            ptxt = f" p={r.p:.3f}" if pd.notna(r.p) else ""
            print(f"  {r.section[2:]:20} {r.name:22} {r.metric:32} {fmt(r)}{ptxt}")


if __name__ == "__main__":
    main()
