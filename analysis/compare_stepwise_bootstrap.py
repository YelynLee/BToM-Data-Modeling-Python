"""
Stepwise 조건(everystep / prefixstep 등) 비교 + 신뢰구간 / 검정.

같은 시나리오 집합에서 논문 everystep 지표를 조건별로 계산하고, 첫 번째 조건 대비 차이를 검정함.

지표
  1) rm_corr (calc_everystep_correlation.py와 같은 정의): 옵션별, 시나리오 내 평균을 뺀 뒤의 상관
       belief: L / M / Empty,  desire: K / L / M  ×  ref: BToM / Hindsight  (+ 옵션 평균, BToM - Hindsight)
  2) VPA (everystep_vpa_analysis.py와 같은 정의, belief): 고유 adjusted R^2 + 표준화 β(부호)
       예측변수: btom, truebelief, nocost, motionheuristic, hindsight  (CI를 위해 0으로 자르지 않음)
  3) hindsight 지수 (belief): 시나리오마다 h_s = Σ_t (LLM - BToM)·(Hind - BToM) / Σ_t ||Hind - BToM||^2
       = LLM이 BToM에서 벗어난 정도를 Hindsight가 BToM에서 벗어난 방향에 투영한 기울기 (원점 통과 회귀).
       0 = BToM과 같음, 1 = Hindsight만큼 벗어남, 1 초과 = Hindsight보다 더 크게 같은 방향으로 벗어남.
       ||Hind - BToM|| < 0.05인 스텝은 제외 (두 모델이 사실상 같은 예측). 두 모델 차이가 큰 스텝에 가중치가 실려
       스텝별 비율 평균보다 안정적. 시나리오 간 평균으로 집계하며 구간(phase)별로도 계산.
  4) |LLM - BToM| 평균 절대오차 (belief, 시나리오 평균 후 평균)

신뢰구간
  - 2단계 bootstrap: 시나리오를 행동 그룹별 층화 복원추출(모든 조건에 같은 리샘플) ->
    각 조건 안에서 피험자를 복원추출 (피험자가 1명이면 시나리오만)
  - 첫 조건 대비 차이: 같은 시나리오 리샘플에서의 차이 분포 -> CI, bootstrap p
검정 (첫 조건 대비)
  - 시나리오 단위 paired sign-flip 순열 검정: hindsight 지수, |LLM - BToM| (시나리오 평균 차이)
피험자 수 맞추기
  - --match_subjects k: 첫 조건(보통 Every-step 16명)에서 k명을 무작위로 여러 번 뽑아 지표 분포를 보고
    (피험자 수 차이가 결론을 바꾸지 않는지 확인)

사용 예)
  python analysis/compare_stepwise_bootstrap.py \\
     --input results/claude-opus-4-6/vanilla/everystep/everystep_valid_only.csv \\
             results/claude-opus-4-6/vanilla/prefixstep/thinking_off \\
     --labels "Every-step" "Prefix-step" --groups 1,4 --match_subjects 5 \\
     --out_dir results/claude-opus-4-6/vanilla/compare
"""
import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "src"), os.path.join(ROOT, "analysis")):
    if p not in sys.path:
        sys.path.append(p)

from src.dataset import df_btom
from src.config import get_group_indices
from src.prepare_everystep import apply_phase_labeling, load_reference_everystep
from analysis.plot_everystep import get_phase_index
from stats_utils import (stratified_boot_indices, percentile_ci, boot_p_two_sided, paired_signflip_test)

OPTS = {"belief": ["L", "M", "Empty"], "desire": ["K", "L", "M"]}
# 열 순서를 명시적으로 고정: desire 3개 다음 belief 3개 (BEL 슬라이스가 이 순서에 의존)
COLS = [f"desire_{o}" for o in OPTS["desire"]] + [f"belief_{o}" for o in OPTS["belief"]]
BEL = slice(3, 6)
assert all(c.startswith("belief_") for c in COLS[BEL])
CORR_REFS = ["btom", "hindsight"]
VPA_REFS = ["btom", "truebelief", "nocost", "motionheuristic", "hindsight"]
ALL_REFS = sorted(set(CORR_REFS) | set(VPA_REFS))
H_EPS = 0.05
PHASE_NAMES = {0: "Start", 1: "Appr G1", 2: "Pass G1", 3: "See G2", 4: "Stop", 5: "Return/Appr G2", 6: "Selected"}


# =========================================================================
# 데이터 준비
# =========================================================================
def load_rows(path):
    """subject_*.csv 폴더, csv 하나, 또는 everystep_valid_only.csv(subject_id 포함) 모두 지원."""
    if os.path.isdir(path):
        frames = []
        for f in sorted(glob.glob(os.path.join(path, "subject_*.csv"))):
            d = pd.read_csv(f)
            d["subject_id"] = int(os.path.basename(f).split("_")[1].split(".")[0])
            frames.append(d)
        if not frames:
            raise FileNotFoundError(f"No subject_*.csv in {path}")
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.read_csv(path)
        if "subject_id" not in df.columns:
            df["subject_id"] = 1
    if "error" in df.columns:
        df = df[df["error"].isna()]
    df = df[["subject_id", "scenario_id", "time_step"] + COLS].dropna()
    exp = df_btom.groupby("scenario_id").size()
    n = df.groupby(["subject_id", "scenario_id"]).size().reset_index(name="n")
    ok = n[n["n"] == n["scenario_id"].map(exp)][["subject_id", "scenario_id"]]
    return df.merge(ok, on=["subject_id", "scenario_id"])


def row_table(scen_ids):
    """(scenario, time_step) 행 목록 + phase_idx. 궤적만으로 정해짐."""
    d = df_btom[df_btom["scenario_id"].isin(scen_ids)].copy()
    d["subject_id"] = 0
    d = apply_phase_labeling(d)
    d["phase_idx"] = [get_phase_index(g, p) for g, p in zip(d["group_id"], d["phase"])]
    d = d[d["phase_idx"] != 9].sort_values(["scenario_id", "time_step"]).reset_index(drop=True)
    return d[["scenario_id", "time_step", "group_id", "phase_idx"]]


def ref_arrays(rows, refs):
    out = {}
    for name, r in refs.items():
        m = rows[["scenario_id", "time_step"]].merge(r[["scenario_id", "time_step"] + COLS],
                                                   on=["scenario_id", "time_step"], how="left")
        out[name] = m[COLS].to_numpy(float)
    return out


def subject_array(df, rows):
    """피험자 x 행 x 6 (원점수). 해당 피험자에게 없는 시나리오는 NaN."""
    subs = sorted(df["subject_id"].unique())
    arr = np.full((len(subs), len(rows), len(COLS)), np.nan)
    key = rows[["scenario_id", "time_step"]]
    for i, s in enumerate(subs):
        m = key.merge(df[df["subject_id"] == s], on=["scenario_id", "time_step"], how="left")
        arr[i] = m[COLS].to_numpy(float)
    return arr, subs


def aggregate(arr, subj_w=None):
    """피험자 (가중)평균 -> belief 확률 정규화 (논문 파이프라인과 같은 순서). Returns: (n_rows, 6)"""
    w = np.ones(arr.shape[0]) if subj_w is None else np.asarray(subj_w, float)
    valid = ~np.isnan(arr)
    num = np.nansum(arr * w[:, None, None], axis=0)
    den = np.sum(valid * w[:, None, None], axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = num / den
    b = np.maximum(m[:, BEL] - 1, 0)
    tot = b.sum(axis=1, keepdims=True)
    tot[tot == 0] = 1.0
    m[:, BEL] = b / tot
    return m


# =========================================================================
# 지표
# =========================================================================
class Engine:
    def __init__(self, rows, refs):
        self.rows = rows
        self.refs = refs
        self.scen = rows["scenario_id"].to_numpy()
        self.scen_ids = sorted(np.unique(self.scen))
        self.pos = {s: i for i, s in enumerate(self.scen_ids)}
        self.sidx = np.array([self.pos[s] for s in self.scen])        # 행 -> 시나리오 위치
        self.n_s = len(self.scen_ids)
        self.phase = rows["phase_idx"].to_numpy()
        d = refs["hindsight"][:, BEL] - refs["btom"][:, BEL]
        self.h_den = (d ** 2).sum(axis=1)
        self.h_ok = self.h_den >= H_EPS ** 2
        self.h_dir = d

    def _center(self, x):
        """시나리오 내 평균 제거"""
        cnt = np.bincount(self.sidx, minlength=self.n_s)
        mean = np.bincount(self.sidx, weights=x, minlength=self.n_s) / np.maximum(cnt, 1)
        return x - mean[self.sidx]

    def per_scenario_mean(self, v, mask):
        num = np.bincount(self.sidx[mask], weights=v[mask], minlength=self.n_s)
        cnt = np.bincount(self.sidx[mask], minlength=self.n_s)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(cnt > 0, num / cnt, np.nan)

    def hindsight_by_scenario(self, agg, mask):
        """시나리오별 Σ(LLM-BToM)·(Hind-BToM) / Σ||Hind-BToM||^2 (mask 안의 스텝만)."""
        proj = ((agg[:, BEL] - self.refs["btom"][:, BEL]) * self.h_dir).sum(axis=1)
        m = mask & self.h_ok & ~np.isnan(proj)
        num = np.bincount(self.sidx[m], weights=proj[m], minlength=self.n_s)
        den = np.bincount(self.sidx[m], weights=self.h_den[m], minlength=self.n_s)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / den, np.nan)

    def metrics(self, agg, sw):
        """agg: (n_rows, 6), sw: 시나리오 가중치(뽑힌 횟수)."""
        out = {}
        rw = sw[self.sidx]
        # 1) rm_corr
        for j, col in enumerate(COLS):
            x = self._center(agg[:, j])
            for ref in CORR_REFS:
                y = self._center(self.refs[ref][:, j])
                ok = ~np.isnan(x) & ~np.isnan(y)
                den = np.sqrt((rw[ok] * x[ok] ** 2).sum() * (rw[ok] * y[ok] ** 2).sum())
                out[f"rm_{col}_{ref}"] = (rw[ok] * x[ok] * y[ok]).sum() / den if den > 0 else np.nan
        for score, opts in OPTS.items():
            for ref in CORR_REFS:
                out[f"rm_{score}_mean_{ref}"] = np.nanmean([out[f"rm_{score}_{o}_{ref}"] for o in opts])
            out[f"rm_{score}_btom_minus_hindsight"] = out[f"rm_{score}_mean_btom"] - out[f"rm_{score}_mean_hindsight"]
        # 2) VPA (belief): 가중치만큼 행을 복제
        rep = np.repeat(np.arange(len(agg)), rw.astype(int))
        y = agg[rep][:, BEL].ravel(order="F")
        X = np.column_stack([self.refs[r][rep][:, BEL].ravel(order="F") for r in VPA_REFS])
        ok = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
        y, X = y[ok], X[ok]
        keep = X.std(axis=0) > 0
        names = [r for r, k in zip(VPA_REFS, keep) if k]
        yz = (y - y.mean()) / y.std()
        Xz = (X[:, keep] - X[:, keep].mean(axis=0)) / X[:, keep].std(axis=0)
        full, beta = _adj_r2(yz, Xz, return_beta=True)
        out["vpa_belief_total"] = full
        for i, r in enumerate(names):
            out[f"vpa_belief_unique_{r}"] = full - _adj_r2(yz, np.delete(Xz, i, axis=1))
            out[f"vpa_belief_beta_{r}"] = beta[i]
        # 3) hindsight 지수 / 4) |LLM - BToM|
        all_rows = np.ones(len(agg), bool)
        h_sc = self.hindsight_by_scenario(agg, all_rows)
        err = np.abs(agg[:, BEL] - self.refs["btom"][:, BEL]).mean(axis=1)
        e_sc = self.per_scenario_mean(err, np.ones(len(err), bool))
        out["hindsight_index"] = _wmean(h_sc, sw)
        out["abs_err_belief_vs_btom"] = _wmean(e_sc, sw)
        for p, name in PHASE_NAMES.items():
            m = self.h_ok & (self.phase == p)
            if m.any():
                out[f"hindsight_index_phase{p}"] = _wmean(self.hindsight_by_scenario(agg, self.phase == p), sw)
        return out

    def scenario_level(self, agg):
        """순열 검정용: 시나리오별 hindsight 지수와 |LLM - BToM|."""
        err = np.abs(agg[:, BEL] - self.refs["btom"][:, BEL]).mean(axis=1)
        return {"hindsight_index": self.hindsight_by_scenario(agg, np.ones(len(agg), bool)),
                "abs_err_belief_vs_btom": self.per_scenario_mean(err, np.ones(len(err), bool))}


def _wmean(v, w):
    ok = ~np.isnan(v) & (w > 0)
    return float((v[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else np.nan


def _adj_r2(y, X, return_beta=False):
    n, k = X.shape
    A = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
    adj = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    return (adj, beta[1:]) if return_beta else adj


# =========================================================================
# 그림
# =========================================================================
COND_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # 고정 순서
PANELS = [
    ("rm_corr with reference (within-scenario)", [
        ("rm_belief_mean_btom", "Belief ~ BToM"),
        ("rm_belief_mean_hindsight", "Belief ~ Hindsight"),
        ("rm_desire_mean_btom", "Desire ~ BToM"),
    ], (-1, 1)),
    ("Belief VPA: unique adjusted R²", [
        ("vpa_belief_unique_btom", "BToM"),
        ("vpa_belief_unique_hindsight", "Hindsight"),
    ], None),
    ("Belief VPA: standardized β (direction)", [
        ("vpa_belief_beta_btom", "BToM"),
        ("vpa_belief_beta_hindsight", "Hindsight"),
    ], None),
    ("Hindsight index (0 = BToM, 1 = Hindsight)", [
        ("hindsight_index", "All steps"),
        ("hindsight_index_phase3", "See G2"),
        ("hindsight_index_phase5", "Return / Appr G2"),
        ("hindsight_index_phase6", "Selected"),
    ], None),
]


def plot_summary(res, labels, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    base = res[res["condition"].isin(labels)]
    fig, axes = plt.subplots(1, len(PANELS), figsize=(4.8 * len(PANELS), 3.8), squeeze=False)
    n = len(labels)
    for ax, (ptitle, items, xlim) in zip(axes[0], PANELS):
        for i, (metric, name) in enumerate(items):
            for j, lab in enumerate(labels):
                row = base[(base["metric"] == metric) & (base["condition"] == lab)]
                if row.empty or pd.isna(row.iloc[0]["estimate"]):
                    continue
                r = row.iloc[0]
                y = i + (j - (n - 1) / 2) * 0.22
                ax.plot([r["ci_low"], r["ci_high"]], [y, y], color=COND_COLORS[j], lw=2, solid_capstyle="round")
                ax.plot(r["estimate"], y, "o", color=COND_COLORS[j], ms=8, markeredgecolor="white",
                        markeredgewidth=1.5, label=lab if i == 0 else None, zorder=3)
        ax.axvline(0, color="#999999", lw=0.8, zorder=0)
        if "Hindsight index" in ptitle:
            ax.axvline(1, color="#cccccc", lw=0.8, ls="--", zorder=0)
        ax.set_yticks(range(len(items)))
        ax.set_yticklabels([nm for _, nm in items], fontsize=9.5)
        ax.invert_yaxis()
        ax.set_title(ptitle, fontsize=10.5, loc="left")
        if xlim:
            ax.set_xlim(*xlim)
        ax.grid(axis="x", color="#EEEEEE", lw=0.6)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", length=0)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=n, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ saved: {out_path}")


# =========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--groups", default="1,2,3,4,5,6,7", help="분석할 group_id (기본: 전체 7개 그룹 = irrational 제외 73개)")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--n_perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_subject_boot", action="store_true", help="2단계 대신 시나리오만 리샘플")
    ap.add_argument("--match_subjects", type=int, default=0,
                    help="첫 조건에서 k명을 무작위로 뽑아 반복 계산한 분포를 함께 보고 (0이면 안 함)")
    ap.add_argument("--n_match", type=int, default=200)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--title", default="Stepwise conditions vs reference models (95% bootstrap CI)")
    args = ap.parse_args()
    assert len(args.input) == len(args.labels), "--input과 --labels 개수가 같아야 합니다."
    os.makedirs(args.out_dir, exist_ok=True)

    gids = [int(g) for g in args.groups.split(",")]
    allowed = get_group_indices(include_irrational=False)
    scen_ids = sorted(s for g in gids for s in allowed[g - 1])

    raws = []
    for path, label in zip(args.input, args.labels):
        df = load_rows(path)
        df = df[df["scenario_id"].isin(scen_ids)]
        raws.append(df)
        print(f"[{label}] subjects={df['subject_id'].nunique()}, scenarios={df['scenario_id'].nunique()}/{len(scen_ids)}")
        scen_ids = [s for s in scen_ids if s in set(df["scenario_id"])]
    print(f"common scenarios: {len(scen_ids)}")

    rows = row_table(scen_ids)
    refs = ref_arrays(rows, {r: load_reference_everystep(r) for r in ALL_REFS})
    eng = Engine(rows, refs)
    arrays = [subject_array(df[df["scenario_id"].isin(scen_ids)], rows)[0] for df in raws]
    group_of = df_btom.groupby("scenario_id")["group_id"].first().to_dict()

    # 점추정
    ones = np.ones(eng.n_s)
    aggs = [aggregate(a) for a in arrays]
    point = [eng.metrics(g, ones) for g in aggs]

    # 2단계 bootstrap (시나리오 리샘플은 조건 간 공유)
    rng = np.random.default_rng(args.seed + 1)
    B = stratified_boot_indices(eng.scen_ids, group_of, args.n_boot, args.seed)
    boots = [[] for _ in arrays]
    for ix in B:
        sw = np.bincount(ix, minlength=eng.n_s).astype(float)
        for j, a in enumerate(arrays):
            if a.shape[0] > 1 and not args.no_subject_boot:
                subj_w = np.bincount(rng.integers(0, a.shape[0], a.shape[0]), minlength=a.shape[0])
                g = aggregate(a, subj_w)
            else:
                g = aggs[j]
            boots[j].append(eng.metrics(g, sw))
    boots = [pd.DataFrame(b) for b in boots]

    out = []
    for j, (label, pt) in enumerate(zip(args.labels, point)):
        for k, v in pt.items():
            lo, hi = percentile_ci(boots[j][k]) if k in boots[j] else (np.nan, np.nan)
            out.append(dict(condition=label, n_subjects=arrays[j].shape[0], metric=k, estimate=v, ci_low=lo, ci_high=hi))
    # 첫 조건 대비 차이
    sc0 = eng.scenario_level(aggs[0])
    perm_rows = []
    for j in range(1, len(arrays)):
        name = f"{args.labels[j]} - {args.labels[0]}"
        for k in point[0]:
            if k not in boots[j] or k not in boots[0]:
                continue
            d = boots[j][k] - boots[0][k]
            lo, hi = percentile_ci(d)
            out.append(dict(condition=name, metric=k, estimate=point[j][k] - point[0][k], ci_low=lo, ci_high=hi,
                            p_boot=boot_p_two_sided(d)))
        scj = eng.scenario_level(aggs[j])
        for k in sc0:
            diff = scj[k] - sc0[k]
            mean_d, p = paired_signflip_test(diff, n_perm=args.n_perm, seed=args.seed)
            out.append(dict(condition=name, metric=f"{k} (scenario-level paired sign-flip)", estimate=mean_d,
                            p_perm=p, note=f"n_scenarios={int(np.sum(~np.isnan(diff)))}"))
            perm_rows.append(pd.DataFrame({"scenario_id": eng.scen_ids, "comparison": name, "metric": k,
                                           args.labels[0]: sc0[k], args.labels[j]: scj[k], "diff": diff}))

    # 피험자 수 맞추기
    if args.match_subjects and arrays[0].shape[0] > args.match_subjects:
        k = args.match_subjects
        rngm = np.random.default_rng(args.seed + 2)
        dist = []
        for _ in range(args.n_match):
            pick = rngm.choice(arrays[0].shape[0], size=k, replace=False)
            dist.append(eng.metrics(aggregate(arrays[0][pick]), ones))
        dist = pd.DataFrame(dist)
        for m in dist.columns:
            lo, hi = percentile_ci(dist[m])
            out.append(dict(condition=f"{args.labels[0]} ({k} random subjects x {args.n_match})", n_subjects=k,
                            metric=m, estimate=dist[m].mean(), ci_low=lo, ci_high=hi,
                            note="ci = 2.5-97.5 percentile over random subsets"))

    res = pd.DataFrame(out)
    res.insert(0, "n_scenarios", len(scen_ids))
    res.to_csv(os.path.join(args.out_dir, "stepwise_bootstrap_summary.csv"), index=False)
    if perm_rows:
        pd.concat(perm_rows).to_csv(os.path.join(args.out_dir, "stepwise_scenario_level.csv"), index=False)
    plot_summary(res, args.labels, os.path.join(args.out_dir, "stepwise_bootstrap_summary.png"), args.title)
    print(f"✅ saved: {os.path.join(args.out_dir, 'stepwise_bootstrap_summary.csv')}")
    return res


if __name__ == "__main__":
    main()
