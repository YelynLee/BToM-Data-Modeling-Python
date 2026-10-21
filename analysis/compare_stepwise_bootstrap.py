"""
Stepwise 조건(everystep / prefixstep 등) 간 비교 + 시나리오 단위 paired bootstrap 신뢰구간.

논문 everystep 분석과 같은 지표를 같은 시나리오 집합에서 계산하고,
모든 조건에 '같은 리샘플 인덱스'를 써서 조건 간 차이의 CI까지 구함.

지표
  1) rm_corr (calc_everystep_correlation.py와 같은 정의): 옵션별, 시나리오 내 평균을 뺀 뒤의 상관
       belief: L / M / Empty,  desire: K / L / M  ×  ref: BToM / Hindsight
  2) BToM 우위 = mean_option(r_BToM) - mean_option(r_Hindsight)   (belief)
  3) VPA 고유 설명력 (everystep_vpa_analysis.py와 같은 정의: 옵션을 melt, z-score, OLS adjusted R^2)
       belief 예측변수: btom, truebelief, nocost, motionheuristic, hindsight
       ※ CI를 위해 0으로 자르지(clip) 않은 값을 보고함
       ※ 고유 설명력은 부호가 없으므로 표준화 회귀계수(vpa_*_beta_*)도 함께 보고함

Bootstrap: 그룹(행동 조건)별로 층화하여 시나리오를 복원추출 (n_boot회). 각 시나리오의 모든 스텝은 함께 뽑힘.
Every-step 1명 기준 분포: 피험자 1명씩만 썼을 때의 지표 (피험자 수 차이의 영향 확인용)

사용 예)
  python analysis/compare_stepwise_bootstrap.py \\
     --input results/claude-opus-4-6/vanilla/everystep/everystep_valid_only.csv results/claude-opus-4-6/vanilla/prefixstep \\
     --labels "Every-step" "Prefix-step" --groups 1,4 --per_subject 0 --out_dir results/claude-opus-4-6/vanilla/compare
"""
import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "src")):
    if p not in sys.path:
        sys.path.append(p)

from src.dataset import df_btom
from src.config import get_group_indices
from src.prepare_everystep import apply_phase_labeling, load_reference_everystep
from analysis.plot_everystep import get_phase_index

OPTS = {"belief": ["L", "M", "Empty"], "desire": ["K", "L", "M"]}
CORR_REFS = ["btom", "hindsight"]
VPA_REFS = {"belief": ["btom", "truebelief", "nocost", "motionheuristic", "hindsight"]}


# -------------------------------------------------------------------------
# 데이터
# -------------------------------------------------------------------------
def load_rows(path):
    """subject_*.csv 폴더, csv 하나, 또는 everystep_valid_only.csv(subject_id 포함) 모두 지원."""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "subject_*.csv")))
        frames = []
        for f in files:
            d = pd.read_csv(f)
            d["subject_id"] = int(os.path.basename(f).split("_")[1].split(".")[0])
            frames.append(d)
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.read_csv(path)
        if "subject_id" not in df.columns:
            df["subject_id"] = 1
    if "error" in df.columns:
        df = df[df["error"].isna()]
    cols = ["subject_id", "scenario_id", "time_step"] + [f"{s}_{o}" for s in OPTS for o in OPTS[s]]
    df = df[cols].dropna()
    # 모든 스텝이 있는 (subject, scenario)만
    exp = df_btom.groupby("scenario_id").size()
    n = df.groupby(["subject_id", "scenario_id"]).size().reset_index(name="n")
    ok = n[n["n"] == n["scenario_id"].map(exp)][["subject_id", "scenario_id"]]
    return df.merge(ok, on=["subject_id", "scenario_id"])


def to_prob(block):
    shifted = np.maximum(block - 1, 0)
    return shifted.div(shifted.sum(axis=1).replace(0, 1.0), axis=0)


_PHASE_CACHE = {}


def phases():
    """(scenario_id, time_step) -> group_id, phase_idx  (궤적만으로 정해지므로 한 번만 계산)"""
    if "p" not in _PHASE_CACHE:
        d = df_btom.copy()
        d["subject_id"] = 0
        d = apply_phase_labeling(d)
        d["phase_idx"] = [get_phase_index(g, p) for g, p in zip(d["group_id"], d["phase"])]
        _PHASE_CACHE["p"] = d[["scenario_id", "time_step", "group_id", "phase_idx"]]
    return _PHASE_CACHE["p"]


def aggregate(df, subjects=None):
    """피험자 평균 -> belief 확률 정규화 (논문 파이프라인과 동일 순서)."""
    if subjects is not None:
        df = df[df["subject_id"].isin(subjects)]
    m = df.groupby(["scenario_id", "time_step"], as_index=False).mean(numeric_only=True)
    bcols = [f"belief_{o}" for o in OPTS["belief"]]
    m[bcols] = to_prob(m[bcols]).values
    return m.drop(columns=["subject_id"])


def attach_refs(m, refs):
    for name, r in refs.items():
        cols = [f"{s}_{o}" for s in OPTS for o in OPTS[s]]
        m = m.merge(r[["scenario_id", "time_step"] + cols].rename(columns={c: f"{name}__{c}" for c in cols}),
                    on=["scenario_id", "time_step"], how="left")
    m = m.merge(phases(), on=["scenario_id", "time_step"], how="left")
    return m[m["phase_idx"] != 9].sort_values(["scenario_id", "time_step"]).reset_index(drop=True)


# -------------------------------------------------------------------------
# 지표 (시나리오 가중치 w 를 받아 bootstrap에 재사용)
# -------------------------------------------------------------------------
class Metrics:
    def __init__(self, m, scenario_order):
        self.m = m
        self.sc = scenario_order
        self.idx = {s: np.where(m["scenario_id"].values == s)[0] for s in scenario_order}
        # rm_corr 충분통계량: 시나리오별 within-centered Sxy, Sxx, Syy
        self.suff = {}
        for score, opts in OPTS.items():
            for o in opts:
                col = f"{score}_{o}"
                for ref in CORR_REFS:
                    rc = f"{ref}__{col}"
                    xs, ys, xy = [], [], []
                    for s in scenario_order:
                        x = m[col].values[self.idx[s]]; y = m[rc].values[self.idx[s]]
                        x = x - x.mean(); y = y - y.mean()
                        xy.append((x * y).sum()); xs.append((x * x).sum()); ys.append((y * y).sum())
                    self.suff[(score, o, ref)] = tuple(map(np.array, (xy, xs, ys)))

    def rm_corr(self, w):
        out = {}
        for (score, o, ref), (xy, xs, ys) in self.suff.items():
            den = np.sqrt((w * xs).sum() * (w * ys).sum())
            out[f"rm_{score}_{o}_{ref}"] = (w * xy).sum() / den if den > 0 else np.nan
        for score, opts in OPTS.items():
            for ref in CORR_REFS:
                out[f"rm_{score}_mean_{ref}"] = np.nanmean([out[f"rm_{score}_{o}_{ref}"] for o in opts])
            out[f"rm_{score}_btom_minus_hindsight"] = out[f"rm_{score}_mean_btom"] - out[f"rm_{score}_mean_hindsight"]
        return out

    def vpa(self, w):
        out = {}
        rows = np.concatenate([np.repeat(self.idx[s], k) for s, k in zip(self.sc, w) if k > 0])
        d = self.m.iloc[rows]
        for score, refs in VPA_REFS.items():
            y = np.concatenate([d[f"{score}_{o}"].values for o in OPTS[score]])
            X = np.column_stack([np.concatenate([d[f"{r}__{score}_{o}"].values for o in OPTS[score]]) for r in refs])
            ok = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
            y, X = y[ok], X[ok]
            sd = X.std(axis=0)
            keep = sd > 0
            y = (y - y.mean()) / y.std()
            X = (X[:, keep] - X[:, keep].mean(axis=0)) / sd[keep]
            names = [r for r, k in zip(refs, keep) if k]
            full, beta = _adj_r2(y, X, return_beta=True)
            out[f"vpa_{score}_total"] = full
            for i, r in enumerate(names):
                out[f"vpa_{score}_unique_{r}"] = full - _adj_r2(y, np.delete(X, i, axis=1))
                # 표준화 회귀계수: 고유 설명력은 부호를 구분하지 않으므로 방향을 함께 확인해야 함
                # (예: 예측변수끼리 상관이 높으면 한쪽이 음수가 되는 억제 효과)
                out[f"vpa_{score}_beta_{r}"] = beta[i]
        return out

    def all(self, w=None):
        w = np.ones(len(self.sc)) if w is None else w
        return {**self.rm_corr(w), **self.vpa(w)}


def _adj_r2(y, X, return_beta=False):
    n, k = X.shape
    A = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
    adj = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    return (adj, beta[1:]) if return_beta else adj


def boot_weights(scenarios, groups_of, n_boot, seed):
    """그룹별 층화 시나리오 복원추출 -> 시나리오별 뽑힌 횟수(가중치) 행렬 (n_boot x n_scen)."""
    rng = np.random.default_rng(seed)
    pos = {s: i for i, s in enumerate(scenarios)}
    W = np.zeros((n_boot, len(scenarios)))
    by_group = {}
    for s in scenarios:
        by_group.setdefault(groups_of[s], []).append(s)
    for b in range(n_boot):
        for g, ss in by_group.items():
            for s in rng.choice(ss, size=len(ss), replace=True):
                W[b, pos[s]] += 1
    return W


# -------------------------------------------------------------------------
# 그림: 조건별 점추정 + 95% bootstrap CI (forest plot)
# -------------------------------------------------------------------------
COND_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # 고정 순서 (조건 순서대로)
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
]


def plot_summary(res, labels, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    base = res[res["condition"].isin(labels)]
    fig, axes = plt.subplots(1, len(PANELS), figsize=(5.2 * len(PANELS), 3.6), squeeze=False)
    n = len(labels)
    for ax, (ptitle, items, xlim) in zip(axes[0], PANELS):
        for i, (metric, name) in enumerate(items):
            for j, lab in enumerate(labels):
                row = base[(base["metric"] == metric) & (base["condition"] == lab)]
                if row.empty:
                    continue
                r = row.iloc[0]
                y = i + (j - (n - 1) / 2) * 0.22
                ax.plot([r["ci_low"], r["ci_high"]], [y, y], color=COND_COLORS[j], lw=2, solid_capstyle="round")
                ax.plot(r["estimate"], y, "o", color=COND_COLORS[j], ms=8, markeredgecolor="white",
                        markeredgewidth=1.5, label=lab if i == 0 else None, zorder=3)
        ax.axvline(0, color="#999999", lw=0.8, zorder=0)
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


# -------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--groups", default="1,4", help="분석할 group_id (기본: Check-GoBack Present/Absent)")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per_subject", type=int, default=0,
                    help="이 번호의 입력에 대해 피험자 1명씩의 지표 분포도 계산 (0-based, -1이면 안 함)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--title", default="Stepwise conditions vs reference models (95% scenario-bootstrap CI)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    gids = [int(g) for g in args.groups.split(",")]
    allowed = get_group_indices(include_irrational=False)
    scen_ids = sorted(s for g in gids for s in allowed[g - 1])
    groups_of = df_btom.groupby("scenario_id")["group_id"].first().to_dict()

    refs = {r: load_reference_everystep(r) for r in sorted(set(CORR_REFS) | set(VPA_REFS["belief"]))}

    raws, mets = [], []
    for path, label in zip(args.input, args.labels):
        df = load_rows(path)
        df = df[df["scenario_id"].isin(scen_ids)]
        have = set(df["scenario_id"])
        raws.append(df)
        print(f"[{label}] subjects={df['subject_id'].nunique()}, scenarios={len(have)}/{len(scen_ids)}")
        scen_ids = [s for s in scen_ids if s in have]
    # 모든 조건에 공통으로 있는 시나리오만 사용
    for df, label in zip(raws, args.labels):
        m = attach_refs(aggregate(df[df["scenario_id"].isin(scen_ids)]), refs)
        mets.append(Metrics(m, scen_ids))
    print(f"common scenarios: {len(scen_ids)}")

    W = boot_weights(scen_ids, groups_of, args.n_boot, args.seed)
    point = [mt.all() for mt in mets]
    boots = [pd.DataFrame([mt.all(w) for w in W]) for mt in mets]

    def ci(arr):
        arr = np.asarray(arr, float); arr = arr[~np.isnan(arr)]
        return (np.percentile(arr, 2.5), np.percentile(arr, 97.5)) if arr.size else (np.nan, np.nan)

    rows = []
    for label, pt, bt, df in zip(args.labels, point, boots, raws):
        for k, v in pt.items():
            lo, hi = ci(bt[k])
            rows.append({"condition": label, "n_subjects": df["subject_id"].nunique(), "metric": k,
                         "estimate": v, "ci_low": lo, "ci_high": hi})
    # 조건 간 paired 차이 (첫 조건 대비)
    for j in range(1, len(mets)):
        for k in point[0]:
            d = boots[j][k] - boots[0][k]
            lo, hi = ci(d)
            p = 2 * min((d > 0).mean(), (d < 0).mean())
            rows.append({"condition": f"{args.labels[j]} - {args.labels[0]}", "n_subjects": None, "metric": k,
                         "estimate": point[j][k] - point[0][k], "ci_low": lo, "ci_high": hi, "p_boot": p})
    res = pd.DataFrame(rows)

    # 피험자 1명씩의 분포
    if args.per_subject >= 0:
        df = raws[args.per_subject]
        subj_rows = []
        for s in sorted(df["subject_id"].unique()):
            m = attach_refs(aggregate(df[df["scenario_id"].isin(scen_ids)], subjects=[s]), refs)
            subj_rows.append({"subject_id": s, **Metrics(m, scen_ids).all()})
        sub = pd.DataFrame(subj_rows)
        sub.to_csv(os.path.join(args.out_dir, "per_subject_metrics.csv"), index=False)
        for k in point[0]:
            rows.append({"condition": f"{args.labels[args.per_subject]} (1 subject, mean of {len(sub)})",
                         "n_subjects": 1, "metric": k, "estimate": sub[k].mean(),
                         "ci_low": sub[k].min(), "ci_high": sub[k].max()})
        res = pd.DataFrame(rows)

    res.to_csv(os.path.join(args.out_dir, "stepwise_bootstrap_summary.csv"), index=False)
    plot_summary(res, args.labels, os.path.join(args.out_dir, "stepwise_bootstrap_summary.png"), args.title)
    print(f"✅ saved: {os.path.join(args.out_dir, 'stepwise_bootstrap_summary.csv')}")
    return res


if __name__ == "__main__":
    main()
