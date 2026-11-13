"""
리뷰어 응답용 그림 / 표 (모델 여러 개를 한 그림에 비교).

  R1 : Every-step / Every-step-single / Prefix-step 분해 (1×4)
       (a) hindsight 지수  (b) 마지막 스텝 belief r(같은 모델 End-step)
       (c) 마지막 스텝 belief r(사람) + End-step  (d) 판단당 생성 토큰
  S1 : 주요 지표 표 (조건별 추정치 [95% CI], Every-step 대비 차이 [CI], bootstrap p, 순열 p)
  C1 : control 사실 문항별 정확도 (q12 제외), 격자 경계 해석과 무관한 기준 함께 표시
  C2 : 시나리오별 사실 정확도 vs End-step |LLM - 사람| (desire, belief), Spearman + bootstrap CI

입력
  --models     표시 이름 (예: "Claude-4.6-Opus" "Gemini-2.5-Pro")
  --compare    모델별 compare_stepwise_bootstrap 출력 폴더 (stepwise_bootstrap_summary.csv, final_step.csv,
               tokens.csv가 있는 곳; 조건 이름은 "Every-step", "Every-step-single", "Prefix-step")
  --control    모델별 control 결과 폴더 (control_scored.csv가 있는 곳)
  --endstep    모델별 End-step model_data.pkl (C2)

사용 예)
  python analysis/plot_reviewer_figures.py --models "Claude-4.6-Opus" "Gemini-2.5-Pro" \\
     --compare results/claude-opus-4-6/vanilla/compare_2x2 results/gemini-2.5-pro/vanilla/compare_2x2 \\
     --control results/claude-opus-4-6/vanilla/control results/gemini-2.5-pro/vanilla/control \\
     --endstep results/claude-opus-4-6/vanilla/model_data.pkl results/gemini-2.5-pro/vanilla/model_data.pkl \\
     --out_dir results/reviewer
"""
import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "src"), os.path.join(ROOT, "analysis")):
    if p not in sys.path:
        sys.path.append(p)

from src.config import HUMAN_PKL_PATH, get_group_indices
from stats_utils import boot_indices, set_stratified, percentile_ci

# 기준 팔레트(dataviz 기본 categorical 1, 2번) — 모델 순서대로 고정
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
CONDS = ["Every-step", "Every-step-single", "Prefix-step"]
COND_SHORT = {"Every-step": "Every-step", "Every-step-single": "Every-step\n-single",
              "Prefix-step": "Prefix-step", "End-step": "End-step"}

plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})


def _style(ax):
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)


def _offsets(n, width=0.22):
    return [(i - (n - 1) / 2) * width for i in range(n)]


# =========================================================================
# R1
# =========================================================================
def fig_r1(models, compare_dirs, out):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.1))
    offs = _offsets(len(models))
    for mi, (name, d) in enumerate(zip(models, compare_dirs)):
        col = MODEL_COLORS[mi]
        s = pd.read_csv(os.path.join(d, "stepwise_bootstrap_summary.csv"))
        f = pd.read_csv(os.path.join(d, "final_step.csv"))
        t = pd.read_csv(os.path.join(d, "tokens.csv"))

        # (a) hindsight index
        rows = [s[(s.condition == c) & (s.metric == "hindsight_index")].iloc[0] for c in CONDS]
        x = np.arange(len(CONDS)) + offs[mi]
        est = [r.estimate for r in rows]
        axes[0].plot(x, est, color=col, lw=1.4, alpha=0.6, zorder=2)
        axes[0].errorbar(x, est, yerr=[[e - r.ci_low for e, r in zip(est, rows)], [r.ci_high - e for e, r in zip(est, rows)]],
                         fmt="o", color=col, ms=7, capsize=0, elinewidth=2, markeredgecolor="white",
                         markeredgewidth=1.2, zorder=3, label=name)

        # (b) final-step belief r(human) incl. End-step
        conds_b = CONDS + ["End-step"]
        fb = f[(f.dimension == "belief") & (f.versus == "Human")].set_index("condition")
        xb = np.arange(len(conds_b)) + offs[mi]
        eb = [fb.loc[c, "r"] for c in conds_b]
        lo = [fb.loc[c, "r"] - fb.loc[c, "ci_low"] for c in conds_b]
        hi = [fb.loc[c, "ci_high"] - fb.loc[c, "r"] for c in conds_b]
        axes[2].plot(xb[:3], eb[:3], color=col, lw=1.4, alpha=0.6, zorder=2)
        axes[2].errorbar(xb, eb, yerr=[lo, hi], fmt="o", color=col, ms=7, elinewidth=2, markeredgecolor="white",
                         markeredgewidth=1.2, zorder=3)
        if mi == 0 and "BToM" in fb.index:
            axes[2].axhline(fb.loc["BToM", "r"], color=INK2, ls="--", lw=1, zorder=1)
            axes[2].text(-0.35, fb.loc["BToM", "r"] + 0.03, "BToM–human", color=INK2, fontsize=8.5, ha="left")

        # (c) final-step belief r(own End-step); Prefix-step at t = T uses the End-step prompt (= reliability)
        fe = f[(f.dimension == "belief") & (f.versus == "End-step")].set_index("condition")
        ee = [fe.loc[c, "r"] for c in CONDS]
        axes[1].plot(x, ee, color=col, lw=1.4, alpha=0.6, zorder=2)
        axes[1].errorbar(x, ee, yerr=[[fe.loc[c, "r"] - fe.loc[c, "ci_low"] for c in CONDS],
                                      [fe.loc[c, "ci_high"] - fe.loc[c, "r"] for c in CONDS]],
                         fmt="o", color=col, ms=7, elinewidth=2, markeredgecolor="white", markeredgewidth=1.2, zorder=3)

        # (d) tokens per judgement (median, p10–p90)
        tj = t[t.metric == "gen_per_judgement"].set_index("condition")
        med = [tj.loc[c, "median"] for c in CONDS]
        axes[3].plot(x, med, color=col, lw=1.4, alpha=0.6, zorder=2)
        axes[3].errorbar(x, med, yerr=[[m - tj.loc[c, "p10"] for m, c in zip(med, CONDS)],
                                       [tj.loc[c, "p90"] - m for m, c in zip(med, CONDS)]],
                         fmt="o", color=col, ms=7, elinewidth=2, markeredgecolor="white", markeredgewidth=1.2, zorder=3)

    ax = axes[0]
    ax.axhline(0, color=INK2, lw=1)
    ax.axhline(1, color=INK2, lw=1, ls="--")
    ax.text(2.45, 0.05, "BToM", color=INK2, fontsize=8.5, ha="right")
    ax.text(2.45, 1.05, "Hindsight", color=INK2, fontsize=8.5, ha="right")
    ax.set_xticks(range(len(CONDS)), [COND_SHORT[c] for c in CONDS])
    ax.set_ylabel("Hindsight index (belief)")
    ax.set_title("(a) Hindsight index", loc="left", fontsize=11)
    ax = axes[2]
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(range(4), [COND_SHORT[c] for c in CONDS + ["End-step"]])
    ax.set_ylabel("r with human (belief, t = T)")
    ax.set_ylim(-0.3, 1.0)
    ax.set_title("(c) Agreement with human ratings", loc="left", fontsize=11)
    ax = axes[1]
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(range(len(CONDS)), [COND_SHORT[c] for c in CONDS])
    ax.set_ylabel("r with own End-step (belief, t = T)")
    ax.set_ylim(-0.5, 1.0)
    ax.set_title("(b) Agreement with own End-step ratings", loc="left", fontsize=11)
    ax.text(2.45, -0.46, "Prefix-step at t = T = End-step prompt\n(run-to-run reliability)", color=INK2,
            fontsize=8, ha="right", va="bottom")
    ax = axes[3]
    ax.set_yscale("log")
    ax.set_xticks(range(len(CONDS)), [COND_SHORT[c] for c in CONDS])
    ax.set_ylabel("Generated tokens per judgement")
    ax.set_title("(d) Compute per judgement (median, 10–90%)", loc="left", fontsize=11)
    for a in axes:
        _style(a)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=len(models), frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ saved: {out}")


# =========================================================================
# S1
# =========================================================================
S1_METRICS = [
    ("hindsight_index", "Hindsight index (all steps)"),
    ("hindsight_index_phase3", "Hindsight index: See G2"),
    ("hindsight_index_phase5", "Hindsight index: Return / Appr G2"),
    ("hindsight_index_phase6", "Hindsight index: Selected"),
    ("rm_belief_mean_btom", "Belief rm_corr with BToM"),
    ("rm_belief_mean_hindsight", "Belief rm_corr with Hindsight"),
    ("rm_desire_mean_btom", "Desire rm_corr with BToM"),
    ("vpa_belief_total", "Belief VPA total adj. R²"),
    ("vpa_belief_unique_btom", "Belief unique R²: BToM"),
    ("vpa_belief_beta_btom", "Belief β: BToM"),
    ("vpa_belief_unique_hindsight", "Belief unique R²: Hindsight"),
    ("vpa_belief_beta_hindsight", "Belief β: Hindsight"),
    ("vpa_belief_unique_truebelief", "Belief unique R²: TrueBelief"),
    ("vpa_belief_unique_motionheuristic", "Belief unique R²: MotionHeuristic"),
    ("abs_err_belief_vs_btom", "Mean abs. error vs BToM (belief)"),
]


def fmt(e, lo, hi, d=2):
    return f"{e:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]" if pd.notna(e) else ""


def table_s1(models, compare_dirs, out_csv):
    rows = []
    for name, d in zip(models, compare_dirs):
        s = pd.read_csv(os.path.join(d, "stepwise_bootstrap_summary.csv"))
        f = pd.read_csv(os.path.join(d, "final_step.csv"))
        for key, label in S1_METRICS:
            r = {"model": name, "metric": label}
            for c in CONDS:
                x = s[(s.condition == c) & (s.metric == key)]
                r[c] = fmt(*x.iloc[0][["estimate", "ci_low", "ci_high"]]) if len(x) else ""
            for c in CONDS[1:]:
                x = s[(s.condition == f"{c} - Every-step") & (s.metric == key)]
                if len(x):
                    e = x.iloc[0]
                    r[f"Δ {c} − Every"] = fmt(e.estimate, e.ci_low, e.ci_high) + (
                        f", p = {e.p_boot:.3g}" if pd.notna(e.p_boot) else "")
                y = s[(s.condition == f"{c} - Every-step") & (s.metric == f"{key} (scenario-level paired sign-flip)")]
                if len(y):
                    r[f"Δ {c} − Every"] += f", perm p = {y.iloc[0].p_perm:.2g}"
            rows.append(r)
        for dim in ("belief", "desire"):
            r = {"model": name, "metric": f"Final step r with human ({dim})"}
            g = f[(f.dimension == dim) & (f.versus == "Human")].set_index("condition")
            for c in CONDS:
                if c in g.index:
                    r[c] = fmt(g.loc[c, "r"], g.loc[c, "ci_low"], g.loc[c, "ci_high"])
            if "End-step" in g.index:
                r["End-step"] = fmt(g.loc["End-step", "r"], g.loc["End-step", "ci_low"], g.loc["End-step", "ci_high"])
            rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(out_csv, index=False, encoding="utf-8-sig")  # Excel에서도 기호가 보이도록
    # Windows 기본 인코딩(cp949)에서 '−', 'Δ', '²'가 깨지지 않도록 UTF-8로 저장
    try:
        md = res.fillna("").to_markdown(index=False)
    except ImportError:  # tabulate 미설치
        md = res.fillna("").to_string(index=False)
    with open(os.path.splitext(out_csv)[0] + ".md", "w", encoding="utf-8") as fh:
        fh.write(md)
    print(f"✅ saved: {out_csv} (+ .md)")
    return res


# =========================================================================
# Control: 정답 열 만들기 + bootstrap
# =========================================================================
C_ITEMS = [  # (열, 표시 이름, 묶음)
    ("q1_first_position", "Start position", "Position"),
    ("q2_last_position", "Final position", "Position"),
    ("q4_final_location", "Final spot", "Position"),
    ("q3_occluded_spot", "Occluded spot at t=1", "Visibility & timing"),
    ("q5a_spot2_visible", "Spot 2 visible at t (a)", "Visibility & timing"),
    ("q5b_spot2_visible", "Spot 2 visible at t (b)", "Visibility & timing"),
    ("q6_first_observed_step", "First step Spot 2 seen", "Visibility & timing"),
    ("q7_shortest_moves_spot1", "Shortest path to Spot 1", "Distance"),
    ("q8_shortest_moves_spot2", "Shortest path to Spot 2", "Distance"),
    ("q11_spot1_change", "Closer/farther: Spot 1", "Distance"),
    ("q11_spot2_change", "Closer/farther: Spot 2", "Distance"),
    ("q9_total_moves", "Total moves", "Movement"),
    ("q10_stay_steps", "Stay steps", "Movement"),
]
EITHER = {"q8_shortest_moves_spot2", "q7_shortest_moves_spot1", "q11_spot2_change", "q11_spot1_change"}


def _q11_unbounded(df):
    """q11을 격자 경계 없는 최단거리로 다시 채점 (q7/q8의 _unbounded와 같은 기준)."""
    from src.dataset import df_btom
    from src.control_items import shortest_moves
    out = {"q11_spot1_change": [], "q11_spot2_change": []}
    for r in df.itertuples():
        g = df_btom[df_btom.scenario_id == r.scenario_id].sort_values("time_step")
        pos = {int(t): (int(x), int(y)) for t, x, y in zip(g.time_step, g.agent_x, g.agent_y)}
        r0 = g.iloc[0]
        wall = (int(r0.wall_start_x), int(r0.wall_start_y), int(r0.wall_width), int(r0.wall_height))
        t = int(r.probe_t11)
        for k, sp in (("q11_spot1_change", (int(r0.goal1_x), int(r0.goal1_y))),
                      ("q11_spot2_change", (int(r0.goal2_x), int(r0.goal2_y)))):
            b = shortest_moves(pos[t - 1], sp, wall, bounded=False)
            a = shortest_moves(pos[t], sp, wall, bounded=False)
            out[k].append("closer" if a < b else "farther" if a > b else "same")
    return out


def control_table(path):
    d = pd.read_csv(os.path.join(path, "control_scored.csv"))
    ub = _q11_unbounded(d)
    for k in ("q11_spot1_change", "q11_spot2_change"):
        d[f"{k}_correct_either"] = (d[f"{k}_pred"].astype(str) == d[f"{k}_gt"].astype(str)) | \
                                   (d[f"{k}_pred"].astype(str) == pd.Series(ub[k], index=d.index))
    main = [f"{c}_correct" for c, _, _ in C_ITEMS]
    d["fact_acc"] = d[main].astype(float).mean(axis=1)
    return d


def boot_item_ci(d, cols, n_boot=2000, seed=0):
    groups = get_group_indices(include_irrational=False)
    group_of = {s: g for g, ids in enumerate(groups) for s in ids}
    d = d[d.scenario_id.isin(group_of)]
    scen = sorted(d.scenario_id.unique())
    per = d.groupby("scenario_id")[cols].mean().reindex(scen).astype(float)
    cnt = d.groupby("scenario_id").size().reindex(scen).to_numpy(float)
    B = boot_indices(scen, group_of, n_boot, seed)
    vals = per.to_numpy()
    est = np.nansum(vals * cnt[:, None], axis=0) / cnt.sum()
    boots = np.array([np.nansum(vals[b] * cnt[b, None], axis=0) / cnt[b].sum() for b in B])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return pd.DataFrame({"est": est, "lo": lo, "hi": hi}, index=cols)


def fig_c1(models, control_dirs, out):
    n = len(C_ITEMS)
    fig, ax = plt.subplots(figsize=(7.6, 0.42 * n + 1.4))
    offs = _offsets(len(models), 0.26)
    for mi, (name, p) in enumerate(zip(models, control_dirs)):
        d = control_table(p)
        main = boot_item_ci(d, [f"{c}_correct" for c, _, _ in C_ITEMS])
        eith = [f"{c}_correct_either" for c, _, _ in C_ITEMS if c in EITHER]
        either = boot_item_ci(d, eith)
        col = MODEL_COLORS[mi]
        for i, (c, label, _) in enumerate(C_ITEMS):
            y = i + offs[mi]
            r = main.loc[f"{c}_correct"]
            ax.plot([r.lo, r.hi], [y, y], color=col, lw=2, solid_capstyle="round")
            ax.plot(r.est, y, "o", color=col, ms=7, markeredgecolor="white", markeredgewidth=1.2,
                    label=name if i == 0 else None, zorder=3)
            if c in EITHER:
                e = either.loc[f"{c}_correct_either"]
                if e.est - r.est > 0.005:
                    ax.plot(e.est, y, "o", ms=7, markerfacecolor="white", markeredgecolor=col, markeredgewidth=1.6,
                            zorder=3, label="Either grid-boundary reading" if (mi == 0 and c == "q8_shortest_moves_spot2") else None)
        print(f"   {name}: fact accuracy {d.fact_acc.mean():.3f}")
    ax.set_yticks(range(n), [lab for _, lab, _ in C_ITEMS])
    ax.invert_yaxis()
    ax.set_xlim(0.6, 1.02)
    ax.set_xlabel("Accuracy (95% scenario bootstrap CI)")
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    prev = None
    for i, (_, _, grp) in enumerate(C_ITEMS):
        if grp != prev:
            ax.text(0.605, i - 0.42, grp.upper(), fontsize=7.5, color=INK2, va="bottom")
            if prev is not None:
                ax.axhline(i - 0.5, color=GRID, lw=0.8)
            prev = grp
    ax.legend(loc="lower left", frameon=False, fontsize=8.5)
    ax.set_title("Spatial comprehension control (full log, 73 scenarios × 5 runs)", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ saved: {out}")


# =========================================================================
# C2
# =========================================================================
def _spearman(x, y):
    from scipy.stats import spearmanr
    ok = ~np.isnan(x) & ~np.isnan(y)
    return spearmanr(x[ok], y[ok])[0]


def fig_c2(models, control_dirs, endstep_pkls, out, n_boot=2000):
    human = pickle.load(open(HUMAN_PKL_PATH, "rb"))
    groups = get_group_indices(include_irrational=False)
    group_of = {s: g for g, ids in enumerate(groups) for s in ids}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), sharex=True)
    rows = []
    for mi, (name, p, pk) in enumerate(zip(models, control_dirs, endstep_pkls)):
        d = control_table(p)
        acc = d.groupby("scenario_id")["fact_acc"].mean()
        e = pickle.load(open(pk, "rb"))
        scen = sorted(s for s in acc.index if s in group_of)
        B = boot_indices(scen, group_of, n_boot, 0)
        col = MODEL_COLORS[mi]
        for ax, (dim, key) in zip(axes, (("desire", "des_inf_mean"), ("belief", "bel_inf_mean_norm"))):
            err = np.abs(e[key] - human[key]).mean(axis=0)  # 시나리오별 옵션 평균 |LLM - 사람|
            x = acc.reindex(scen).to_numpy(float)
            y = err[np.array(scen) - 1]
            rho = _spearman(x, y)
            lo, hi = percentile_ci([_spearman(x[b], y[b]) for b in B])
            jit = (np.random.default_rng(mi).random(len(x)) - 0.5) * 0.006
            ax.scatter(x + jit, y, s=26, color=col, alpha=0.75, edgecolor="white", linewidth=0.6,
                       label=f"{name}: ρ = {rho:.2f} [{lo:.2f}, {hi:.2f}]")
            rows.append(dict(model=name, dim=dim, spearman=rho, ci_low=lo, ci_high=hi, n=len(x)))
    for ax, dim in zip(axes, ("Desire", "Belief")):
        ax.set_xlabel("Fact accuracy per scenario (control)")
        ax.set_ylabel(f"End-step |LLM − human| ({dim.lower()})")
        ax.set_title(dim, loc="left", fontsize=11)
        ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
        _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(os.path.splitext(out)[0] + ".csv", index=False)
    print(f"✅ saved: {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--compare", nargs="+", default=None)
    ap.add_argument("--control", nargs="+", default=None)
    ap.add_argument("--endstep", nargs="+", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--stratified", action="store_true",
                    help="행동 그룹별 층화 bootstrap (기본: 층화 없음 = 논문 방식). 강건성 점검용")
    args = ap.parse_args()
    set_stratified(args.stratified)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.compare:
        fig_r1(args.models, args.compare, os.path.join(args.out_dir, "R1_decomposition.png"))
        table_s1(args.models, args.compare, os.path.join(args.out_dir, "S1_stepwise_table.csv"))
    if args.control:
        fig_c1(args.models, args.control, os.path.join(args.out_dir, "C1_control_accuracy.png"))
        if args.endstep:
            fig_c2(args.models, args.control, args.endstep, os.path.join(args.out_dir, "C2_control_vs_tom_error.png"))


if __name__ == "__main__":
    main()
