"""
단계별 '갱신' 일치도: 연속한 두 스텝 사이 평정 변화(Δ)가 레퍼런스 모델의 변화와 얼마나 맞는지 구간(phase)별로 계산.

  수준(level) 상관은 시나리오 간 차이에 좌우되므로, 같은 궤적 안에서 '언제, 어느 방향으로' 판단을 바꾸는지를 보기 위해
  Δ_t = rating_t − rating_{t−1} (같은 시나리오 안)을 레퍼런스의 Δ와 상관시킴. 스텝 t의 변화는 스텝 t의 phase에 귀속.
  r은 (시나리오 x 스텝 x 옵션)을 펼쳐 계산하고, CI는 시나리오 bootstrap(층화 없음).
  belief는 실행 평균 후 정규화(논문 파이프라인), desire는 원점수. 레퍼런스: BToM, TrueBelief, NoCost, MotionHeuristic, Hindsight
  (desire에서 Hindsight = BToM이므로 desire 히트맵에서는 제외).

입력: --runs "모델|조건|경로" 여러 개 (경로 = subject_*.csv 폴더 또는 everystep_valid_only.csv)
출력: update_agreement.csv, U1_update_vs_btom.png (모델 x 차원, 조건별 선), U2_update_by_reference.png (지정 조건의 레퍼런스 x 구간 히트맵)

사용 예)
  python analysis/update_agreement.py --out_dir results/reviewer --heatmap_condition "Prefix-step" \\
     --runs "Claude Opus 4.6|Every-step|results/claude-opus-4-6/vanilla/everystep/thinking_off/rerun" \\
            "Claude Opus 4.6|Prefix-step|results/claude-opus-4-6/vanilla/prefixstep/thinking_off" \\
            "Gemini 2.5 Pro|Prefix-step|results/gemini-2.5-pro/vanilla/prefixstep" \\
            "GPT-5.4|Prefix-step|results/gpt-5.4/vanilla/prefixstep"
"""
import os
import sys
import argparse
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
                                                 PHASE_NAMES, COLS)
from src.prepare_everystep import load_reference_everystep
from src.config import get_group_indices

REFS = ["btom", "truebelief", "nocost", "motionheuristic", "hindsight"]
REF_LABEL = {"btom": "BToM", "truebelief": "TrueBelief", "nocost": "NoCost", "motionheuristic": "MotionHeuristic",
             "hindsight": "Hindsight"}
DIMS = {"desire": slice(0, 3), "belief": slice(3, 6)}
PHASES = [1, 2, 3, 4, 5, 6]
PH_LABEL = {1: "Approach\nG1", 2: "Pass\nG1", 3: "See\nG2", 4: "Stop", 5: "Return G1 /\nApproach G2", 6: "Selected"}
SIGN_EPS = 1e-3   # 이보다 작은 레퍼런스 변화는 '변화 없음'으로 보고 방향 일치 계산에서 제외
SHORT = {1: "App G1", 2: "Pass G1", 3: "See G2", 4: "Stop", 5: "Ret/App", 6: "Select"}
COND_COL = {"Every-step": "#888780", "Every-step-single": "#D85A30", "Single-trunc": "#BA7517", "Prefix-step": "#1D9E75"}
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False})


def deltas(values, scen):
    d = np.full_like(values, np.nan, dtype=float)
    same = scen[1:] == scen[:-1]
    d[1:][same] = values[1:][same] - values[:-1][same]
    return d


def suff_stats(x, y, scen, mask):
    """시나리오별 상관 충분통계량 (n, Σx, Σy, Σxy, Σx², Σy²)."""
    ok = mask[:, None] & ~np.isnan(x) & ~np.isnan(y)
    out = {}
    for s in np.unique(scen):
        m = (scen == s)[:, None] & ok
        if m.any():
            a, b = x[m], y[m]
            out[s] = np.array([m.sum(), a.sum(), b.sum(), (a * b).sum(), (a * a).sum(), (b * b).sum()])
    return out


def r_from(st):
    n, sx, sy, sxy, sxx, syy = st
    if n < 3:
        return np.nan
    cov = sxy - sx * sy / n
    vx, vy = sxx - sx * sx / n, syy - sy * sy / n
    return np.nan if vx <= 1e-12 or vy <= 1e-12 else cov / np.sqrt(vx * vy)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--heatmap_condition", default="Prefix-step")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)

    keep = sorted(s for g in get_group_indices(include_irrational=False) for s in g)
    rows = row_table(keep)
    scen = rows["scenario_id"].to_numpy()
    phase = rows["phase_idx"].to_numpy()
    ref = ref_arrays(rows, {r: load_reference_everystep(r) for r in REFS})
    dref = {r: deltas(v, scen) for r, v in ref.items()}
    uniq = np.unique(scen)
    boot = [rng.choice(uniq, len(uniq)) for _ in range(a.n_boot)]

    res, mag = [], []
    runs = [r.split("|") for r in a.runs]
    flat = {(r, dim) for r in REFS for dim, sl in DIMS.items()
            if np.nanstd(dref[r][:, sl][phase >= 1]) < 1e-6}          # 스텝 간 변화가 없는 레퍼런스
    for r_, dim in sorted(flat):
        print(f"note: {REF_LABEL[r_]} {dim} has no step-to-step change -> update correlation undefined")

    def mag_row(model, cond, dim, src, d):
        for ph in PHASES + ["all"]:
            mask = (phase >= 1) if ph == "all" else (phase == ph)
            v = np.abs(d[mask]).mean(axis=1)
            sc = scen[mask]
            per = pd.Series(v).groupby(sc).mean()
            bs = [per.reindex(b).mean() for b in boot]
            mag.append(dict(model=model, condition=cond, dimension=dim, source=src, phase_idx=ph,
                            mean_abs_delta=per.mean(), ci_low=np.nanpercentile(bs, 2.5), ci_high=np.nanpercentile(bs, 97.5)))

    for model, cond, path in runs:
        df = load_rows(path)
        arr, _ = subject_array(df[df.scenario_id.isin(keep)], rows)
        dl = deltas(aggregate(arr), scen)
        for dim, sl in DIMS.items():
            mag_row(model, cond, dim, "LLM", dl[:, sl])
            for refname in REFS:
                if (dim == "desire" and refname == "hindsight") or (refname, dim) in flat:
                    continue
                for ph in PHASES + ["all"]:
                    mask = (phase >= 1) if ph == "all" else (phase == ph)
                    st = suff_stats(dl[:, sl], dref[refname][:, sl], scen, mask)
                    if not st:
                        continue
                    est = r_from(sum(st.values()))
                    bs = [r_from(sum(st[s] for s in b if s in st)) for b in boot]
                    lo, hi = np.nanpercentile(bs, [2.5, 97.5]) if np.isfinite(est) else (np.nan, np.nan)
                    n_steps = int(sum(v[0] for v in st.values()) // 3)
                    # 방향 일치: 레퍼런스가 조금이라도 바뀐 (옵션 x 스텝)에서 LLM 변화의 부호가 같은 비율 / 반대인 비율
                    x_, y_ = dl[:, sl][mask], dref[refname][:, sl][mask]
                    ok = np.isfinite(x_) & np.isfinite(y_) & (np.abs(y_) > SIGN_EPS)
                    # |Δ레퍼런스| 가중: 큰 갱신일수록 크게 반영 (임계값 없이 미세한 변화의 부호 잡음을 줄임)
                    w = np.abs(y_[ok])
                    same = float(np.sum(w * (np.sign(x_[ok]) == np.sign(y_[ok]))) / w.sum()) if ok.any() else np.nan
                    opp = float(np.sum(w * (np.sign(x_[ok]) == -np.sign(y_[ok]))) / w.sum()) if ok.any() else np.nan
                    res.append(dict(model=model, condition=cond, dimension=dim, reference=REF_LABEL[refname],
                                    phase=PHASE_NAMES.get(ph, "all") if ph != "all" else "all", phase_idx=ph,
                                    r=est, ci_low=lo, ci_high=hi, n_steps=n_steps, n_scenarios=len(st),
                                    sign_same=same, sign_opposite=opp, n_sign=int(ok.sum())))
        print(f"done: {model} / {cond}")
    for dim, sl in DIMS.items():
        for refname in ["btom"]:
            mag_row("reference", REF_LABEL[refname], dim, REF_LABEL[refname], dref[refname][:, sl])
    R = pd.DataFrame(res)
    MG = pd.DataFrame(mag)
    R.to_csv(os.path.join(a.out_dir, "update_agreement.csv"), index=False)
    MG.to_csv(os.path.join(a.out_dir, "update_magnitude.csv"), index=False)

    models = list(dict.fromkeys(m for m, _, _ in runs))
    xt = PHASES + [7.4]
    xl = [SHORT[p] for p in PHASES] + ["all"]

    def style(ax):
        if ax.get_yscale() != "log":
            ax.axhline(0, color="k", lw=0.6)
        ax.axvline(6.7, color="#bbb", lw=0.6, ls=":")
        ax.set_xticks(xt, xl, fontsize=8)

    # ---------- U1 ----------
    fig, axes = plt.subplots(3, len(models), figsize=(4.3 * len(models), 8.6), squeeze=False, sharex=True)
    handles = {}
    for j, m in enumerate(models):
        for i, dim in enumerate(DIMS):
            ax = axes[i, j]
            sub = R[(R.model == m) & (R.dimension == dim) & (R.reference == "BToM")]
            conds = [c for c in COND_COL if c in set(sub.condition)]
            for k, c in enumerate(conds):
                g = sub[sub.condition == c].copy()
                g["x"] = [7.4 if p == "all" else int(p) for p in g.phase_idx]
                g = g.sort_values("x")
                off = (k - (len(conds) - 1) / 2) * 0.1
                ph_ = g[g.x <= 6]
                h = ax.errorbar(ph_.x + off, ph_.r, yerr=[ph_.r - ph_.ci_low, ph_.ci_high - ph_.r], color=COND_COL[c],
                                marker="o", ms=4.5, lw=1.4, elinewidth=0.9)
                al = g[g.x > 6]
                ax.errorbar(al.x + off, al.r, yerr=[al.r - al.ci_low, al.ci_high - al.r], color=COND_COL[c],
                            marker="D", ms=5, elinewidth=0.9, ls="none")
                if len(al) and np.isfinite(al.sign_same.iloc[0]):
                    ax.text(7.95, 0.92 - 0.13 * k, f"{al.sign_same.iloc[0]*100:.0f}/{al.sign_opposite.iloc[0]*100:.0f}", color=COND_COL[c], fontsize=7.2,
                            ha="left", va="center", fontweight="bold")
                handles[c] = h
            style(ax)
            ax.set_ylim(-0.75, 1.0)
            ax.set_xlim(0.5, 8.6)
            ax.text(7.95, 1.04, "same/opp\n(%)", fontsize=6.5, color="#555", ha="left", va="bottom")
            if j == 0:
                ax.set_ylabel(f"{dim.capitalize()} updates\nr(Δ LLM, Δ BToM)")
            else:
                ax.set_yticklabels([])
            if i == 0:
                ax.set_title(m, fontsize=11)
        ax = axes[2, j]
        sub = MG[(MG.model == m) & (MG.dimension == "belief")]
        conds = [c for c in COND_COL if c in set(sub.condition)]
        for k, c in enumerate(conds):
            g = sub[sub.condition == c].copy()
            g["x"] = [7.4 if p == "all" else int(p) for p in g.phase_idx]
            off = (k - (len(conds) - 1) / 2) * 0.1
            ph_ = g[g.x <= 6].sort_values("x")
            ax.plot(ph_.x + off, ph_.mean_abs_delta, color=COND_COL[c], marker="o", ms=4, lw=1.3)
            al = g[g.x > 6]
            ax.plot(al.x + off, al.mean_abs_delta, color=COND_COL[c], marker="D", ms=5, ls="none")
        b = MG[(MG.model == "reference") & (MG.dimension == "belief")].copy()
        b["x"] = [7.4 if p == "all" else int(p) for p in b.phase_idx]
        bp = b[b.x <= 6].sort_values("x")
        hb, = ax.plot(bp.x, bp.mean_abs_delta, color="k", ls="--", lw=1.2, marker="s", ms=3.5)
        ax.plot(b[b.x > 6].x, b[b.x > 6].mean_abs_delta, color="k", marker="s", ms=4, ls="none")
        handles["BToM (update size)"] = hb
        ax.set_yscale("log")
        style(ax)
        ax.set_yscale("log")
        ax.set_ylim(0.0015, 0.6)
        ax.axhline(1, color="none")
        if j == 0:
            ax.set_ylabel("Belief update size\nmean |Δ| (log scale)")
        else:
            ax.set_yticklabels([])
        ax.set_xlim(0.5, 8.6)
    fig.legend(handles.values(), handles.keys(), loc="upper center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, 1.035), fontsize=9)
    fig.suptitle("Step-to-step updates by trajectory phase (95% scenario-bootstrap CI; ◆ = all steps; "
                 "same/opp = % of BToM change (weighted by |Δ BToM|) matched / reversed in sign; remainder = no LLM change)",
                 fontsize=11, y=1.07)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, "U1_update_vs_btom.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---------- U2 ----------
    H = R[R.condition == a.heatmap_condition]
    hm_models = [m for m in models if m in set(H.model)]
    fig, axes = plt.subplots(2, len(hm_models), figsize=(4.6 * len(hm_models), 6.6), squeeze=False,
                             gridspec_kw={"height_ratios": [4, 5], "hspace": 0.35, "wspace": 0.08})
    cols = PHASES + ["all"]
    for j, m in enumerate(hm_models):
        for i, dim in enumerate(DIMS):
            ax = axes[i, j]
            refs = [r for r in REFS if not (dim == "desire" and r == "hindsight")]
            M = np.full((len(refs), len(cols)), np.nan)
            for a_, rf in enumerate(refs):
                for b_, ph in enumerate(cols):
                    v = H[(H.model == m) & (H.dimension == dim) & (H.reference == REF_LABEL[rf])
                          & (H.phase_idx.astype(str) == str(ph))]
                    if len(v):
                        M[a_, b_] = v.r.iloc[0]
            im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
            for a_, rf in enumerate(refs):
                if (rf, dim) in flat:
                    ax.add_patch(plt.Rectangle((-0.5, a_ - 0.5), len(cols), 1, color="#eeeeec", zorder=2))
                    ax.text((len(cols) - 1) / 2, a_, "no step-to-step change (flat)", ha="center", va="center",
                            fontsize=7.5, color="#777", zorder=3)
            for b_ in range(len(cols)):
                colv = M[:, b_]
                best = int(np.nanargmax(colv)) if np.isfinite(colv).any() else None
                for a_ in range(len(refs)):
                    if np.isfinite(M[a_, b_]):
                        ax.text(b_, a_, f"{M[a_, b_]:.2f}", ha="center", va="center", fontsize=7.2,
                                color="white" if abs(M[a_, b_]) > 0.55 else "k",
                                fontweight="bold" if a_ == best else None)
                if best is not None:
                    ax.add_patch(plt.Rectangle((b_ - 0.5, best - 0.5), 1, 1, fill=False, ec="k", lw=1.4, zorder=4))
            ax.axvline(len(cols) - 1.5, color="white", lw=3)
            ax.set_xticks(range(len(cols)), [SHORT[p] for p in PHASES] + ["all"], fontsize=7.5)
            ax.set_yticks(range(len(refs)), [REF_LABEL[r] for r in refs] if j == 0 else [], fontsize=8)
            ax.tick_params(length=0)
            ax.spines[:].set_visible(False)
            if i == 0:
                ax.set_title(m, fontsize=11)
            if j == 0:
                ax.set_ylabel(dim.capitalize(), fontsize=10, labelpad=8)
    cb = fig.colorbar(im, ax=axes, shrink=0.55, pad=0.015)
    cb.set_label("r(Δ LLM, Δ reference)")
    fig.suptitle(f"{a.heatmap_condition}: whose step-to-step updates does each LLM follow?  (box = highest r in each phase)",
                 fontsize=11, y=0.98)
    fig.savefig(os.path.join(a.out_dir, "U2_update_by_reference.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("✅ saved:", a.out_dir)


if __name__ == "__main__":
    main()
