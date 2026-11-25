"""
Stepwise(everystep / prefixstep) 응답을 phase 정규화 x축 위에 그려 여러 실행을 나란히 비교함 (Figure 5a 형식).

  - 열: 입력 실행 (예: everystep vs prefixstep)
  - 행: 시나리오 그룹 (데이터가 있는 그룹만)
  - 선: phase 평균 굵은 선(트럭별 색)
  - 점(◆): 같은 모델의 End-step 응답, 그룹 평균 ± SD(시나리오 간). plot_everystep.py처럼 마지막 구간에 표시
           (기본: 입력 경로에서 results/<model>/<condition>/model_data.pkl을 찾음. --endstep으로 지정/끄기)
  - 배경(선택): --show_scenarios 시나리오별 얇은 선, --refs btom,hindsight 레퍼런스 모델의 phase 평균

--delta        : LLM − BToM 차이를 그림 (0 = BToM). End-step 점도 End-step BToM과의 차이
--ci           : phase 평균에 시나리오 bootstrap 95% CI 띠 (기본 켬, --no_ci로 끔)
--better       : Better 그룹 열 제목(--labels 값) -> 초록, 나머지 -> 자홍 (논문 Figure 3 색)
--score belief : initial belief(t=1) 응답, 논문과 같은 확률 정규화
--score desire : desire 응답 (1~7)

사용 예)
  python analysis/plot_stepwise_phase.py --score belief \\
      --input results/claude-opus-4-6/vanilla/everystep/everystep_valid_only.csv results/claude-opus-4-6/vanilla/prefixstep \\
      --labels "Every-step" "Prefix-step" --out results/claude-opus-4-6/vanilla/compare/phase_belief.png
  배경까지 그리기: --show_scenarios --refs btom,hindsight
  2x2 (계열 x 성능):
  python analysis/plot_stepwise_phase.py --score belief --delta \\
      --input results/claude-opus-4-6/vanilla/prefixstep results/claude-sonnet-4-6/vanilla/prefixstep \\
              results/gemini-2.5-pro/vanilla/prefixstep results/gemini-2.5-flash/vanilla/prefixstep \\
      --labels "Claude-4.6-Opus" "Claude-4.6-Sonnet" "Gemini-2.5-Pro" "Gemini-2.5-Flash" \\
      --better "Claude-4.6-Opus" "Gemini-2.5-Pro" --out results/stats/prefix_2x2/phase_belief_delta.png
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pickle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "src")):
    if p not in sys.path:
        sys.path.append(p)

from src.dataset import df_btom
from src.config import get_group_indices
from src.prepare_everystep import apply_phase_labeling, load_reference_everystep
from analysis.phase_utils import normalize_scenario_x, get_group_phase_labels, get_phase_index
from analysis.phase_utils import keep_complete, to_prob, phase_means

SCORES = {
    "belief": {"options": ["L", "M", "Empty"], "prefix": "belief",
               "colors": {"L": "#457B9D", "M": "#ACCB20", "Empty": "#8D8E86"},
               "names": {"L": "Truck L", "M": "Truck M", "Empty": "None"},
               "ylabel": "Initial belief (probability)", "ylim": (-0.03, 1.03)},
    "desire": {"options": ["K", "L", "M"], "prefix": "desire",
               "colors": {"K": "#E63946", "L": "#457B9D", "M": "#ACCB20"},
               "names": {"K": "Truck K", "L": "Truck L", "M": "Truck M"},
               "ylabel": "Desire (1–7)", "ylim": (0.8, 7.2)},
}
REF_STYLE = {"btom": ((0, (4, 2)), "BToM"), "hindsight": ((0, (1, 1.5)), "Hindsight"),
             "truebelief": ((0, (6, 2, 1, 2)), "TrueBelief"), "nocost": ((0, (2, 3)), "NoCost")}
GROUP_TITLES = {1: "Check-GoBack (Present)", 2: "Check-Stay (Present)", 4: "Check-GoBack (Absent)",
                6: "Check-Partial (Present)", 7: "Check-Partial (Absent)"}


def load_run(path):
    import glob
    files = [path] if path.endswith(".csv") else sorted(glob.glob(os.path.join(path, "subject_*.csv")))
    if not files and os.path.isdir(path):
        # 설정별 하위 폴더(thinking_off 등)가 하나뿐이면 그 폴더를 씀
        subs = [d for d in sorted(os.listdir(path)) if glob.glob(os.path.join(path, d, "subject_*.csv"))]
        if len(subs) == 1:
            print(f"  {path}: using sub-folder {subs[0]}/")
            path = os.path.join(path, subs[0])
            files = sorted(glob.glob(os.path.join(path, "subject_*.csv")))
        elif len(subs) > 1:
            raise FileNotFoundError(f"{path} has several setting folders {subs}; give one explicitly")
    if not files:
        raise FileNotFoundError(f"No subject_*.csv in {path}")
    frames = []
    for i, f in enumerate(files, start=1):
        d = pd.read_csv(f)
        if "subject_id" not in d.columns or not path.endswith(".csv"):
            d["subject_id"] = i   # everystep_valid_only.csv처럼 subject_id가 있는 단일 csv는 그대로 사용
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    if "error" in df.columns:
        df = df[df["error"].isna()]
    return keep_complete(df), df["subject_id"].nunique()


def prepare_run(df, score, refs):
    cfg = SCORES[score]
    cols = [f"{cfg['prefix']}_{o}" for o in cfg["options"]]
    # 궤적 정보는 항상 df_btom에서 가져옴 (everystep_valid_only.csv처럼 궤적·phase 열이 이미 있는 입력도 지원)
    traj_cols = [c for c in df_btom.columns if c not in cols + ["subject_id"]]
    merged = df[["subject_id", "scenario_id", "time_step"] + cols].merge(
        df_btom[traj_cols], on=["scenario_id", "time_step"], how="inner")
    merged = apply_phase_labeling(merged)
    m = merged.groupby(["scenario_id", "group_id", "time_step", "phase"], as_index=False)[cols].mean()
    if score == "belief":
        prob = to_prob(m[cols].set_axis(cfg["options"], axis=1))
        for o in cfg["options"]:
            m[f"{cfg['prefix']}_{o}"] = prob[o]
    for ref, ref_df in refs.items():
        r = ref_df[["scenario_id", "time_step"] + cols].rename(columns={c: f"{ref}__{c}" for c in cols})
        m = m.merge(r, on=["scenario_id", "time_step"], how="left")
    m = pd.concat([normalize_scenario_x(g) for _, g in m.groupby("scenario_id")], ignore_index=True)
    m["phase_idx"] = [get_phase_index(g, p) for g, p in zip(m["group_id"], m["phase"])]
    return m[m["phase_idx"] != 9]


# =========================================================================
# End-step 점 (같은 모델의 normal 모드 결과)
# =========================================================================
def find_endstep_pkl(path):
    """입력 경로(.../<model>/<condition>/everystep[...] 또는 prefixstep[...])에서 End-step pkl 위치를 추정."""
    parts = os.path.normpath(os.path.abspath(path)).split(os.sep)
    for i, part in enumerate(parts):
        if part.startswith(("everystep", "prefixstep")):
            cand = os.sep.join(parts[:i] + ["model_data.pkl"])
            return cand if os.path.exists(cand) else None
    return None


def btom_endstep(score):
    """BToM의 End-step 값 (3, 78) = 단계별 BToM 레퍼런스의 마지막 스텝."""
    cfg = SCORES[score]
    ref = load_reference_everystep("btom")
    last = ref.loc[ref.groupby("scenario_id")["time_step"].idxmax()].set_index("scenario_id")
    cols = [f"{cfg['prefix']}_{o}" for o in cfg["options"]]
    out = np.full((3, 78), np.nan)
    for sid, row in last[cols].iterrows():
        out[:, int(sid) - 1] = row.values
    return out


def endstep_group_stats(pkl_path, score, baseline=None):
    """{group_id: {option: (mean, sd)}} — 시나리오별 값(논문 pkl의 평균 / 정규화 belief)의 그룹 평균과 시나리오 간 SD.
    baseline (3, 78)을 주면 시나리오별로 뺀 값(LLM − BToM)의 통계."""
    d = pickle.load(open(pkl_path, "rb"))
    arr = d["bel_inf_mean_norm"] if score == "belief" else d["des_inf_mean"]   # (3, 78)
    if baseline is not None:
        arr = arr - baseline
    opts = SCORES[score]["options"]
    out = {}
    for gid, ids in enumerate(get_group_indices(include_irrational=False), start=1):
        v = arr[:, np.array(ids) - 1]
        out[gid] = {o: (float(np.nanmean(v[k])), float(np.nanstd(v[k], ddof=1))) for k, o in enumerate(opts)}
    return out


def phase_ci(sub, col, n_boot=1000, seed=0):
    """phase 평균의 시나리오 bootstrap 95% CI (phase_means와 같은 집계)."""
    per_sc = sub.groupby(["scenario_id", "phase_idx"])[col].mean().unstack("phase_idx")
    vals = per_sc.to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    import warnings
    with warnings.catch_warnings():            # 일부 phase가 없는 bootstrap 표본 → NaN (무시)
        warnings.simplefilter("ignore", RuntimeWarning)
        boots = np.nanmean(vals[idx], axis=1)  # (n_boot, n_phase)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    return per_sc.columns.values + 0.5, lo, hi


def end_labels(ax, ends, gap):
    items = sorted([[y, x, n] for n, (x, y) in ends.items()], key=lambda it: it[0])
    for i in range(1, len(items)):
        items[i][0] = max(items[i][0], items[i - 1][0] + gap)
    for y, x, n in items:
        ax.annotate(n, (x, y), xytext=(7, 0), textcoords="offset points", va="center",
                    fontsize=8.5, color="#333333")


C_BETTER, C_WORSE = "#60A917", "#D80073"   # 논문 Figure 3의 Better / Worse 색


def plot(runs, labels, score, refs, groups, title, out_path, show_scenarios=False, endsteps=None,
         delta=False, ci=True, better=None):
    cfg = dict(SCORES[score])
    if delta:
        cfg["ylabel"] = "LLM − BToM" + (" (probability)" if score == "belief" else " (rating)")
        cfg["ylim"] = (-0.5, 0.5) if score == "belief" else (-3.5, 3.5)
    endsteps = endsteps or [None] * len(runs)
    END_DX = 0.35   # End-step 점을 마지막 phase 평균점 오른쪽에 살짝 띄움
    nrow, ncol = len(groups), len(runs)
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 3.6 * nrow + 0.9), sharey=True, squeeze=False)
    gap = 0.065 if score == "belief" else 0.42

    for r, gid in enumerate(groups):
        for c, ((m, n_subj), label) in enumerate(zip(runs, labels)):
            ax = axes[r, c]
            gdf = m[m["group_id"] == gid]
            labels_x = get_group_phase_labels(gid)
            if gdf.empty:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, color="#999999")
            ends = {}
            est = endsteps[c].get(gid) if endsteps[c] else None
            for o in cfg["options"]:
                col = f"{cfg['prefix']}_{o}"
                color = cfg["colors"][o]
                if gdf.empty:
                    continue
                # (선택) 배경: 시나리오별 얇은 선
                if show_scenarios:
                    for _, s in gdf.groupby("scenario_id"):
                        ax.plot(s["x_norm"], s[col] - (s[f"btom__{col}"] if delta else 0), color=color,
                                alpha=0.16, lw=1)
                # (선택) 배경: 레퍼런스 모델 phase 평균 (--refs)
                for ref in refs:
                    if delta and ref == "btom":
                        continue          # delta에서는 BToM = 0 선
                    x, y = phase_means(gdf, f"{ref}__{col}")
                    if delta:
                        y = y - phase_means(gdf, f"btom__{col}")[1]
                    ax.plot(x, y, color=color, lw=1.6, ls=REF_STYLE[ref][0], alpha=0.9)
                vcol = col
                if delta:
                    vcol = f"delta__{col}"
                    gdf = gdf.assign(**{vcol: gdf[col] - gdf[f"btom__{col}"]})
                if ci:
                    xc, lo, hi = phase_ci(gdf, vcol)
                    ax.fill_between(xc, lo, hi, color=color, alpha=0.16, lw=0, zorder=2)
                x, y = phase_means(gdf, vcol)
                ax.plot(x, y, color=color, lw=2.6, marker="o", ms=5, markeredgecolor="white",
                        markeredgewidth=1, zorder=3)
                x_lab = x[-1]
                # End-step 응답 (그룹 평균 ± 시나리오 간 SD)
                if est is not None and not np.isnan(est[o][0]):
                    x_lab = x[-1] + END_DX
                    ax.errorbar(x_lab, est[o][0], yerr=est[o][1], fmt="D", color=color, ms=7, capsize=3.5,
                                elinewidth=1.6, markeredgecolor="white", markeredgewidth=1, zorder=6)
                ends[cfg["names"][o]] = (x_lab, y[-1])
            # 완전히 겹치는 선의 라벨 합치기 (예: Absent 그룹의 L = M)
            merged_ends, used = {}, set()
            for a, (xa, ya) in ends.items():
                if a in used:
                    continue
                same = [b for b, (xb, yb) in ends.items() if b != a and b not in used and abs(ya - yb) < 1e-9]
                if same:
                    pairs = [a] + same
                    vals = [gdf[f"{cfg['prefix']}_{o}"] - (gdf[f"btom__{cfg['prefix']}_{o}"] if delta else 0)
                            for o in cfg["options"] if cfg["names"][o] in pairs]
                    if all(np.allclose(vals[0].values, v.values) for v in vals[1:]):
                        used.update(same)
                        a = " = ".join([pairs[0]] + [p.replace("Truck ", "") for p in pairs[1:]])
                merged_ends[a] = (xa, ya)
                used.add(a)
            end_labels(ax, merged_ends, gap)

            for k in labels_x:
                ax.axvline(k, color="#BBBBBB", ls="--", lw=0.6, zorder=0)
            if delta:
                ax.axhline(0, color="#333333", lw=1.0, zorder=1)
            ax.set_xticks([k + 0.5 for k in labels_x])
            ax.set_xticklabels([labels_x[k] for k in labels_x], rotation=35, ha="right", fontsize=8.5)
            ax.set_xlim(0, max(labels_x) + (1.45 + END_DX if any(endsteps) else 1.45))
            ax.set_ylim(*cfg["ylim"])
            if score == "desire" and not delta:
                ax.set_yticks(range(1, 8))
            ax.grid(axis="y", color="#EEEEEE", lw=0.6)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if r == 0:
                tcol = "#222222" if better is None else (C_BETTER if label in better else C_WORSE)
                ax.set_title(f"{label}\n({n_subj} run{'s' if n_subj > 1 else ''})", fontsize=11.5,
                             fontweight="bold", pad=10, color=tcol)
            if c == 0:
                n_sc = gdf["scenario_id"].nunique()
                ax.set_ylabel(f"{GROUP_TITLES.get(gid, gid)}\n(n = {n_sc} scenarios)\n\n{cfg['ylabel']}",
                              fontsize=9.5)

    handles = [mlines.Line2D([], [], color=cfg["colors"][o], lw=2.6, marker="o", ms=5, label=cfg["names"][o])
               for o in cfg["options"]]
    handles += [mlines.Line2D([], [], color="#555555", lw=2.6, marker="o", ms=5,
                              label="Phase mean (LLM − BToM)" if delta else "Phase mean (LLM)")]
    if ci:
        import matplotlib.patches as mpatches
        handles.append(mpatches.Patch(color="#555555", alpha=0.2, label="95% CI (scenario bootstrap)"))
    if delta:
        handles.append(mlines.Line2D([], [], color="#333333", lw=1.0, label="BToM (= 0)"))
    if any(endsteps):
        handles.append(mlines.Line2D([], [], color="#555555", marker="D", ls="None", ms=7,
                                     label="End-step" + (" − BToM" if delta else "") + " (mean ± SD across scenarios)"))
    if show_scenarios:
        handles.append(mlines.Line2D([], [], color="#777777", lw=1, alpha=0.4, label="Individual scenarios"))
    handles += [mlines.Line2D([], [], color="#555555", lw=1.6, ls=REF_STYLE[ref][0], label=REF_STYLE[ref][1])
                for ref in refs if not (delta and ref == "btom")]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(title, fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True, help="실행 결과 폴더 또는 csv (여러 개 = 열)")
    ap.add_argument("--labels", nargs="+", default=None, help="각 입력의 열 제목")
    ap.add_argument("--score", default="belief", choices=list(SCORES))
    ap.add_argument("--refs", default="btom", help="배경에 그릴 레퍼런스 모델 phase 평균 (쉼표). 기본: btom. 끄기: --refs none")
    ap.add_argument("--delta", action="store_true", help="LLM − BToM 차이를 그림")
    ap.add_argument("--no_ci", action="store_true", help="phase 평균의 95% CI 띠를 그리지 않음")
    ap.add_argument("--better", nargs="*", default=None, help="Better 그룹 열 제목 (초록), 나머지는 자홍")
    ap.add_argument("--show_scenarios", action="store_true", help="(선택) 시나리오별 얇은 선을 배경에 그림")
    ap.add_argument("--endstep", nargs="+", default=["auto"],
                    help="End-step pkl 경로 (입력 순서대로). auto = 입력 경로에서 추정, none = 표시 안 함")
    ap.add_argument("--groups", default="1,2,4,6,7")
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ref_names = [r for r in args.refs.split(",") if r and r != "none"]
    if args.delta and "btom" not in ref_names:
        ref_names = ["btom"] + ref_names
    refs = {r: load_reference_everystep(r) for r in ref_names}
    labels = args.labels or [os.path.basename(os.path.normpath(p)) for p in args.input]

    runs = []
    for p in args.input:
        df, n = load_run(p)
        runs.append((prepare_run(df, args.score, refs), n))

    # End-step 점
    spec = args.endstep if len(args.endstep) == len(args.input) else args.endstep[:1] * len(args.input)
    endsteps = []
    for p, e in zip(args.input, spec):
        path = find_endstep_pkl(p) if e == "auto" else (None if e == "none" else e)
        base = btom_endstep(args.score) if args.delta else None
        endsteps.append(endstep_group_stats(path, args.score, base) if path else None)
        print(f"  End-step: {path or '(표시 안 함)'}")

    want = [int(g) for g in args.groups.split(",")]
    present = set().union(*[set(m["group_id"]) for m, _ in runs])
    groups = [g for g in want if g in present]
    title = args.title or f"Stepwise {args.score} by phase"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    plot(runs, labels, args.score, ref_names, groups, title, args.out,
         show_scenarios=args.show_scenarios, endsteps=endsteps, delta=args.delta, ci=not args.no_ci,
         better=args.better)


if __name__ == "__main__":
    main()
