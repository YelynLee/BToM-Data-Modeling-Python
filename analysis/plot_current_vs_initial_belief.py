"""
Initial belief(t=1) vs Current belief(t) 응답 비교 (Figure 5a 형식의 phase 정규화 line plot).

--current_belief로 돌린 everystep / prefixstep 결과에서
  belief_L/M/Empty      : "t=1에 학생이 무엇을 믿었나" (initial)
  belief_now_L/M/Empty  : "t에 학생이 무엇을 믿고 있나" (current)
를 같은 x축(phase 정규화) 위에 나란히 그려, 모델이 두 질문을 구분하는지 확인함.

점선 기준선
  - Initial 열: BToM의 스텝별 initial-belief 사후분포 (1..t만 사용)
  - Current 열: 규범적 current belief
        G2를 보기 전(t < t_vis): 새 관찰이 없으므로 에이전트 믿음 = initial belief -> BToM 사후분포
        G2를 본 후 (t >= t_vis): 실제 상태 one-hot (Present 그룹 = L, Absent 그룹 = Empty)

사용 예)
  python analysis/plot_current_vs_initial_belief.py --input results/claude-opus-4-6/vanilla/prefixstep_cur
  python analysis/plot_current_vs_initial_belief.py --input path/to/subject_01.csv --scale raw
"""
import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "src")):
    if p not in sys.path:
        sys.path.append(p)

from src.dataset import df_btom
from src.prepare_everystep import apply_phase_labeling, load_reference_everystep
from analysis.plot_everystep import normalize_scenario_x, get_group_phase_labels, get_phase_index

# Figure 5와 동일한 belief 색
OPTIONS = ["L", "M", "Empty"]
COLORS = {"L": "#457B9D", "M": "#ACCB20", "Empty": "#8D8E86"}
NAMES = {"L": "Truck L", "M": "Truck M", "Empty": "None"}
GROUP_TITLES = {1: "Check-GoBack (Present)", 2: "Check-Stay (Present)", 4: "Check-GoBack (Absent)",
                6: "Check-Partial (Present)", 7: "Check-Partial (Absent)"}
ABSENT_GROUPS = (4, 5, 7)


# -------------------------------------------------------------------------
# 데이터 로드
# -------------------------------------------------------------------------
def load_llm(input_path):
    files = [input_path] if input_path.endswith(".csv") else \
        sorted(glob.glob(os.path.join(input_path, "subject_*.csv")))
    if not files:
        raise FileNotFoundError(f"No subject_*.csv in {input_path}")
    frames = []
    for i, f in enumerate(files, start=1):
        d = pd.read_csv(f)
        d["subject_id"] = i
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    need = [f"belief_{o}" for o in OPTIONS] + [f"belief_now_{o}" for o in OPTIONS]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"current_belief 결과가 아닙니다 (없는 컬럼: {missing})")
    if "error" in df.columns:
        df = df[df["error"].isna()]
    return df, len(files)


def keep_complete(df):
    """모든 time step이 있는 (subject, scenario)만 남김."""
    expected = df_btom.groupby("scenario_id").size()
    got = df.groupby(["subject_id", "scenario_id"]).size().reset_index(name="n")
    got["exp"] = got["scenario_id"].map(expected)
    ok = got[got["n"] == got["exp"]][["subject_id", "scenario_id"]]
    dropped = got[got["n"] != got["exp"]]
    for r in dropped.itertuples():
        print(f"  ⚠️ drop subject {r.subject_id} scenario {r.scenario_id}: {r.n}/{r.exp} steps")
    return df.merge(ok, on=["subject_id", "scenario_id"])


def to_prob(block):
    """1~7 척도 -> (x-1)/sum, 논문 bel_inf_mean_norm과 동일한 처리."""
    shifted = np.maximum(block - 1, 0)
    total = shifted.sum(axis=1).replace(0, 1.0)
    return shifted.div(total, axis=0)


# -------------------------------------------------------------------------
# 전처리: phase labeling + 피험자 평균 + 정규화 + x축 정규화 + 기준선
# -------------------------------------------------------------------------
def prepare(df, scale):
    score_cols = [f"belief_{o}" for o in OPTIONS] + [f"belief_now_{o}" for o in OPTIONS]
    traj_cols = [c for c in df_btom.columns if c not in df.columns or c in ("scenario_id", "time_step")]
    merged = df[["subject_id", "scenario_id", "time_step"] + score_cols].merge(
        df_btom[traj_cols], on=["scenario_id", "time_step"], how="inner")
    merged = apply_phase_labeling(merged)

    # 피험자 평균 (시나리오 x 시점)
    m = merged.groupby(["scenario_id", "group_id", "time_step", "phase"], as_index=False)[score_cols].mean()

    if scale == "prob":
        ini = to_prob(m[[f"belief_{o}" for o in OPTIONS]].set_axis(OPTIONS, axis=1))
        now = to_prob(m[[f"belief_now_{o}" for o in OPTIONS]].set_axis(OPTIONS, axis=1))
        for o in OPTIONS:
            m[f"belief_{o}"], m[f"belief_now_{o}"] = ini[o], now[o]

    # 규범적 기준선 (확률 스케일에서만 의미가 있음)
    if scale == "prob":
        ref = load_reference_everystep("btom")
        ref = ref[["scenario_id", "time_step"] + [f"belief_{o}" for o in OPTIONS]] \
            .rename(columns={f"belief_{o}": f"ref_ini_{o}" for o in OPTIONS})
        m = m.merge(ref, on=["scenario_id", "time_step"], how="left")

        vis = df_btom[df_btom["visible_goal2"] == 1].groupby("scenario_id")["time_step"].min()
        m["t_vis"] = m["scenario_id"].map(vis)
        seen = m["time_step"] >= m["t_vis"]
        truth = np.where(m["group_id"].isin(ABSENT_GROUPS), "Empty", "L")
        for o in OPTIONS:
            m[f"ref_now_{o}"] = np.where(seen, (truth == o).astype(float), m[f"ref_ini_{o}"])

    # pandas 버전에 상관없이 scenario_id 열이 유지되도록 그룹별로 직접 이어붙임
    m = pd.concat([normalize_scenario_x(g) for _, g in m.groupby("scenario_id")], ignore_index=True)
    m["phase_idx"] = [get_phase_index(g, p) for g, p in zip(m["group_id"], m["phase"])]
    return m[m["phase_idx"] != 9]


def phase_means(sub, col):
    """시나리오별 phase 평균 -> 시나리오 간 평균. x = 구간 중앙."""
    per_sc = sub.groupby(["scenario_id", "phase_idx"])[col].mean().reset_index()
    out = per_sc.groupby("phase_idx")[col].mean()
    return out.index.values + 0.5, out.values


# -------------------------------------------------------------------------
# 그리기
# -------------------------------------------------------------------------
def add_end_labels(ax, means, same_as, scale):
    """선 끝 직접 라벨. 겹친 옵션은 'Truck L = M'으로 합치고, 가까운 라벨은 위아래로 벌림."""
    items = []
    for o in OPTIONS:
        if o in same_as:
            continue
        twins = [a for a, b in same_as.items() if b == o]
        name = NAMES[o] if not twins else " = ".join([NAMES[o]] + [NAMES[t].replace("Truck ", "") for t in twins])
        x, y = means[o]
        items.append([y[-1], x[-1], name])
    gap = 0.065 if scale == "prob" else 0.45
    items.sort(key=lambda it: it[0])
    for i in range(1, len(items)):          # 아래에서 위로 최소 간격 확보
        items[i][0] = max(items[i][0], items[i - 1][0] + gap)
    for y, x, name in items:
        ax.annotate(name, (x, y), xytext=(7, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color="#333333")

def plot(m, groups, scale, n_subj, title, out_path):
    nrow = len(groups)
    fig, axes = plt.subplots(nrow, 2, figsize=(12, 3.6 * nrow + 0.8), sharey=True, squeeze=False)
    kinds = [("belief", "ref_ini", "Initial belief (t = 1)"),
             ("belief_now", "ref_now", "Current belief (t)")]
    has_ref = "ref_ini_L" in m.columns

    for r, gid in enumerate(groups):
        gdf = m[m["group_id"] == gid]
        n_sc = gdf["scenario_id"].nunique()
        labels = get_group_phase_labels(gid)
        for c, (prefix, ref_prefix, col_title) in enumerate(kinds):
            ax = axes[r, c]
            means = {o: phase_means(gdf, f"{prefix}_{o}") for o in OPTIONS}
            # 두 옵션의 평균 곡선이 전 구간에서 같으면(예: Absent 그룹의 L = M) 한 선이 다른 선을
            # 완전히 가리므로, 겹친 선은 점선 무늬로 위에 다시 그려 둘 다 보이게 함
            same_as = {}
            for i, a in enumerate(OPTIONS):
                for b in OPTIONS[:i]:
                    if np.allclose(means[a][1], means[b][1], atol=1e-6):
                        same_as[a] = b
            for o in OPTIONS:
                col = f"{prefix}_{o}"
                for _, s in gdf.groupby("scenario_id"):
                    ax.plot(s["x_norm"], s[col], color=COLORS[o], alpha=0.18, lw=1)
                if has_ref:
                    xr, yr = phase_means(gdf, f"{ref_prefix}_{o}")
                    ax.plot(xr, yr, color=COLORS[o], lw=1.6, ls=(0, (3, 2)), alpha=0.9)
            for o in OPTIONS:
                x, y = means[o]
                if o in same_as:   # 겹친 선: 먼저 그린 선 위에 반쯤 끊긴 무늬로
                    ax.plot(x, y, color=COLORS[o], lw=2.6, ls=(4, (4, 4)), zorder=4)
                else:
                    ax.plot(x, y, color=COLORS[o], lw=2.6, marker="o", ms=5,
                            markeredgecolor="white", markeredgewidth=1, zorder=3)
            add_end_labels(ax, means, same_as, scale)

            for k in labels:
                ax.axvline(k, color="#BBBBBB", ls="--", lw=0.6, zorder=0)
            ax.set_xticks([k + 0.5 for k in labels])
            ax.set_xticklabels([labels[k] for k in labels], rotation=35, ha="right", fontsize=8.5)
            ax.set_xlim(0, max(labels) + 1.35)
            ax.grid(axis="y", color="#EEEEEE", lw=0.6)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if scale == "prob":
                ax.set_ylim(-0.03, 1.03)
            else:
                ax.set_ylim(0.8, 7.2)
                ax.set_yticks(range(1, 8))
            if r == 0:
                ax.set_title(col_title, fontsize=12, fontweight="bold", pad=10)
            if c == 0:
                ax.set_ylabel(f"{GROUP_TITLES.get(gid, gid)}\n(n = {n_sc} scenarios)\n\n"
                              + ("Belief probability" if scale == "prob" else "Rating (1–7)"), fontsize=9.5)

    handles = [mlines.Line2D([], [], color=COLORS[o], lw=2.6, marker="o", ms=5, label=NAMES[o]) for o in OPTIONS]
    handles += [mlines.Line2D([], [], color="#777777", lw=1, alpha=0.4, label="Individual scenarios"),
                mlines.Line2D([], [], color="#555555", lw=2.6, label="Phase mean (LLM)")]
    if has_ref:
        handles.append(mlines.Line2D([], [], color="#555555", lw=1.6, ls=(0, (3, 2)),
                                     label="Normative (initial: BToM; current: BToM before See G2, truth after)"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"{title}  ·  {n_subj} subject{'s' if n_subj > 1 else ''} averaged",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0.09 if has_ref else 0.06, 1, 0.97))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="subject_*.csv가 있는 폴더 또는 csv 파일 하나")
    ap.add_argument("--scale", default="prob", choices=["prob", "raw"],
                    help="prob: 논문과 같은 확률 정규화(기준선 포함) / raw: 1~7 원점수")
    ap.add_argument("--groups", default="1,2,4,6,7", help="그릴 group_id (데이터가 있는 것만 그림)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df, n_subj = load_llm(args.input)
    df = keep_complete(df)
    m = prepare(df, args.scale)

    want = [int(g) for g in args.groups.split(",")]
    groups = [g for g in want if g in set(m["group_id"])]
    mode = str(df["mode"].iloc[0]) if "mode" in df.columns else "stepwise"
    model = str(df["model"].iloc[0]) if "model" in df.columns else ""
    title = args.title or f"{model} · {mode} · initial vs current belief"
    out_dir = args.input if os.path.isdir(args.input) else os.path.dirname(os.path.abspath(args.input))
    out = args.out or os.path.join(out_dir, f"current_vs_initial_belief_{args.scale}.png")
    plot(m, groups, args.scale, n_subj, title, out)


if __name__ == "__main__":
    main()
