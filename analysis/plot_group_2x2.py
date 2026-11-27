"""
대표 그룹(기본 Check-GoBack Present) 하나의 단계별 궤적: BToM / Claude 계열 / Gemini 계열 (Figure 5a 대체안).

레이아웃
  열: Desire 궤적 | Belief 궤적 | (오른쪽 작은 패널) 스텝 간 갱신(update)의 BToM 일치도
  행: BToM | Claude (Better 실선 vs Worse 점선) | Gemini (Better 실선 vs Worse 점선)
궤적 패널
  - phase 평균 (그 그룹 시나리오 평균; belief는 실행 평균 후 확률 정규화)
  - 핵심 옵션(key option): 마지막 3개 phase에서 |Better − Worse|가 가장 큰 옵션 (두 계열 평균, --key로 지정 가능)
    -> 진하게 그리고, Better와 Worse 사이를 옵션 색으로 채움 + 회색 점선 = 그 옵션의 BToM. 나머지 옵션은 옅게
  - 분기 시점(diverges, half-max 기준): 핵심 옵션의 (Better − Worse) 차이가 그 phase부터 마지막 phase까지
    (a) 최대 차이와 같은 부호, (b) 최대 차이의 절반 이상, (c) 시나리오 bootstrap 95% CI가 0을 포함하지 않음
    을 모두 만족하는 첫 phase (Start 제외). 시나리오 간 변동이 작아 CI만으로는 아주 작은 차이도 유의해지므로
    효과 크기 조건(a, b)을 함께 씀. (기록용, _divergence.csv의 onset 열)
  - 그림의 굵은 x축 레이블 = 주요 분기 구간 (scenario_max_widening): 시나리오마다 스텝 단위 갭 변화량이
    처음으로 최대가 되는 스텝을 찾고, 시나리오의 절반 이상이 같은 곳(동률 없음)일 때 그 구간을 굵게.
    직전 스텝과 다른 phase면 경계이므로 두 phase 모두 굵게
  - 기록용 (_divergence_summary.csv): half-max onset, 평균 곡선의 최대 증가 전환, relative closure
    R = (Better − Worse)/(BToM − Worse), BToM 최대 갱신 스텝과의 일치
update 패널
  - Δ_t = 판단_t − 판단_{t−1} (같은 시나리오 안)과 BToM의 Δ의 Pearson r (옵션 3개 pooled), 시나리오 bootstrap 95% CI
  - prefix_2x2.py의 update_r과 같은 정의 (여기서는 이 그룹 시나리오만)

사용 예)
  python analysis/plot_group_2x2.py --group 1 --out results/stats/prefix_2x2/group1_trajectory_2x2.png \\
     --runs "Opus|Claude|better|results/claude-opus-4-6/vanilla/prefixstep" \\
            "Sonnet|Claude|worse|results/claude-sonnet-4-6/vanilla/prefixstep" \\
            "Gemini-Pro|Gemini|better|results/gemini-2.5-pro/vanilla/prefixstep" \\
            "Gemini-Flash|Gemini|worse|results/gemini-2.5-flash/vanilla/prefixstep"
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
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "src"), os.path.join(ROOT, "analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)
warnings.filterwarnings("ignore")

from analysis.plot_stepwise_phase import load_run, prepare_run, SCORES, GROUP_TITLES
from analysis.phase_utils import get_group_phase_labels
from src.prepare_everystep import load_reference_everystep

C_BETTER, C_WORSE = "#60A917", "#D80073"
DIMS = ("desire", "belief")


def per_scenario_phase(m, col):
    """시나리오 x phase 평균 표 (phase_means와 같은 1단계 집계)."""
    return m.groupby(["scenario_id", "phase_idx"])[col].mean().unstack("phase_idx")


def diff_ci(mb, mw, col, n_boot, rng):
    """phase별 (Better − Worse) 평균과 시나리오 bootstrap 95% CI."""
    d = per_scenario_phase(mb, col) - per_scenario_phase(mw, col)
    v = d.to_numpy()
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    boots = np.nanmean(v[idx], axis=1)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    return pd.DataFrame({"diff": np.nanmean(v, axis=0), "lo": lo, "hi": hi}, index=d.columns)


def sustained_onset(dci, frac=0.5):
    """최대 차이와 같은 부호 + 최대 차이의 frac 이상 + CI가 0 제외, 이 상태가 끝까지 이어지는 첫 phase (Start 제외)."""
    d = dci["diff"].to_numpy()
    peak = d[np.nanargmax(np.abs(d))]
    sig = (((dci["lo"] > 0) | (dci["hi"] < 0)).to_numpy() & (np.sign(d) == np.sign(peak))
           & (np.abs(d) >= frac * abs(peak)))
    phases = dci.index.values
    onset = None
    for i in range(len(phases) - 1, -1, -1):
        if sig[i] and phases[i] > 0:
            onset = phases[i]
        else:
            break
    return onset


def jump_phases(mb, mw, col, peak_sign):
    """Better − Worse 차이가 가장 크게 벌어지는 지점의 phase(들).
    phase를 스텝 위치(x_norm)로 앞/뒤 절반으로 나눈 소구간(스텝이 하나뿐인 phase는 하나)마다 시나리오 평균 차이를 구하고,
    연속한 소구간 사이에서 (최대 차이 부호 기준) 차이가 가장 많이 커지는 곳을 찾음.
    두 소구간이 다른 phase면 두 phase 모두(경계에서 분기), 같은 phase 안이면 그 phase만."""
    def sub(m):
        m = m.assign(half=((m["x_norm"] - m["phase_idx"]) >= 0.5).astype(int))
        single = m.groupby("phase_idx")["time_step"].transform("nunique") <= 1
        m.loc[single, "half"] = 0
        return m.groupby(["scenario_id", "phase_idx", "half"])[col].mean()
    g = (sub(mb) - sub(mw)).groupby(level=["phase_idx", "half"]).mean().sort_index() * peak_sign
    keys = list(g.index)
    jumps = np.diff(g.values)
    if len(jumps) == 0:
        return [], g
    j = int(np.nanargmax(jumps))
    (pa, _), (pb, _) = keys[j], keys[j + 1]
    return ([pa, pb] if pa != pb else [pb]), g


def scenario_max_widening(mb, mw, col, sign, labels_x):
    """교수님 안 (2): 시나리오마다 스텝 단위 갭 g_t = sign·(Better_t − Worse_t)의 변화량 Δg_t = g_t − g_{t−1}이
    처음으로 최대가 되는 스텝 t* (동률이면 앞쪽). t*−1과 t*가 다른 phase면 '경계'(두 phase), 같으면 그 phase.
    같은 시나리오에서 BToM 자체가 가장 크게 바뀌는 스텝(|ΔBToM| 최대)도 기록해 일치 여부를 봄.
    굵게 표시: 최빈 위치가 시나리오의 절반 이상이고 동률이 없을 때만 (localized). 아니면 표시하지 않음."""
    a = mb.set_index(["scenario_id", "time_step"])
    b = mw.set_index(["scenario_id", "time_step"])
    rec = []
    for sid in sorted(a.index.get_level_values(0).unique()):
        A, B = a.loc[sid].sort_index(), b.loc[sid].sort_index()
        steps = A.index.intersection(B.index)
        A, B = A.loc[steps], B.loc[steps]
        g = sign * (A[col] - B[col]).to_numpy()
        dg = np.diff(g)
        if len(dg) == 0:
            continue
        j = int(np.argmax(dg))                     # 첫 번째 최대
        ph = A["phase_idx"].to_numpy()
        p0, p1 = int(ph[j]), int(ph[j + 1])
        cat = labels_x[p1] if p0 == p1 else f"{labels_x[p0]} → {labels_x[p1]}"
        db = np.abs(np.diff(A[f"btom__{col}"].to_numpy()))
        jb = int(np.argmax(db))
        rec.append(dict(scenario_id=sid, step=int(steps[j + 1]), widening=dg[j], category=cat,
                        phases=(p0, p1) if p0 != p1 else (p1,),
                        btom_step=int(steps[jb + 1]), same_as_btom=abs(jb - j) <= 0,
                        within1_of_btom=abs(jb - j) <= 1))
    df = pd.DataFrame(rec)
    counts = df["category"].value_counts()
    mode = counts.index[0]
    tie = len(counts) > 1 and counts.iloc[1] == counts.iloc[0]
    localized = (counts.iloc[0] / len(df) >= 0.5) and not tie     # 시나리오 절반 이상이 같은 곳 + 동률 없음
    bold = list(df.loc[df["category"] == mode, "phases"].iloc[0]) if localized else []
    summ = dict(mode=mode + (" (tie)" if tie else ""), mode_share=f"{counts.iloc[0]}/{len(df)}", n=len(df),
                localized=localized,
                btom_match=f"{int(df['same_as_btom'].sum())}/{len(df)}",
                btom_match_pm1=f"{int(df['within1_of_btom'].sum())}/{len(df)}",
                all_categories="; ".join(f"{k} {v}" for k, v in counts.items()))
    return bold, summ, df


def relative_closure(mb, mw, col, sign, floor=0.1):
    """교수님 안 (1): phase별 R = (Better − Worse) / (BToM − Worse) (시나리오 평균의 비).
    |BToM − Worse|가 그 옵션 BToM 범위의 floor배보다 작으면 분모가 불안정하므로 NaN."""
    B = per_scenario_phase(mb, col).mean(axis=0)
    W = per_scenario_phase(mw, col).mean(axis=0)
    T = per_scenario_phase(mb, f"btom__{col}").mean(axis=0)
    rng_ = T.max() - T.min()
    den = T - W
    R = (B - W) / den
    R[np.abs(den) < floor * max(rng_, 1e-9)] = np.nan
    return R


def update_r(m, score, n_boot, rng):
    """Δ(LLM) vs Δ(BToM) Pearson r, 옵션 pooled, 시나리오 bootstrap 95% CI."""
    cfg = SCORES[score]
    cols = [f"{cfg['prefix']}_{o}" for o in cfg["options"]]
    m = m.sort_values(["scenario_id", "time_step"])
    per = []
    for sid, g in m.groupby("scenario_id"):
        dy = np.diff(g[cols].to_numpy(), axis=0).ravel()
        db = np.diff(g[[f"btom__{c}" for c in cols]].to_numpy(), axis=0).ravel()
        per.append((dy, db))

    def r_of(sel):
        y = np.concatenate([per[i][0] for i in sel])
        b = np.concatenate([per[i][1] for i in sel])
        return np.corrcoef(y, b)[0, 1]
    est = r_of(range(len(per)))
    boots = [r_of(rng.integers(0, len(per), len(per))) for _ in range(n_boot)]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return est, lo, hi


def place_labels(ax, items, x, gap):
    """items: [(y, text, kwargs)] -> 세로로 최소 gap만큼 띄워 x 위치에 라벨."""
    items = sorted(items, key=lambda t: t[0])
    ys = [t[0] for t in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    for (y0, text, kw), y in zip(items, ys):
        ax.text(x, y, text, va="center", fontsize=8.5, **kw)


def phase_mean(m, col):
    out = per_scenario_phase(m, col).mean(axis=0)
    return out.index.values + 0.5, out.values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help='"label|family|better/worse|path"')
    ap.add_argument("--group", type=int, default=1)
    ap.add_argument("--key", default=None, help="핵심 옵션, 예: desire:M,belief:M (기본: 자동)")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    refs = {"btom": load_reference_everystep("btom")}
    fams = {}
    data = {}
    for spec in args.runs:
        lab, fam, tier, path = spec.split("|")
        fams.setdefault(fam, {})[tier] = lab
        df, n = load_run(path)
        data[lab] = {score: prepare_run(df, score, refs) for score in DIMS}
        data[lab]["n"] = n
        for score in DIMS:
            data[lab][score] = data[lab][score][data[lab][score]["group_id"] == args.group]

    gid = args.group
    key = {}
    if args.key:
        key = dict(kv.split(":") for kv in args.key.split(","))
    for score in DIMS:
        if score in key:
            continue
        cfg = SCORES[score]
        sc = {}
        for o in cfg["options"]:
            col = f"{cfg['prefix']}_{o}"
            vals = []
            for f in fams:
                d = (per_scenario_phase(data[fams[f]["better"]][score], col)
                     - per_scenario_phase(data[fams[f]["worse"]][score], col)).mean(axis=0)
                vals.append(np.abs(d.values[-3:]).mean())
            sc[o] = np.mean(vals)
        key[score] = max(sc, key=sc.get)
    print(f"  key options: {key}")
    onsets = {}
    labels_x = get_group_phase_labels(gid)
    any_lab = next(iter(data))
    n_sc = data[any_lab]["desire"]["scenario_id"].nunique()
    fam_names = list(fams)

    fig = plt.figure(figsize=(14.5, 3.3 * (1 + len(fam_names)) + 0.9))
    gs = GridSpec(1 + len(fam_names), 3, figure=fig, width_ratios=[1, 1, 0.42], wspace=0.22, hspace=0.5)
    rows = ["BToM"] + fam_names
    upd = {}
    for r, rname in enumerate(rows):
        for c, score in enumerate(DIMS):
            cfg = SCORES[score]
            ax = fig.add_subplot(gs[r, c])
            gap = 0.06 if score == "belief" else 0.4
            if r == 0:
                m = data[any_lab][score]
                items = []
                for o in cfg["options"]:
                    col = f"btom__{cfg['prefix']}_{o}"
                    x, y = phase_mean(m, col)
                    ax.plot(x, y, color=cfg["colors"][o], lw=2.6, marker="o", ms=5, mec="white", mew=1, zorder=3)
                    items.append((y[-1], cfg["names"][o], {"color": "#333333"}))
                place_labels(ax, items, x[-1] + 0.15, gap)
            else:
                lb, lw_ = fams[rname]["better"], fams[rname]["worse"]
                mb, mw = data[lb][score], data[lw_][score]
                kcol = f"{cfg['prefix']}_{key[score]}"
                dci = diff_ci(mb, mw, kcol, args.n_boot, rng)
                onset = sustained_onset(dci)
                fin = dci["diff"].iloc[-1]
                sign = np.sign(fin) if fin != 0 else 1.0
                bold, summ, sdf = scenario_max_widening(mb, mw, kcol, sign, labels_x)
                Rcl = relative_closure(mb, mw, kcol, sign)
                # 비교용: 평균 곡선(phase 평균)에서 갭 증가가 가장 큰 phase 전환
                gm = sign * dci["diff"]
                jm = int(np.nanargmax(np.diff(gm.values)))
                summ.update(halfmax_onset=labels_x.get(int(onset)) if onset is not None else None,
                            meancurve_max_widening=f"{labels_x[int(gm.index[jm])]} → {labels_x[int(gm.index[jm + 1])]}",
                            closure_R=" ".join(f"{labels_x[int(k)]}:{v:.2f}" for k, v in Rcl.items() if not np.isnan(v)))
                onsets[(rname, score)] = (onset, dci, bold, summ, sdf, Rcl)
                for o in cfg["options"]:
                    col = f"{cfg['prefix']}_{o}"
                    color = cfg["colors"][o]
                    is_key = o == key[score]
                    a = 1.0 if is_key else 0.32
                    xb, yb = phase_mean(mb, col)
                    xw, yw = phase_mean(mw, col)
                    if is_key:
                        xr, yr = phase_mean(mb, f"btom__{col}")
                        ax.plot(xr, yr, color="#777777", lw=1.4, ls=(0, (3, 2)), zorder=2)
                        ax.fill_between(xb, yb, yw, color=color, alpha=0.22, lw=0, zorder=1)
                    ax.plot(xb, yb, color=color, lw=2.8 if is_key else 1.8, marker="o", ms=5.5 if is_key else 4,
                            mec="white", mew=1, alpha=a, zorder=4 if is_key else 3)
                    ax.plot(xw, yw, color=color, lw=2.0 if is_key else 1.4, ls=(0, (4, 1.6)), marker="o",
                            ms=5 if is_key else 3.5, mfc="white", mec=color, mew=1.2, alpha=a, zorder=4 if is_key else 3)
                    if is_key:
                        place_labels(ax, [(yb[-1], f"{cfg['names'][o]} ({lb.replace('Gemini-', '')})", {"color": "#222222", "fontweight": "bold"}),
                                          (yw[-1], f"{cfg['names'][o]} ({lw_.replace('Gemini-', '')})", {"color": "#222222"})],
                                     xb[-1] + 0.15, gap)
                upd[(rname, score)] = {t: update_r(data[fams[rname][t]][score], score, args.n_boot, rng)
                                       for t in ("better", "worse")}
            for k in labels_x:
                ax.axvline(k, color="#CCCCCC", ls="--", lw=0.6, zorder=0)
            ax.set_xticks([k + 0.5 for k in labels_x])
            ax.set_xlim(0, max(labels_x) + 2.3)
            ax.set_ylim(*cfg["ylim"])
            if score == "desire":
                ax.set_yticks(range(1, 8))
            ax.set_xticklabels([labels_x[k] for k in labels_x], rotation=30, ha="right", fontsize=9)
            if r > 0:
                for k, t in zip(labels_x, ax.get_xticklabels()):
                    if k in onsets[(rname, score)][2]:
                        t.set_fontweight("bold")
                        t.set_color("#000000")
                        t.set_fontsize(9.8)
            ax.set_ylabel({"desire": "Desire (1–7)", "belief": "Belief (probability)"}[score], fontsize=9.5)
            ax.grid(axis="y", color="#F0F0F0", lw=0.6)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if r == 0:
                ax.set_title(score.capitalize(), fontsize=14, fontweight="bold", pad=16)
            if c == 0:
                sub = "" if r == 0 else f"\n{fams[rname]['better']} vs {fams[rname]['worse']}"
                ax.annotate(rname + sub, xy=(0, 0.5), xycoords="axes fraction", xytext=(-48, 0),
                            textcoords="offset points", ha="right", va="center", fontsize=12.5,
                            fontweight="bold", linespacing=1.4)

        # 오른쪽 update 패널
        ax = fig.add_subplot(gs[r, 2])
        if r == 0:
            ax.axis("off")
            h = [mlines.Line2D([], [], color="#444444", lw=2.6, marker="o", ms=5, label="Better (solid)"),
                 mlines.Line2D([], [], color="#444444", lw=1.9, ls=(0, (4, 1.6)), marker="o", ms=4.5, mfc="white",
                               label="Worse (dashed)"),
                 mlines.Line2D([], [], color="#777777", lw=1.4, ls=(0, (3, 2)), label="BToM (key option)"),
                 mpatches.Patch(color="#ACCB20", alpha=0.3, label="Better − Worse gap (key option)")]
            ax.legend(handles=h, loc="center left", frameon=False, fontsize=9, borderaxespad=0,
                      bbox_to_anchor=(0, 0.6))
            ax.text(0, 0.12, "Bold phase label = where the Better − Worse\ngap first widens the most (step-level, modal\n"
                    "in ≥ half of scenarios; both phases if the\njump spans their boundary). No bold = no\nconsistent divergence point", transform=ax.transAxes,
                    fontsize=8.2, va="top", color="#333333")
            continue
        for j, score in enumerate(DIMS):
            for tier, dx in (("better", -0.13), ("worse", 0.13)):
                est, lo, hi = upd[(rname, score)][tier]
                col = C_BETTER if tier == "better" else C_WORSE
                ax.errorbar(j + dx, est, yerr=[[est - lo], [hi - est]], fmt="o", color=col, ms=8, capsize=3,
                            lw=1.8, mec="white", mew=1, zorder=3)
                ax.text(j + dx + (0.08 if tier == "worse" else -0.08), est, f"{est:.2f}", fontsize=8,
                        ha="left" if tier == "worse" else "right", va="center", color="#333333")
        ax.axhline(0, color="#333333", lw=0.8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Desire", "Belief"], fontsize=9.5)
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(-0.25, 1.0)
        ax.set_ylabel("Update r vs BToM", fontsize=9)
        ax.tick_params(axis="y", labelsize=8.5)
        ax.grid(axis="y", color="#F0F0F0", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        if r == 1:
            ax.set_title("Step-to-step update\n(Δ LLM vs Δ BToM)", fontsize=10.5, fontweight="bold", pad=8)
        ax.legend(handles=[mlines.Line2D([], [], color=C_BETTER, marker="o", ls="None", label=fams[rname]["better"]),
                           mlines.Line2D([], [], color=C_WORSE, marker="o", ls="None", label=fams[rname]["worse"])],
                  loc="upper right", fontsize=7.8, frameon=False, handletextpad=0.2)

    fig.suptitle(f"{GROUP_TITLES.get(gid, f'Group {gid}')}: Prefix-step judgments by phase "
                 f"(n = {n_sc} scenarios, 5 runs each)", fontsize=12.5, y=0.995)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ saved: {args.out}")
    rec = [dict(family=f, dimension=s, tier=t, update_r=v[0], ci_low=v[1], ci_high=v[2])
           for (f, s), d in upd.items() for t, v in d.items()]
    out_csv = os.path.splitext(args.out)[0] + "_update.csv"
    pd.DataFrame(rec).to_csv(out_csv, index=False)
    print(pd.DataFrame(rec).round(2).to_string(index=False))
    drec = []
    srec, scen_rec = [], []
    for (f, sc), (on, dci, bold, summ, sdf, Rcl) in onsets.items():
        for ph, row in dci.iterrows():
            drec.append(dict(family=f, dimension=sc, key_option=key[sc], phase=labels_x.get(int(ph), ph),
                             better_minus_worse=row["diff"], ci_low=row["lo"], ci_high=row["hi"],
                             relative_closure=Rcl.get(ph, np.nan)))
        srec.append(dict(family=f, dimension=sc, key_option=key[sc],
                         bold=" + ".join(labels_x[int(b)] for b in bold), **summ))
        scen_rec.append(sdf.assign(family=f, dimension=sc).drop(columns=["phases"]))
    base = os.path.splitext(args.out)[0]
    pd.DataFrame(drec).to_csv(base + "_divergence.csv", index=False)
    pd.DataFrame(srec).to_csv(base + "_divergence_summary.csv", index=False)
    pd.concat(scen_rec).to_csv(base + "_divergence_by_scenario.csv", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(srec).to_string(index=False))


if __name__ == "__main__":
    main()
