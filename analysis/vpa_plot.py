"""
변이 분할(VPA) 공통 계산·그림 모듈. 논문 그림(Fig 4b, Fig 5b, Fig S3)의 디자인을 그대로 따르고 다음을 추가함.
  - 막대 끝 가로선: 고유 adjusted R²의 95% 시나리오 bootstrap CI
  - 막대 옆 숫자: 고유 adjusted R² (%)와 표준화 계수 β (음수는 빨간색)
  - 각 행 오른쪽 위: 전체 모형의 adjusted R² (total)
고유 adjusted R²은 잘라내지 않은 값(음수 가능)을 CSV에 저장하고, 그림의 막대는 0에서 자름.

다른 스크립트에서 사용:
  from analysis.vpa_plot import vpa_with_ci, draw_vpa, COLOR_MAP, REF_NAMES
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REF_ORDER = ["btom", "truebelief", "nocost", "motionheuristic", "hindsight"]
COLOR_MAP = {"btom": "#E63946", "truebelief": "#F5A623", "nocost": "#DC91EB",
             "motionheuristic": "#B8969A", "hindsight": "#747ED2"}
REF_NAMES = {"btom": "BToM", "truebelief": "TrueBelief", "nocost": "NoCost",
             "motionheuristic": "MotionHeuristic", "hindsight": "Hindsight"}


# ------------------------------------------------------------------ 계산
def _fit(yz, Xz):
    X1 = np.column_stack([np.ones(len(yz)), Xz])
    beta, *_ = np.linalg.lstsq(X1, yz, rcond=None)
    resid = yz - X1 @ beta
    n, k = Xz.shape
    r2 = 1 - (resid @ resid) / (yz @ yz)
    return 1 - (1 - r2) * (n - 1) / (n - k - 1), beta[1:]


def vpa_point(y, X, names):
    """z-점수화 후 전체 adjusted R², 각 예측변수의 고유 adjusted R²(잘라내지 않음), 표준화 β."""
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y, X = y[ok], X[ok]
    keep = X.std(axis=0) > 1e-9                      # 상수 예측변수는 제외
    names = [n for n, k in zip(names, keep) if k]
    X = X[:, keep]
    if len(y) < 10 or y.std() == 0 or not names:
        return None
    yz = (y - y.mean()) / y.std()
    Xz = (X - X.mean(axis=0)) / X.std(axis=0)
    total, beta = _fit(yz, Xz)
    out = {"total": total, "unique": {}, "beta": {}}
    for i, n in enumerate(names):
        red = _fit(yz, np.delete(Xz, i, axis=1))[0] if Xz.shape[1] > 1 else 0.0
        out["unique"][n] = total - red
        out["beta"][n] = beta[i]
    return out


def vpa_with_ci(y, X, names, scen, n_boot=1000, seed=0):
    """시나리오 단위 bootstrap (층화 없음). 반환: {'total': (est, lo, hi), 'unique': {ref: (est, lo, hi)}, 'beta': {...}}"""
    est = vpa_point(y, X, names)
    if est is None:
        return None
    rng = np.random.default_rng(seed)
    uniq = np.unique(scen)
    idx_of = {s: np.where(scen == s)[0] for s in uniq}
    boots = []
    for _ in range(n_boot):
        pick = np.concatenate([idx_of[s] for s in rng.choice(uniq, len(uniq))])
        b = vpa_point(y[pick], X[pick], names)
        if b is not None:
            boots.append(b)

    def ci(get):
        v = np.array([get(b) for b in boots], float)
        v = v[np.isfinite(v)]
        return tuple(np.percentile(v, [2.5, 97.5])) if len(v) else (np.nan, np.nan)

    res = {"total": (est["total"], *ci(lambda b: b["total"])), "unique": {}, "beta": {}}
    for n in est["unique"]:
        res["unique"][n] = (est["unique"][n], *ci(lambda b, n=n: b["unique"].get(n, np.nan)))
        res["beta"][n] = (est["beta"][n], *ci(lambda b, n=n: b["beta"].get(n, np.nan)))
    return res


def results_to_frame(rows):
    """draw_vpa 입력과 같은 rows → 긴 형식 DataFrame (CSV 저장용)."""
    out = []
    for label, by_dim in rows:
        for dim, r in by_dim.items():
            if r is None:
                continue
            out.append(dict(row=label, dimension=dim, term="total", estimate=r["total"][0],
                            ci_low=r["total"][1], ci_high=r["total"][2]))
            for kind in ("unique", "beta"):
                for ref, (e, lo, hi) in r[kind].items():
                    out.append(dict(row=label, dimension=dim, term=f"{kind}_{ref}", estimate=e, ci_low=lo, ci_high=hi))
    return pd.DataFrame(out)


def rows_from_summary(summary, labels, prefix="vpa"):
    """compare_stepwise_bootstrap의 stepwise_bootstrap_summary.csv에서 draw_vpa용 rows 생성."""
    rows = []
    for lab in labels:
        sub = summary[summary.condition == lab].set_index("metric")
        by_dim = {}
        for dim in ("desire", "belief"):
            g = lambda k: tuple(sub.loc[k, ["estimate", "ci_low", "ci_high"]]) if k in sub.index else None
            tot = g(f"{prefix}_{dim}_total")
            if tot is None:
                by_dim[dim.capitalize()] = None
                continue
            r = {"total": tot, "unique": {}, "beta": {}}
            for ref in REF_ORDER:
                u = g(f"{prefix}_{dim}_unique_{ref}")
                if u is not None and np.isfinite(u[0]):
                    r["unique"][ref] = u
                    b = g(f"{prefix}_{dim}_beta_{ref}")
                    r["beta"][ref] = b if b is not None else (np.nan, np.nan, np.nan)
            by_dim[dim.capitalize()] = r
        rows.append((lab, by_dim))
    return rows


# ------------------------------------------------------------------ 그림
def draw_vpa(rows, out_path, predictors=None, blocks=1, title=None, header=None, header_color="#5FA318",
             xmax=0.6, show_beta=True, show_ci=True, show_total=True, legend_refs=None, dpi=250,
             columns=None, col_fontsize=17, note=None):
    """
    rows: [(행 이름, {열 이름: res, ...}), ...]  위에서 아래 순서. res는 vpa_with_ci 결과(또는 None = no data)
    columns: 열 이름 목록. 기본 ["Desire", "Belief"]. 예: 조건 이름들 (행 = 모델, belief만)
    predictors: {열 이름: [...]} 막대 순서. None이면 REF_ORDER 중 res에 있는 것
    blocks: 행을 몇 묶음으로 나눠 옆으로 배치할지 (논문 Fig 4b는 2)
    header: 왼쪽 위 라벨 박스 문구(예: "Better"). None이면 생략
    """
    dims = list(columns) if columns else ["Desire", "Belief"]
    nc = len(dims)
    per_block = int(np.ceil(len(rows) / blocks))
    chunks = [rows[i * per_block:(i + 1) * per_block] for i in range(blocks)]
    # x축 상한: 막대·CI가 넘치지 않게
    hi_all = [r["unique"][x][2 if show_ci else 0] for _, bd in rows for r in bd.values() if r for x in r["unique"]]
    hi_all = [v for v in hi_all if np.isfinite(v)]
    xmax = max(xmax, (max(hi_all) + 0.13) if hi_all else xmax)
    # 열: [D, B] (+ 블록 사이 여백 열)
    widths = []
    for i in range(blocks):
        widths += [1] * nc + ([0.55] if i < blocks - 1 else [])
    fig = plt.figure(figsize=(4.6 * nc * blocks + 1.2 * blocks, 1.75 * per_block + 1.6))
    gs = fig.add_gridspec(1, len(widths), width_ratios=widths, wspace=0.42 if show_beta else 0.18)
    cols = [i for i, w in enumerate(widths) if w == 1]
    axes = [fig.add_subplot(gs[0, c]) for c in cols]
    bar_h, gap = 0.13, 0.155
    for b, chunk in enumerate(chunks):
        y_pos = np.arange(per_block)[::-1][:len(chunk)]
        for d, dim in enumerate(dims):
            ax = axes[nc * b + d]
            order = (predictors or {}).get(dim) or REF_ORDER
            for (label, by_dim), y0 in zip(chunk, y_pos):
                r = by_dim.get(dim)
                if r is None:
                    ax.text(0.01, y0, "no data", color="gray", style="italic", va="center", fontsize=9)
                    continue
                refs = [x for x in order if x in r["unique"]]
                offs = [((len(refs) - 1) / 2 - i) * gap for i in range(len(refs))]
                vals = {x: r["unique"][x][0] for x in refs}
                best = max(vals, key=vals.get) if vals else None
                for x, off in zip(refs, offs):
                    e, lo, hi = r["unique"][x]
                    is_max = (x == best and e > 0)
                    ax.barh(y0 + off, max(e, 0), height=bar_h, color=COLOR_MAP[x],
                            edgecolor="black" if is_max else "white", linewidth=1.8 if is_max else 0.5, alpha=0.95)
                    xt = max(e, 0)
                    if show_ci and np.isfinite(lo):
                        ax.plot([max(lo, 0), max(hi, 0)], [y0 + off] * 2, color="#222222", lw=0.9, zorder=4)
                        xt = max(xt, hi)
                    if e >= 0.01:
                        ax.text(xt + 0.008, y0 + off, f"{e*100:.1f}%", va="center", fontsize=10 if is_max else 7.5,
                                fontweight="bold" if is_max else "normal")
                    if show_beta and x in r["beta"] and np.isfinite(r["beta"][x][0]):
                        bv = r["beta"][x][0]
                        ax.text(1.02, y0 + off, f"β {bv:+.2f}", transform=ax.get_yaxis_transform(), va="center",
                                fontsize=7.2, color="#C0392B" if bv < 0 else "#555555",
                                fontweight="bold" if is_max else "normal")
                if show_total:
                    t = r["total"]
                    ax.text(xmax, y0 + (len(refs) / 2) * gap + 0.02,
                            f"total adj. R² {t[0]:.2f}" + (f" [{t[1]:.2f}, {t[2]:.2f}]" if np.isfinite(t[1]) else ""),
                            ha="right", va="bottom", fontsize=7.5, color="#666666", style="italic")
            ax.set_ylim(-0.6, per_block - 0.4)
            ax.set_xlim(0, xmax)
            ticks = np.arange(0, xmax + 1e-9, 0.2 if xmax > 0.7 else 0.1)
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{v:.1f}" for v in ticks], fontsize=9)
            ax.grid(axis="x", linestyle="--", alpha=0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_title(dim, fontsize=col_fontsize, fontweight="bold", pad=10)
            ax.set_xlabel("Adjusted $R^2$", fontsize=13)
            if d == 0:
                ax.set_yticks(y_pos)
                ax.set_yticklabels([lab for lab, _ in chunk], fontsize=12.5, fontweight="bold",
                                   bbox=dict(boxstyle="round,pad=0.35", fc="#F2F2F2", ec="#999999"))
                ax.tick_params(axis="y", length=0, pad=10)
            else:
                ax.set_yticks([])
    refs_leg = legend_refs or [x for x in REF_ORDER
                               if any(x in (bd.get(dm) or {}).get("unique", {}) for _, bd in rows for dm in dims)]
    handles = [Line2D([0], [0], marker="s", color="w", markerfacecolor=COLOR_MAP[x], markersize=14, label=REF_NAMES[x])
               for x in refs_leg]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0 + 0.32 / (per_block + 1)),
               ncol=len(handles), fontsize=12, frameon=True, edgecolor="#CCCCCC")
    note = note or "bar = unique adjusted R² (95% scenario-bootstrap CI); β = standardized coefficient in the full model"
    if title:
        fig.suptitle(title, fontsize=14, y=1.0 + 0.62 / (per_block + 1))
    fig.text(0.5, -0.02 - 0.06 / per_block, note, ha="center", fontsize=8.5, color="#666666")
    if header:
        fig.text(0.02, 0.97, header, fontsize=15, fontweight="bold", color="white",
                 bbox=dict(boxstyle="round,pad=0.4", fc=header_color, ec="none"))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ saved: {out_path}")
