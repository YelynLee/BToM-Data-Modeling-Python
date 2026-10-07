"""
S5 Figure: correlation with human ratings, with scenario-level bootstrap CIs.
Rows = LLMs (sorted by desire r) + reference models. Cols = desire, belief.
"""
import os, sys, pickle, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import pearsonr

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.config import (REFERENCE_PKL_DIR, BASE_RESULTS_DIR,
                        HUMAN_PKL_PATH, get_group_indices)

# ------------------------------------------------------------------ config
BETTER = ['gemini-2.5-pro', 'deepseek-reasoner', 'deepseek-v4-pro', 'claude-opus-4-6']
WORSE  = ['gpt-4o', 'gpt-5.4', 'o4-mini', 'gemini-2.5-flash',
          'deepseek-chat', 'claude-sonnet-4-6']
LLMS   = BETTER + WORSE

# REFS = ['btom'] # other references are not needed for tripartite comparison

DISPLAY = {
    'gemini-2.5-pro': 'Gemini-2.5-Pro', 'gemini-2.5-flash': 'Gemini-2.5-Flash',
    'deepseek-reasoner': 'DeepSeek-R1', 'deepseek-v4-pro': 'DeepSeek-V4-Pro',
    'deepseek-chat': 'DeepSeek-V3.2',
    'claude-opus-4-6': 'Claude-4.6-Opus', 'claude-sonnet-4-6': 'Claude-4.6-Sonnet',
    'gpt-4o': 'GPT-4o', 'gpt-5.4': 'GPT-5.4', 'o4-mini': 'GPT-o4-mini'
    # 'btom': 'BToM', 'truebelief': 'TrueBelief', 'nocost': 'NoCost',
    # 'motionheuristic': 'MotionHeuristic', 'hindsight': 'HindSight',
}

FAMILY_COLOR = {'openai': '#E8703A', 'gemini': '#6C8EBF',
                'deepseek': '#7FB069', 'anthropic': '#C77DBB'
                # 'reference': '#8C8C8C'
                }

def family(name):
    # if name in REFS: return 'reference'
    if name.startswith(('gpt', 'o4')): return 'openai'
    if name.startswith('gemini'):      return 'gemini'
    if name.startswith('deepseek'):    return 'deepseek'
    if name.startswith('claude'):      return 'anthropic'
    return 'reference'

BENCHMARK = {'desire': 0.91, 'belief': 0.78}   # human-BToM, individual level
KEY = {'desire': 'des_inf_mean', 'belief': 'bel_inf_mean_norm'}

# ------------------------------------------------------------------ io
def load_pkl(path):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

def valid_mask(include_partial=True, total=78):
    """Irrational paths always excluded; Check-Partial optional."""
    groups = get_group_indices(include_irrational=False)
    n_groups = 7 if include_partial else 5
    ids = np.concatenate([np.asarray(groups[i]) for i in range(n_groups)])
    m = np.zeros(total, dtype=bool)
    m[ids - 1] = True
    return m

# ------------------------------------------------------------------ stats
def corr(x, y):
    """x, y: (3, n_scen). Flattened Pearson r, NaN-safe."""
    xf, yf = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    ok = ~np.isnan(xf) & ~np.isnan(yf)
    if ok.sum() < 3:
        return np.nan
    xc, yc = xf[ok], yf[ok]
    if np.std(xc) == 0 or np.std(yc) == 0:
        return np.nan
    return pearsonr(xc, yc)[0]

def rmse(x, y):
    xf, yf = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    ok = ~np.isnan(xf) & ~np.isnan(yf)
    return np.sqrt(np.mean((xf[ok] - yf[ok]) ** 2)) if ok.sum() else np.nan

def make_boot_index(n_scen, n_boot, seed=0):
    """Shared resampling indices -> paired bootstrap across all models."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_scen, size=(n_boot, n_scen))

def boot_ci(x, y, boot_idx, alpha=0.05):
    """Scenario-level (column) bootstrap; 3 options move together."""
    rs = np.array([corr(x[:, idx], y[:, idx]) for idx in boot_idx])
    rs = rs[~np.isnan(rs)]
    if rs.size == 0:
        return np.nan, np.nan, rs
    lo, hi = np.percentile(rs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi, rs

# ------------------------------------------------------------------ main
def build(condition='vanilla', include_partial=True, n_boot=2000, seed=0):
    mask = valid_mask(include_partial)
    n_scen = int(mask.sum())
    print(f'[info] {n_scen} scenarios, {n_scen * 3} data points')
    boot_idx = make_boot_index(n_scen, n_boot, seed)

    human = load_pkl(HUMAN_PKL_PATH)
    if human is None:
        raise FileNotFoundError(HUMAN_PKL_PATH)

    # --- LLM correlations ---
    rows, boot_store = [], {}
    for m in LLMS:
        d = load_pkl(os.path.join(BASE_RESULTS_DIR, m, condition, 'model_data.pkl'))
        if d is None:
            print(f'[warn] missing: {m}')
            continue
        rec = {'model': m, 'group': 'better' if m in BETTER else 'worse'}
        for dim, key in KEY.items():
            x = np.asarray(d[key], float)[:, mask]
            y = np.asarray(human[key], float)[:, mask]
            r = corr(x, y)
            lo, hi, rs = boot_ci(x, y, boot_idx)
            rec[f'r_{dim}'], rec[f'lo_{dim}'], rec[f'hi_{dim}'] = r, lo, hi
            rec[f'rmse_{dim}'] = rmse(x, y)
            boot_store[(m, dim)] = rs
        rows.append(rec)

    return pd.DataFrame(rows), boot_store

    # # paired comparison against BToM
    # for dim in KEY:
    #     if ('btom', dim) in boot_store:
    #         b = boot_store[('btom', dim)]
    #         df[f'p_le_btom_{dim}'] = [
    #             np.mean(boot_store[(m, dim)] <= b) if (m, dim) in boot_store else np.nan
    #             for m in df['model']]
    # return df, boot_store

# ------------------------------------------------------------------ plot
def plot(df, out_path, show_rmse=True):
    d = df.sort_values('r_desire', ascending=False).reset_index(drop=True)
    order = list(d.model)
    n = len(order)
    ypos = {m: -i for i, m in enumerate(order)}

    FS_TITLE, FS_LABEL, FS_TICK, FS_LEG = 17, 15, 13, 13

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 0.52 * n + 3.2), sharey=True)

    for ax, dim in zip(axes, ['desire', 'belief']):

        ax.axvline(0, color='k', lw=0.9, alpha=0.5, zorder=1)
        ax.axvline(BENCHMARK[dim], color='#D1495B', ls='--', lw=1.6, zorder=2)

        # label entirely to the LEFT of the dashed line
        ax.text(BENCHMARK[dim] - 0.025, 0.62,
                f'human–BToM ({BENCHMARK[dim]:.2f})',
                color='#D1495B', fontsize=FS_TICK - 1,
                ha='right', va='center', fontweight='bold')

        for _, row in d.iterrows():
            m = row['model']
            r, lo, hi = row[f'r_{dim}'], row[f'lo_{dim}'], row[f'hi_{dim}']
            c = FAMILY_COLOR[family(m)]
            filled = row['group'] == 'better'
            ax.errorbar(r, ypos[m],
                        xerr=[[max(r - lo, 0)], [max(hi - r, 0)]],
                        fmt='o', ms=8, color=c, ecolor=c, elinewidth=1.7,
                        capsize=3.2, capthick=1.7, zorder=4,
                        markerfacecolor=c if filled else 'white',
                        markeredgecolor=c, markeredgewidth=2.0)
            if show_rmse and np.isfinite(row[f'rmse_{dim}']):
                ax.annotate(f'{row[f"rmse_{dim}"]:.2f}', xy=(1.03, ypos[m]),
                            xycoords=('axes fraction', 'data'),
                            fontsize=FS_TICK - 2, color='0.35', va='center')

        if show_rmse:
            ax.annotate('RMSE', xy=(1.03, 0.62),
                        xycoords=('axes fraction', 'data'),
                        fontsize=FS_TICK - 2, color='0.35', fontweight='bold',
                        va='center')

        ax.set_xlim(-0.55, 1.0)
        ax.set_xlabel(f'r with human {dim} ratings', fontsize=FS_LABEL)
        ax.set_title(dim.capitalize(), fontsize=FS_TITLE, fontweight='bold', pad=12)
        ax.tick_params(axis='x', labelsize=FS_TICK)
        ax.grid(axis='x', ls=':', alpha=0.4, zorder=0)

    axes[0].set_yticks([ypos[m] for m in order])
    axes[0].set_yticklabels([DISPLAY.get(m, m) for m in order], fontsize=FS_TICK + 1)
    axes[0].set_ylim(-(n - 1) - 0.8, 1.3)

    handles = [
        Line2D([], [], marker='o', ls='', ms=9, color='0.3',
               markerfacecolor='0.3', label='Better'),
        Line2D([], [], marker='o', ls='', ms=9, color='0.3',
               markerfacecolor='white', markeredgewidth=2.0, label='Worse'),
        Line2D([], [], color='#D1495B', ls='--', lw=1.6, label='human–BToM')
    ]
    fig.legend(handles=handles, 
               loc='center left',
               bbox_to_anchor=(0.91, 0.5), # bbox_to_anchor=(0.5, -0.005)
               ncol=1,
               fontsize=FS_LEG, frameon=True, framealpha=1.0, edgecolor='0.7')

    fig.tight_layout(rect=[0, 0, 0.90, 1])
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f'[saved] {out_path}')
    plt.close(fig)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--condition', default='vanilla')
    p.add_argument('--exclude_partial', action='store_true')
    p.add_argument('--n_boot', type=int, default=2000)
    p.add_argument('--no_rmse', action='store_true')
    args = p.parse_args()

    df, _ = build(args.condition,
                  include_partial=not args.exclude_partial,
                  n_boot=args.n_boot)

    out_dir = os.path.join(parent_dir, 'results')
    os.makedirs(out_dir, exist_ok=True)
    tag = 'no_partial' if args.exclude_partial else 'with_partial'

    df.to_csv(os.path.join(out_dir, f'S5_figS1_{args.condition}_{tag}.csv'),
              index=False, float_format='%.4f')
    plot(df, os.path.join(out_dir, f'S5_figS1_{args.condition}_{tag}.png'),
         show_rmse=not args.no_rmse)

    print(df[['model', 'group', 'r_desire', 'lo_desire', 'hi_desire',
              'r_belief', 'lo_belief', 'hi_belief']].to_string(index=False))