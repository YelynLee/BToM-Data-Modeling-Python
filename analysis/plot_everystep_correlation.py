import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from scipy.stats import pearsonr
import matplotlib.lines as mlines
import warnings

warnings.filterwarnings("ignore")

# 1. 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.config import REFERENCE_PKL_DIR, BASE_RESULTS_DIR, HUMAN_PKL_PATH, get_group_indices

# =========================================================================
# 🌟 [설정] 모델 및 시나리오
# =========================================================================
REF_MODELS = ['human', 'btom', 'truebelief', 'nocost', 'motionheuristic', 'hindsight']
DISPLAY_MODELS = ['human', 'btom', 'truebelief', 'nocost', 'motionheur', 'hindsight']

# 💡 [NEW] 우수 모델 리스트 및 통합(Aggregate) 모델 추가
BETTER_MODELS = ['gemini-2.5-pro', 'deepseek-reasoner', 'deepseek-v4-pro', 'claude-opus-4-6']
PLOT_ROWS = ['Aggregate (Better Models)'] + BETTER_MODELS

# Human vs BToM 기준값 (Benchmark)
BENCHMARKS = {
    'desire': {'ind': 0.91, 'grp': 0.97},
    'belief': {'ind': 0.78, 'grp': 0.90}
}

# =========================================================================
# 🌟 [함수] 데이터 추출 및 연산 헬퍼
# =========================================================================
def get_valid_indices(total_scenarios=78, include_partial=False):
    """비합리적 시나리오 및 Check-Partial 포함 여부에 따른 유효 시나리오 마스크 반환"""
    group_inds = get_group_indices(include_irrational=False)
    num_groups = 7 if include_partial else 5

    valid_scenarios_1based = []
    for i in range(num_groups): 
        valid_scenarios_1based.extend(group_inds[i])
        
    valid_indices_0based = np.array(valid_scenarios_1based) - 1
    valid_mask = np.zeros(total_scenarios, dtype=bool)
    valid_mask[valid_indices_0based] = True
    
    return valid_mask

def calc_group_means(mat_3x78, include_partial=False):
    """(3, 78) 행렬을 받아 그룹 단위 평균 (3, num_groups) 행렬로 변환"""
    group_inds = get_group_indices(include_irrational=False)
    num_groups = 7 if include_partial else 5
    
    grp_mat = np.full((3, num_groups), np.nan)
    for g_idx in range(num_groups):
        # 1-based ID를 0-based index로 변환
        sc_idxs = np.array(group_inds[g_idx]) - 1
        grp_mat[:, g_idx] = np.nanmean(mat_3x78[:, sc_idxs], axis=1)
    return grp_mat

def calc_r(x, y):
    """상관계수(r) 계산 (NaN 제외)"""
    x_flat, y_flat = x.flatten(), y.flatten()
    mask = ~np.isnan(x_flat) & ~np.isnan(y_flat)
    x_clean, y_clean = x_flat[mask], y_flat[mask]
    
    if len(x_clean) < 2 or np.std(x_clean) == 0 or np.std(y_clean) == 0:
        return np.nan
        
    r, _ = pearsonr(x_clean, y_clean)
    return r

def load_pickle_safe(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

def get_everystep_final_matrices(model_name, condition):
    """💡 [NEW] 특정 모델의 Everystep 마지막 응답을 추출하여 (3, 78) 행렬로 반환"""
    csv_path = os.path.join(BASE_RESULTS_DIR, model_name, condition, "everystep", "everystep_valid_only.csv")
    if not os.path.exists(csv_path): return None, None
    
    df = pd.read_csv(csv_path)
    score_cols = ['desire_K', 'desire_L', 'desire_M', 'belief_L', 'belief_M', 'belief_Empty']
    df_mean = df.groupby(['scenario_id', 'group_id', 'time_step'])[score_cols].mean().reset_index()

    belief_cols = ['belief_L', 'belief_M', 'belief_Empty']
    df_mean_shifted = np.maximum(df_mean[belief_cols] - 1, 0)
    bel_sum = df_mean_shifted.sum(axis=1).replace(0, 1.0)
    df_mean[belief_cols] = df_mean_shifted.div(bel_sum, axis=0)

    # 각 시나리오의 마지막 time_step 추출
    df_final = df_mean.sort_values('time_step').drop_duplicates(subset=['scenario_id', 'group_id'], keep='last')
    
    des_mat = np.full((3, 78), np.nan)
    bel_mat = np.full((3, 78), np.nan)
    
    for _, row in df_final.iterrows():
        sc_idx = int(row['scenario_id']) - 1 # 0-based
        des_mat[0, sc_idx] = row['desire_K']
        des_mat[1, sc_idx] = row['desire_L']
        des_mat[2, sc_idx] = row['desire_M']
        bel_mat[0, sc_idx] = row['belief_L']
        bel_mat[1, sc_idx] = row['belief_M']
        bel_mat[2, sc_idx] = row['belief_Empty']
        
    return des_mat, bel_mat

# =========================================================================
# 🌟 [메인] 시각화 함수
# =========================================================================
def plot_everystep_correlation_bars(condition="vanilla", include_partial=False):
    mode_text = "INCL. PARTIAL" if include_partial else "EXCL. PARTIAL"
    print(f"📊 [{condition.upper()} | {mode_text}] Every-step Final Response Correlation 생성을 시작합니다...")
    
    valid_mask = get_valid_indices(include_partial=include_partial)
    
    # ---------------------------------------------------------
    # 1. Reference 데이터 미리 로드 (End-step 데이터 기준)
    # ---------------------------------------------------------
    ref_matrices = {}
    for ref in REF_MODELS:
        if ref == 'human':
            data = load_pickle_safe(HUMAN_PKL_PATH)
        else:
            data = load_pickle_safe(os.path.join(REFERENCE_PKL_DIR, ref, f"{ref}_data.pkl"))
            
        if data:
            des_m = data['des_inf_mean']
            bel_m = data['bel_inf_mean_norm']
            ref_matrices[ref] = {
                'des_ind': des_m, 
                'bel_ind': bel_m,
                'des_grp': calc_group_means(des_m, include_partial),
                'bel_grp': calc_group_means(bel_m, include_partial)
            }

    # ---------------------------------------------------------
    # 2. LLM (Better Models) 데이터 로드 및 Aggregate 연산
    # ---------------------------------------------------------
    llm_matrices = {}
    all_des_mats, all_bel_mats = [], []
    
    for llm in BETTER_MODELS:
        des_mat, bel_mat = get_everystep_final_matrices(llm, condition)
        if des_mat is not None and bel_mat is not None:
            llm_matrices[llm] = {
                'des_ind': des_mat,
                'bel_ind': bel_mat,
                'des_grp': calc_group_means(des_mat, include_partial),
                'bel_grp': calc_group_means(bel_mat, include_partial)
            }
            all_des_mats.append(des_mat)
            all_bel_mats.append(bel_mat)
            
    # 앙상블(Aggregate) 행렬 생성
    if all_des_mats:
        agg_des = np.nanmean(np.stack(all_des_mats), axis=0)
        agg_bel = np.nanmean(np.stack(all_bel_mats), axis=0)
        llm_matrices['Aggregate (Better Models)'] = {
            'des_ind': agg_des,
            'bel_ind': agg_bel,
            'des_grp': calc_group_means(agg_des, include_partial),
            'bel_grp': calc_group_means(agg_bel, include_partial)
        }

    # ---------------------------------------------------------
    # 3. 캔버스 준비 (Rows: Aggregate + Better Models)
    # ---------------------------------------------------------
    n_rows = len(PLOT_ROWS)
    fig, axes = plt.subplots(nrows=n_rows, ncols=2, figsize=(12, 3 * n_rows))
    plt.subplots_adjust(hspace=0.4, wspace=0.2)
    
    colors = ["#91BBEA", "#9BE8D7", '#F5A623', '#F8E71C', "#B8969A", "#E4A1F2"]

    def draw_bars_with_group_ext(ax, r_ind_values, r_grp_values, col_type):
        x = np.arange(len(REF_MODELS))
        ref_ind, ref_grp = BENCHMARKS[col_type]['ind'], BENCHMARKS[col_type]['grp']

        ax.axhline(ref_ind, color='red', linestyle='-', linewidth=1.5, alpha=0.7, zorder=0)
        ax.axhline(ref_grp, color='red', linestyle='--', linewidth=1.2, alpha=0.5, zorder=0)

        if ax.get_subplotspec().rowspan.start == 0 and ax.get_subplotspec().colspan.start == 0:
            ax.text(len(REF_MODELS)-0.5, ref_ind - 0.12, f'Indiv ({ref_ind})', color='red', fontsize=8, ha='right', va='bottom', fontweight='bold', alpha=0.8)
            ax.text(len(REF_MODELS)-0.5, ref_grp + 0.02, f'Group ({ref_grp})', color='red', fontsize=8, ha='right', va='bottom', alpha=0.6)

        valid_inds = [v for v in r_ind_values if not np.isnan(v)]
        max_val = max(valid_inds) if valid_inds else np.nan
        max_idx = r_ind_values.index(max_val) if not np.isnan(max_val) else -1
        
        for i, ref in enumerate(REF_MODELS):
            r_ind, r_grp = r_ind_values[i], r_grp_values[i]
            is_max = (i == max_idx)
            edge_color, line_width = ('red', 2.5) if is_max else ('black', 1.0)
            
            if np.isnan(r_ind):
                ax.text(i, 0.05, 'N/A', ha='center', va='bottom', color='gray', fontsize=10, fontweight='bold')
                continue
                
            ax.bar(i, r_ind, color=colors[i], edgecolor=edge_color, linewidth=line_width, alpha=0.85, zorder=2)
            
            if not np.isnan(r_grp):
                ax.vlines(i, r_ind, r_grp, color='black', linewidth=1.5, linestyle='--', zorder=3)
                ax.hlines(r_grp, i - 0.2, i + 0.2, color='black', linewidth=1.5, zorder=3)
                
            font_weight, font_size = ('bold', 11) if is_max else ('normal', 10)
            
            if abs(r_ind) < 0.15:
                y_pos, va = (r_ind + 0.05, 'bottom') if r_ind >= 0 else (r_ind - 0.05, 'top')
            else:
                y_pos, va = r_ind / 2, 'center'
                
            txt = ax.text(i, y_pos, f'{r_ind:.2f}', ha='center', va=va, color='black', fontweight=font_weight, fontsize=font_size)
            txt.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])

        ax.set_ylim(-1.0, 1.15)
        ax.axhline(0, color='black', linewidth=1.2, zorder=1)
        ax.set_xlim(-0.6, len(REF_MODELS) - 0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(DISPLAY_MODELS, ha='center', fontsize=11)
        ax.grid(axis='y', linestyle=':', alpha=0.6, zorder=1)

    # ---------------------------------------------------------
    # 4. LLM 순회하며 서브플롯 그리기
    # ---------------------------------------------------------
    for row_idx, llm in enumerate(PLOT_ROWS):
        ax_des, ax_bel = axes[row_idx, 0], axes[row_idx, 1]

        if row_idx == 0:
            ax_des.set_title("Desire Correlation", fontsize=16, fontweight='bold', pad=15)
            ax_bel.set_title("Belief Correlation", fontsize=16, fontweight='bold', pad=15)
            
        # 💡 [NEW] Aggregate 행은 색상과 폰트 크기를 다르게 하여 시각적으로 분리
        label_color = '#E63946' if llm == 'Aggregate (Better Models)' else 'black'
        font_size = 15 if llm == 'Aggregate (Better Models)' else 13
        ax_des.set_ylabel(llm, fontsize=font_size, fontweight='bold', labelpad=15, color=label_color)

        llm_data = llm_matrices.get(llm)
        r_des_ind, r_bel_ind = [], []
        r_des_grp, r_bel_grp = [], []
        
        for ref in REF_MODELS:
            ref_data = ref_matrices.get(ref)
            if llm_data is None or ref_data is None:
                r_des_ind.append(np.nan); r_bel_ind.append(np.nan)
                r_des_grp.append(np.nan); r_bel_grp.append(np.nan)
                continue
                
            # Individual Data 연산
            x_des_i, y_des_i = llm_data['des_ind'][:, valid_mask], ref_data['des_ind'][:, valid_mask]
            x_bel_i, y_bel_i = llm_data['bel_ind'][:, valid_mask], ref_data['bel_ind'][:, valid_mask]
            
            r_des_ind.append(calc_r(x_des_i, y_des_i))
            r_bel_ind.append(calc_r(x_bel_i, y_bel_i))

            # Grouped Data 연산
            x_des_g, y_des_g = llm_data['des_grp'], ref_data['des_grp']
            x_bel_g, y_bel_g = llm_data['bel_grp'], ref_data['bel_grp']
            
            r_des_grp.append(calc_r(x_des_g, y_des_g))
            r_bel_grp.append(calc_r(x_bel_g, y_bel_g))

        draw_bars_with_group_ext(ax_des, r_des_ind, r_des_grp, 'desire')
        draw_bars_with_group_ext(ax_bel, r_bel_ind, r_bel_grp, 'belief')

    # 범례 추가
    custom_line = mlines.Line2D([], [], color='black', linestyle='--', marker='_', markersize=10, markeredgewidth=1.5, label='Grouped Correlation')
    ref_line = mlines.Line2D([], [], color='red', linestyle='-', linewidth=1.5, label='Human-BToM Correlation')
    axes[0, 0].legend(handles=[custom_line, ref_line], loc='lower right', fontsize=10)

    # ---------------------------------------------------------
    # 5. 마무리 및 저장
    # ---------------------------------------------------------
    results_dir = os.path.join(parent_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    suffix = "with_partial" if include_partial else "no_partial"
    
    save_path = os.path.join(results_dir, f"correlation_everystep_final_bars_{condition}_{suffix}.png")
    
    plt.suptitle(f"Everystep Final Response vs Reference Models (Condition: {condition.capitalize()})", 
                 fontsize=24, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0.3)
    print(f"✅ Everystep Correlation Bar graph saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, default="vanilla")
    parser.add_argument("--include_partial", action="store_true", help="Include Check-Partial (G6, G7) groups")
    args = parser.parse_args()
    
    plot_everystep_correlation_bars(condition=args.condition, include_partial=args.include_partial)