import os
import sys
import argparse
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import zscore
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import warnings

warnings.filterwarnings("ignore")

# 1. 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.config import BASE_RESULTS_DIR
from src.prepare_everystep import load_reference_everystep

PATH_GROUPS = {
    'NoCheck': [3, 5],
    'Check-Stay': [2],
    'Check-Partial': [6, 7],
    'Check-GoBack': [1, 4]
}

TARGETS = {
    'Desire': ['desire_K', 'desire_L', 'desire_M'], 
    'Belief': ['belief_L', 'belief_M', 'belief_Empty']
}
SCORE_COLS = TARGETS['Desire'] + TARGETS['Belief']

# 💡 [핵심] 다중공선성 통제를 위해 카테고리별로 Predictor를 다르게 설정
PREDICTORS = {
    'Desire': ['btom', 'truebelief', 'nocost', 'motionheuristic'],
    'Belief': ['btom', 'truebelief', 'nocost', 'motionheuristic', 'hindsight']
}
# 💡 [NEW] 통합 Y축을 위해 전체 변수 리스트를 고정 순서로 정의합니다.
ALL_PREDICTORS = ['btom', 'truebelief', 'nocost', 'motionheuristic', 'hindsight']
ALL_REFS = ALL_PREDICTORS

def load_and_process_llm_data(model_list, condition):
    all_dfs = []
    
    for model in model_list:
        csv_path = os.path.join(BASE_RESULTS_DIR, model, condition, "everystep", "everystep_valid_only.csv")
        if os.path.exists(csv_path):
            all_dfs.append(pd.read_csv(csv_path))
            
    if not all_dfs: return None
        
    df_concat = pd.concat(all_dfs, ignore_index=True)
    df_mean = df_concat.groupby(['scenario_id', 'group_id', 'time_step'])[SCORE_COLS].mean().reset_index()
    
    belief_cols = TARGETS['Belief']
    df_mean_shifted = np.maximum(df_mean[belief_cols] - 1, 0)
    bel_sum = df_mean_shifted.sum(axis=1).replace(0, 1.0)
    df_mean[belief_cols] = df_mean_shifted.div(bel_sum, axis=0)
    
    return df_mean

def get_regression_stats(df_llm, group_ids, target_cols, current_refs, ref_dfs):
    """특정 LLM 그룹과 Path에 대해 회귀분석을 수행하고 통계치를 딕셔너리로 반환"""
    df_llm_path = df_llm[df_llm['group_id'].isin(group_ids)]
    if len(df_llm_path) == 0: return None

    df_llm_melt = df_llm_path.melt(id_vars=['scenario_id', 'time_step'], value_vars=target_cols, 
                                   var_name='target_type', value_name='LLM_AVG')
    df_reg = df_llm_melt.copy()
    
    for ref in current_refs:
        if ref in ref_dfs:
            df_ref_path = ref_dfs[ref][ref_dfs[ref]['group_id'].isin(group_ids)]
            df_ref_melt = df_ref_path.melt(id_vars=['scenario_id', 'time_step'], value_vars=target_cols, 
                                           var_name='target_type', value_name=ref)
            df_reg = pd.merge(df_reg, df_ref_melt, on=['scenario_id', 'time_step', 'target_type'], how='left')

    df_reg_clean = df_reg.drop(columns=['scenario_id', 'time_step', 'target_type']).dropna()
    if len(df_reg_clean) == 0: return None

    constant_cols = [col for col in df_reg_clean.columns if df_reg_clean[col].nunique() <= 1]
    if constant_cols: df_reg_clean.drop(columns=constant_cols, inplace=True)

    if 'btom' not in df_reg_clean.columns or 'LLM_AVG' not in df_reg_clean.columns: return None

    df_reg_std = df_reg_clean.apply(zscore)
    available_refs = [r for r in current_refs if r in df_reg_std.columns]
    
    X_all_with_const = sm.add_constant(df_reg_std[available_refs])
    X_base = sm.add_constant(df_reg_std[['btom']])
    
    model_1 = sm.OLS(df_reg_std['LLM_AVG'], X_base).fit()
    model_2 = sm.OLS(df_reg_std['LLM_AVG'], X_all_with_const).fit()
    
    return {
        'params': model_2.params.drop('const', errors='ignore'),
        'conf_int': model_2.conf_int().drop('const', errors='ignore'),
        'pvalues': model_2.pvalues.drop('const', errors='ignore'),
        'base_r2': model_1.rsquared_adj,
        'delta_r2': model_2.rsquared_adj - model_1.rsquared_adj
    }

def set_dynamic_xlim(ax, ci_lower_list, ci_upper_list):
    if not ci_lower_list or not ci_upper_list: return
    x_min, x_max = min(ci_lower_list), max(ci_upper_list)
    margin = (x_max - x_min) * 0.15
    if margin == 0: margin = 0.1 
    ax.set_xlim(x_min - margin, x_max + margin)

def get_top_predictor(stats):
    """유의미한(p<0.05) 양수(coef>0) 계수 중 가장 큰 값을 가진 변수 반환"""
    if not stats: return None
    params = stats['params']
    pvals = stats['pvalues']
    valid_vars = [v for v in params.index if pvals[v] < 0.05 and params[v] > 0]
    if not valid_vars: return None
    return max(valid_vars, key=lambda v: params[v])

def get_top_lesion_predictor(stats):
    """BToM을 제외한 결함 모델 중 유의미한 가장 큰 양수 계수 반환"""
    if not stats: return None
    params = stats['params']
    pvals = stats['pvalues']
    valid_vars = [v for v in params.index if v != 'btom' and pvals[v] < 0.05 and params[v] > 0]
    if not valid_vars: return None
    return max(valid_vars, key=lambda v: params[v])

def plot_dodged_forest(ax, stats_better, stats_worse, predictors, title, hide_y_labels=False, match_better=None, match_worse=None):
    # 💡 [NEW] Y축은 항상 4개(ALL_PREDICTORS)를 기준으로 고정합니다.
    y_pos_map = {var: i for i, var in enumerate(reversed(ALL_PREDICTORS))}
    y_ticks = list(y_pos_map.values())
    labels = [var.capitalize() if var != 'btom' else 'BToM' for var in reversed(ALL_PREDICTORS)]
    
    c_better_top, c_worse_top = "#33B137", "#F86FBD"   # 초록색(Better), 자홍색(Worse)
    c_btom_absolute = "#E63946"                          # BToM이 탑일 때의 빨간색
    c_gray = "#B0B0B0"                                   # 그 외 노이즈 모델들의 회색
    
    ci_lowers, ci_uppers = [], []
    
    # 💡 더동적 색상 제어를 위해 상위 예측 변수들을 미리 산출
    top_overall_better = get_top_predictor(stats_better)
    top_lesion_better = get_top_lesion_predictor(stats_better)
    
    top_overall_worse = get_top_predictor(stats_worse)
    top_lesion_worse = get_top_lesion_predictor(stats_worse)

    coord_better, coord_worse = None, None

    for var in predictors:
        base_y = y_pos_map[var]
        
        # 🔵 Better Models (+0.15)
        if stats_better and var in stats_better['params']:
            coef = stats_better['params'][var]
            ci_l, ci_u = stats_better['conf_int'].loc[var]
            pval = stats_better['pvalues'][var]
            y_val = base_y + 0.15
            
            # 💡 [색상 할당 논리 적용]
            if var == 'btom':
                color_b = c_btom_absolute if top_overall_better == 'btom' else c_gray
            else:
                color_b = c_better_top if var == top_lesion_better else c_gray

            # 💡 [NEW] Global Match일 경우 검은색 굵은 테두리 적용
            e_color_b = 'black' if var == match_better else 'white'
            e_width_b = 2.0 if var == match_better else 1.2

            ax.errorbar(coef, y_val, xerr=[[coef - ci_l], [ci_u - coef]],
                        fmt='o', color=color_b, markeredgecolor=e_color_b, markeredgewidth=e_width_b,
                        elinewidth=2.5, capsize=5, markersize=10)
            if pval < 0.05:
                stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*"
                ax.text(ci_u + 0.02, y_val, stars, color=color_b, va='center', fontweight='bold', fontsize=12)
            ci_lowers.append(ci_l); ci_uppers.append(ci_u)
            
            if var == top_overall_better: coord_better = (coef, y_val)

        # 🔴 Worse Models (-0.15)
        if stats_worse and var in stats_worse['params']:
            coef = stats_worse['params'][var]
            ci_l, ci_u = stats_worse['conf_int'].loc[var]
            pval = stats_worse['pvalues'][var]
            y_val = base_y - 0.15

            # 💡 [색상 할당 논리 적용]
            if var == 'btom':
                color_w = c_btom_absolute if top_overall_worse == 'btom' else c_gray
            else:
                color_w = c_worse_top if var == top_lesion_worse else c_gray

            # 💡 [NEW] Global Match일 경우 검은색 굵은 테두리 적용
            e_color_w = 'black' if var == match_worse else 'white'
            e_width_w = 2.0 if var == match_worse else 1.2
            
            ax.errorbar(coef, y_val, xerr=[[coef - ci_l], [ci_u - coef]],
                        fmt='^', color=color_w, markeredgecolor=e_color_w, markeredgewidth=e_width_w,
                        elinewidth=2.5, capsize=5, markersize=11)
            if pval < 0.05:
                stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*"
                ax.text(ci_u + 0.02, y_val, stars, color=color_w, va='center', fontweight='bold', fontsize=12)
            ci_lowers.append(ci_l); ci_uppers.append(ci_u)
            
            if var == top_overall_worse: coord_worse = (coef, y_val)

    # # 💡 [NEW] Top Predictor 끼리 Shift Arrow 연결
    # if coord_better and coord_worse:
    #     ax.annotate("", xy=coord_worse, xytext=coord_better,
    #                 arrowprops=dict(arrowstyle="->", color="#8186A9", alpha=0.6, lw=2.0, ls="--"))

    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Y축 라벨 설정 (Belief 플롯은 숨김 처리)
    ax.set_yticks(y_ticks)
    if hide_y_labels:
        ax.set_yticklabels([])
    else:
        ax.set_yticklabels(labels, fontsize=12, fontweight='bold')

    ax.set_ylim(min(y_ticks) - 0.6, max(y_ticks) + 0.6)
    set_dynamic_xlim(ax, ci_lowers, ci_uppers)
    
    ax.grid(axis='x', linestyle=':', alpha=0.6)
    ax.set_xlabel("Standardized Coefficient (β)", fontsize=10, fontweight='bold')
    if title: ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

def plot_paired_bars(ax, stats_better, stats_worse, title, show_ticks=None):
    width = 0.4
    x = np.array([0, 0.6]) # 두 막대의 X축 위치
    
    b_base = max(0, stats_better['base_r2']) if stats_better else 0
    b_delta = max(0, stats_better['delta_r2']) if stats_better else 0
    b_unexpl = max(0, 1.0 - (b_base + b_delta)) if stats_better else 0
    
    w_base = max(0, stats_worse['base_r2']) if stats_worse else 0
    w_delta = max(0, stats_worse['delta_r2']) if stats_worse else 0
    w_unexpl = max(0, 1.0 - (w_base + w_delta)) if stats_worse else 0
    
    base = [b_base, w_base]
    delta = [b_delta, w_delta]
    unexpl = [b_unexpl, w_unexpl]
    
    c_base, c_delta, c_unexpl = '#E63946', '#4A4E69', '#E5E5E5'
    
    ax.bar(x, base, width, color=c_base, edgecolor='white')
    ax.bar(x, delta, width, bottom=base, color=c_delta, edgecolor='white')
    ax.bar(x, unexpl, width, bottom=np.array(base)+np.array(delta), color=c_unexpl, edgecolor='white')
    
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(['Better', 'Worse'], fontsize=11, fontweight='bold')
    if title: ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    
    # 💡 [수정] 조건별 눈금 및 레이블 위치 분기 설정
    if show_ticks == 'left':
        ax.yaxis.tick_left()
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=10)
    elif show_ticks == 'right':
        ax.yaxis.tick_right()
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=10)
    else:
        ax.set_yticks([])
        ax.set_yticklabels([])
        
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    for i, (b, d) in enumerate(zip(base, delta)):
        if b > 0.05:
            ax.text(x[i], b/2, f"{b*100:.1f}", ha='center', va='center', color='white', fontsize=10, fontweight='bold')
        if d > 0.05:
            ax.text(x[i], b + d/2, f"{d*100:.1f}", ha='center', va='center', color='white', fontsize=10, fontweight='bold')

def draw_master_visualization(master_results, condition, save_dir):
    """4x2 매트릭스 도판 생성 (좌측: Forest 2개 / 우측: Bar 2개)"""
    n_rows = len(PATH_GROUPS)
    fig = plt.figure(figsize=(14, 5.5 * n_rows))
    
    # 좌측(Forest) 70%, 우측(Bar) 30% 비율 할당
    gs_main = gridspec.GridSpec(n_rows, 2, width_ratios=[7.0, 3.0], wspace=0.15, hspace=0.18)

    for row_idx, (path_name, path_data) in enumerate(master_results.items()):
        # 💡 [NEW] 통합 Y축을 위해 wspace를 0.05로 확 줄이고 sharey=True 적용
        gs_forest = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[row_idx, 0], wspace=0.05)
        ax_des = fig.add_subplot(gs_forest[0])
        ax_bel = fig.add_subplot(gs_forest[1])
        
        # Bar Plots 구역 (0~100%이므로 sharey=True)
        gs_bar = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[row_idx, 1], wspace=0.1)
        ax_des_r2 = fig.add_subplot(gs_bar[0])
        ax_bel_r2 = fig.add_subplot(gs_bar[1])
        
        # 행(Row) 라벨 표시 (Desire 플롯의 Y축 라벨로 활용)
        ax_des.set_ylabel(path_name, fontsize=16, fontweight='bold', labelpad=20, 
                          bbox=dict(facecolor='#F8F9FA', edgecolor='gray', boxstyle='round,pad=0.5'))

        # 타이틀은 첫 번째 행에만 표시
        title_des = "Desire Predictors" if row_idx == 0 else ""
        title_bel = "Belief Predictors" if row_idx == 0 else ""
        title_des_r2 = "Desire Adj. $R^2$" if row_idx == 0 else ""
        title_bel_r2 = "Belief Adj. $R^2$" if row_idx == 0 else ""

        # 💡 [NEW] Desire와 Belief 양쪽에서 Top Predictor 산출 및 일치(Match) 여부 확인
        des_b_top = get_top_predictor(path_data['Desire']['better'])
        des_w_top = get_top_predictor(path_data['Desire']['worse'])
        bel_b_top = get_top_predictor(path_data['Belief']['better'])
        bel_w_top = get_top_predictor(path_data['Belief']['worse'])

        match_b = des_b_top if (des_b_top == bel_b_top and des_b_top is not None) else None
        match_w = des_w_top if (des_w_top == bel_w_top and des_w_top is not None) else None

        # 1. Forest 그리기
        plot_dodged_forest(ax_des, path_data['Desire']['better'], path_data['Desire']['worse'], 
                           PREDICTORS['Desire'], title_des, hide_y_labels=False,
                           match_better=match_b, match_worse=match_w)
        plot_dodged_forest(ax_bel, path_data['Belief']['better'], path_data['Belief']['worse'], 
                           PREDICTORS['Belief'], title_bel, hide_y_labels=True,
                           match_better=match_b, match_worse=match_w)
        
        # 2. Paired Bar 그리기
        plot_paired_bars(ax_des_r2, path_data['Desire']['better'], path_data['Desire']['worse'], 
                         title_des_r2, show_ticks='left')
        plot_paired_bars(ax_bel_r2, path_data['Belief']['better'], path_data['Belief']['worse'], 
                         title_bel_r2, show_ticks='right')

    # 전체 범례
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor="#33B137", markersize=12, label='Better Models (Avg)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor="#F86FBD", markersize=12, label='Worse Models (Avg)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#E63946', markersize=12, label='BToM ($R^2$)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#4A4E69', markersize=12, label='Lesion ($R^2$)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#E5E5E5', markeredgecolor='gray', markersize=12, label='Unexplained')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.975), ncol=5, fontsize=13)

    fig.suptitle(f"Path-Specific Error Drivers (Condition: {condition})", 
                 fontsize=26, fontweight='bold', y=0.99)
    
    plt.subplots_adjust(top=0.92)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"path_forest_r2_plot_{condition}.png")
    plt.savefig(save_path, dpi=250, bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print(f"\n📊 SUCCESS: Master Visualization saved to {save_path}")

def run_path_specific_regression(better_models, worse_models, condition):
    print("\n" + "="*90)
    print(f"🚀 Processing Path-Specific Everystep Regression...")
    print("="*90)

    df_better = load_and_process_llm_data(better_models, condition)
    df_worse = load_and_process_llm_data(worse_models, condition)

    if df_better is None or df_worse is None: return

    ref_dfs = {}
    for ref in ALL_REFS:
        df_ref = load_reference_everystep(ref)
        if df_ref is not None:
            ref_dfs[ref] = df_ref[['scenario_id', 'time_step', 'group_id'] + SCORE_COLS]

    master_results = {}

    for path_name, group_ids in PATH_GROUPS.items():
        master_results[path_name] = {'Desire': {}, 'Belief': {}}
        
        for cat_name, target_cols in TARGETS.items():
            current_refs = PREDICTORS[cat_name]
            
            # Better 그룹 계산
            b_stats = get_regression_stats(df_better, group_ids, target_cols, current_refs, ref_dfs)
            master_results[path_name][cat_name]['better'] = b_stats
            
            # Worse 그룹 계산
            w_stats = get_regression_stats(df_worse, group_ids, target_cols, current_refs, ref_dfs)
            master_results[path_name][cat_name]['worse'] = w_stats

    # 시각화 실행
    plot_save_dir = os.path.join(parent_dir, "results")
    draw_master_visualization(master_results, condition, plot_save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Path-Specific Regression on Everystep data.")
    parser.add_argument("--condition", type=str, default="vanilla", help="Experiment condition")

    args = parser.parse_args()
    
    BETTER_MODELS = ['gemini-2.5-pro', 'deepseek-reasoner', 'deepseek-v4-pro', 'claude-opus-4-6'] 
    WORSE_MODELS = ['gpt-5.4', 'o4-mini', 'gemini-2.5-flash', 'claude-sonnet-4-6'] # gpt-4o, deepseek-chat은 데이터 없음
    
    run_path_specific_regression(BETTER_MODELS, WORSE_MODELS, args.condition)