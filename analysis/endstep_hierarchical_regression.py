import os
import sys
import argparse
import pickle
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import zscore
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import warnings

warnings.filterwarnings("ignore")

# 1. 경로 설정 (기존 프로젝트 구조 반영)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.config import REFERENCE_PKL_DIR, BASE_RESULTS_DIR, get_group_indices

# 평가할 모델 리스트
LLM_MODELS = ['gpt-4o', 'gpt-5.4', 'o4-mini', 'gemini-2.5-flash', 'gemini-2.5-pro', 
              'deepseek-chat', 'deepseek-reasoner', 
              'claude-sonnet-4-6', 'claude-opus-4-6']

# 분석에 사용할 예측 모델 (베이스라인: btom / 결함 및 휴리스틱 모델: 나머지)
LESION_MODELS = ['truebelief', 'nocost', 'motionheuristic'] # hindsight 제외
ALL_REFS = ['btom'] + LESION_MODELS

def get_valid_indices(total_scenarios=78, exclude_partial=False):
    """비합리적 시나리오를 제외한 유효 인덱스 반환 (plot_correlation.py 차용)"""
    group_inds = get_group_indices(include_irrational=False)
    num_groups = 5 if exclude_partial else 7

    valid_scenarios_1based = []
    for i in range(num_groups): 
        valid_scenarios_1based.extend(group_inds[i])
        
    valid_indices_0based = np.array(valid_scenarios_1based) - 1
    valid_mask = np.zeros(total_scenarios, dtype=bool)
    valid_mask[valid_indices_0based] = True
    
    return valid_mask

def load_pickle_safe(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

def extract_flat_data(data_dict, valid_mask, key):
    """
    (3, 78) 형태의 배열에서 유효한 시나리오만 뽑은 뒤 1차원으로 펼침
    """
    # shape: (3, num_valid)
    filtered_data = data_dict[key][:, valid_mask]
    # shape: (3 * num_valid,)
    return filtered_data.flatten()

def get_top_predictor_lesion(params, pvalues):
    """BToM을 제외한 Lesion 중 가장 큰 양의 계수를 가진 변수"""
    valid_vars = [v for v in params.index if v != 'btom' and pvalues[v] < 0.05 and params[v] > 0]
    if not valid_vars: return None
    return max(valid_vars, key=lambda v: params[v])

def get_top_predictor_overall(params, pvalues):
    """BToM을 포함하여 전체 중 가장 큰 양의 계수를 가진 변수"""
    valid_vars = [v for v in params.index if pvalues[v] < 0.05 and params[v] > 0]
    if not valid_vars: return None
    return max(valid_vars, key=lambda v: params[v])

def set_dynamic_xlim(ax, ci_lower_list, ci_upper_list):
    """X축의 최소/최대값을 찾아 15% 여백을 주고 스케일링"""
    if not ci_lower_list or not ci_upper_list: return
    x_min, x_max = min(ci_lower_list), max(ci_upper_list)
    margin = (x_max - x_min) * 0.15
    # 만약 분산이 0에 가까워 margin이 0이라면 최소 여백 보장
    if margin == 0: margin = 0.1 
    ax.set_xlim(x_min - margin, x_max + margin)

def draw_stacked_r2_bar(ax, base_r2, delta_r2, title="Adj. $R^2$", show_ticks=False):
    """우측에 작게 붙는 Stacked Bar Chart 그리기 함수"""
    base = max(0, base_r2)
    delta = max(0, delta_r2)
    unexplained = max(0, 1.0 - (base + delta))
    
    width = 0.6
    color_base = '#E63946'    
    color_delta = '#4A4E69'   
    color_unexpl = '#E5E5E5'  
    
    p1 = ax.bar(" ", base, width, color=color_base, edgecolor='white')
    p2 = ax.bar(" ", delta, width, bottom=base, color=color_delta, edgecolor='white')
    p3 = ax.bar(" ", unexplained, width, bottom=base+delta, color=color_unexpl, edgecolor='white')
    
    ax.set_ylim(0, 1.0)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=15)
    
    if show_ticks:
        ax.yaxis.tick_right()
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=10)
    else:
        ax.set_yticks([]) 
        ax.set_yticklabels([])

    if base > 0.05:
        ax.text(0, base/2, f"{base*100:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=10)
    if delta > 0.05:
        ax.text(0, base + delta/2, f"{delta*100:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=10)

    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_visible(False)

def draw_forest_plot(master_results_dict, condition, save_dir):
    """9개의 LLM 모델을 9행(Row)으로 시각화 (좌측: Forest 2개, 우측: Stacked Bar 2개)"""
    n_models = len(LLM_MODELS)
    
    # 전체 도판 크기 및 메인 GridSpec (좌측 75%, 우측 25% 할당)
    fig = plt.figure(figsize=(14, 5.5 * n_models))
    gs_main = gridspec.GridSpec(n_models, 2, width_ratios=[7.5, 2.5], wspace=0.15, hspace=0.4)
    
    # 색상 테마
    color_btom_top = '#E63946' # 강렬한 빨간색 (BToM Overall Top)
    color_des_top = '#F4A261' # 주황색 (Desire Top)
    color_bel_top = "#629AE4" # 파란색 (Belief Top)
    color_gray = '#B0B0B0'    # 회색

    for row_idx, llm_name in enumerate(LLM_MODELS):

        # 행별로 좌측 공간(Forest)을 2개로, 우측 공간(Bar)을 2개로 쪼개기
        gs_forest = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[row_idx, 0], wspace=0.05)
        ax_des = fig.add_subplot(gs_forest[0])
        ax_bel = fig.add_subplot(gs_forest[1], sharey=ax_des)
        
        gs_bar = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[row_idx, 1], wspace=0.1)
        ax_des_r2 = fig.add_subplot(gs_bar[0])
        ax_bel_r2 = fig.add_subplot(gs_bar[1], sharey=ax_des_r2)
        
        # 행(Row)의 Y축 라벨로 모델명 크게 표시
        ax_des.set_ylabel(llm_name, fontsize=16, fontweight='bold', labelpad=15)
        
        if llm_name not in master_results_dict:
            ax_des.text(0.5, 0.5, "Data Missing", ha='center', va='center', fontsize=12, color='gray')
            ax_bel.text(0.5, 0.5, "Data Missing", ha='center', va='center', fontsize=12, color='gray')
            # 결측 시 R2 Bar도 숨김 처리
            ax_des_r2.axis('off')
            ax_bel_r2.axis('off')
            continue
            
        results_dict = master_results_dict[llm_name]
        variables = results_dict['Desire']['params'].index
        labels = [idx.capitalize() if idx != 'btom' else 'BToM' for idx in variables]
        y_pos = np.arange(len(variables))[::-1] 
        
        # Desire 최고 기여도 산출
        top_lesion_des = get_top_predictor_lesion(results_dict['Desire']['params'], results_dict['Desire']['pvalues'])
        top_overall_des = get_top_predictor_overall(results_dict['Desire']['params'], results_dict['Desire']['pvalues'])
        
        # Belief 최고 기여도 산출
        top_lesion_bel = get_top_predictor_lesion(results_dict['Belief']['params'], results_dict['Belief']['pvalues'])
        top_overall_bel = get_top_predictor_overall(results_dict['Belief']['params'], results_dict['Belief']['pvalues'])

        # Desire와 Belief의 최고 모델이 동일한지(Global Match) 판별
        is_global_match = (top_overall_des == top_overall_bel) and (top_overall_des is not None)

        # ---------------------------------------------------------
        # 1. 왼쪽 덩어리: Desire Forest Plot
        # ---------------------------------------------------------
        des_ci_lower_all, des_ci_upper_all = [], []
        for i, var_name in enumerate(variables):
            coef = results_dict['Desire']['params'][var_name]
            ci_l, ci_u = results_dict['Desire']['conf_int'].loc[var_name]
            pval = results_dict['Desire']['pvalues'][var_name]
            
            # 하이라이트 논리 적용
            if var_name == 'btom':
                c_des = color_btom_top if top_overall_des == 'btom' else color_gray
            else:
                c_des = color_des_top if var_name == top_lesion_des else color_gray
            
            # Global Match일 경우 검은색 굵은 윤곽선 부여
            is_overall_top = (var_name == top_overall_des)
            edge_color = 'black' if (is_overall_top and is_global_match) else c_des
            edge_width = 2.0 if (is_overall_top and is_global_match) else 0.0

            ax_des.errorbar(coef, y_pos[i], xerr=[[coef - ci_l], [ci_u - coef]], 
                            fmt='o', color=c_des, markeredgecolor=edge_color, markeredgewidth=edge_width, 
                            ecolor=c_des, elinewidth=2.5, capsize=5, markersize=8)
            
            # Top 모델 여부와 상관없이 p < 0.05이면 별표 출력
            if pval < 0.05:
                stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*"
                ax_des.text(ci_u + 0.02, y_pos[i], stars, 
                            color=c_des, va='center', fontweight='bold', fontsize=12)

            # 가장 잘 설명하는 모델 점 바로 아래에 베타 수치 작성
            if is_overall_top:
                ax_des.text(coef, y_pos[i] - 0.28, f"{coef:.2f}", ha='center', va='top', 
                            color=c_des, fontweight='bold', fontsize=11)

            des_ci_lower_all.append(ci_l)
            des_ci_upper_all.append(ci_u)

        ax_des.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        set_dynamic_xlim(ax_des, des_ci_lower_all, des_ci_upper_all)
        
        ax_des.set_yticks(y_pos)
        ax_des.set_yticklabels(labels, fontsize=12, fontweight='bold')
        # Y축의 맨 아래(최소점 - 0.8)와 맨 위(최고점 + 0.8)에 빈 공간
        ax_des.set_ylim(min(y_pos) - 1, max(y_pos) + 1)

        ax_des.set_xlabel("Standardized Coefficient (β)", fontsize=11, fontweight='bold')
        ax_des.grid(axis='x', linestyle=':', alpha=0.6)

        # 첫 줄에만 타이틀을 주고, 기존 텍스트 박스는 삭제 후 Bar 함수 호출
        title_des = "Desire Predictors (β)" if row_idx == 0 else " "
        ax_des.set_title(title_des, fontsize=16, fontweight='bold', color=color_des_top, pad=15)
        
        # ---------------------------------------------------------
        # 2. 중간 덩어리: Belief Forest Plot
        # ---------------------------------------------------------
        bel_ci_lower_all, bel_ci_upper_all = [], []
        for i, var_name in enumerate(variables):
            coef_b = results_dict['Belief']['params'][var_name]
            ci_l_b, ci_u_b = results_dict['Belief']['conf_int'].loc[var_name]
            pval_b = results_dict['Belief']['pvalues'][var_name]
            
            # 하이라이트 논리 적용
            if var_name == 'btom':
                c_bel = color_btom_top if top_overall_bel == 'btom' else color_gray
            else:
                c_bel = color_bel_top if var_name == top_lesion_bel else color_gray
            
            # Global Match 윤곽선 적용
            is_overall_top_b = (var_name == top_overall_bel)
            edge_color_b = 'black' if (is_overall_top_b and is_global_match) else c_bel
            edge_width_b = 2.0 if (is_overall_top_b and is_global_match) else 0.0

            ax_bel.errorbar(coef_b, y_pos[i], xerr=[[coef_b - ci_l_b], [ci_u_b - coef_b]], 
                            fmt='o', color=c_bel, markeredgecolor=edge_color_b, markeredgewidth=edge_width_b,
                            ecolor=c_bel, elinewidth=2.5, capsize=5, markersize=8) # 모양 통일(o)
                            
            # Top 모델 여부와 상관없이 p < 0.05이면 별표 출력
            if pval_b < 0.05:
                stars = "***" if pval_b < 0.001 else "**" if pval_b < 0.01 else "*"
                ax_bel.text(ci_u_b + 0.02, y_pos[i], stars, 
                            color=c_bel, va='center', fontweight='bold', fontsize=12)
            
            # 가장 잘 설명하는 모델 점 바로 아래에 베타 수치 작성
            if is_overall_top_b:
                ax_bel.text(coef_b, y_pos[i] - 0.28, f"{coef:.2f}", ha='center', va='top', 
                            color=c_bel, fontweight='bold', fontsize=11)

            bel_ci_lower_all.append(ci_l_b)
            bel_ci_upper_all.append(ci_u_b)

        ax_bel.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        set_dynamic_xlim(ax_bel, bel_ci_lower_all, bel_ci_upper_all)
        
        plt.setp(ax_bel.get_yticklabels(), visible=False)
        ax_bel.set_xlabel("Standardized Coefficient (β)", fontsize=11, fontweight='bold')
        ax_bel.grid(axis='x', linestyle=':', alpha=0.6)
        
        title_bel = "Belief Predictors (β)" if row_idx == 0 else " "
        ax_bel.set_title(title_bel, fontsize=16, fontweight='bold', color=color_bel_top, pad=15)

        # ---------------------------------------------------------
        # 3. 오른쪽 덩어리: Stacked R2 Bars 모음
        # ---------------------------------------------------------
        title_des_r2 = "Desire\nAdj. $R^2$" if row_idx == 0 else " "
        title_bel_r2 = "Belief\nAdj. $R^2$" if row_idx == 0 else " "

        draw_stacked_r2_bar(ax_des_r2, results_dict['Desire']['base_r2'], results_dict['Desire']['delta_r2'], 
                            title=title_des_r2, show_ticks=False)
        
        draw_stacked_r2_bar(ax_bel_r2, results_dict['Belief']['base_r2'], results_dict['Belief']['delta_r2'], 
                            title=title_bel_r2, show_ticks=True)

    # ---------------------------------------------------------
    # 전체 마무리
    # ---------------------------------------------------------
    fig.suptitle(f"Primary Drivers of LLM Errors: (Condition: {condition})", 
                 fontsize=22, fontweight='bold', y=0.995)
    
    # 간격 조정
    plt.subplots_adjust(hspace=0.3, wspace=0.1, top=0.97)
    
    # 저장
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"forest_r2_plot_{condition}.png")

    # 이미지가 길기 때문에 해상도 유지 및 잘림 방지를 위해 bbox_inches 사용
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"   📊 Subplot forest plot saved to: {save_path}")

def run_hierarchical_regression(condition, exclude_partial):
    mode_text = "EXCL. PARTIAL" if exclude_partial else "INCL. PARTIAL"
    valid_mask = get_valid_indices(exclude_partial=exclude_partial)
    
    print("\n" + "="*90)
    print(f"🚀 Hierarchical Regression: [ALL MODELS | {condition}) | {mode_text}]")
    print("="*90)

    # 1. Reference 모델 데이터 한 번만 로드 (메모리 최적화)
    ref_data_dict = {}
    for ref in ALL_REFS:
        path = os.path.join(REFERENCE_PKL_DIR, ref, f"{ref}_data.pkl")
        data = load_pickle_safe(path)
        if data is not None:
            ref_data_dict[ref] = data
        else:
            print(f"⚠️ Warning: Reference data missing for {ref}")

    # 두 가지 카테고리(Desire, Belief)에 대해 각각 회귀분석 수행
    target_keys = {
        'Desire': 'des_inf_mean',
        'Belief': 'bel_inf_mean_norm'
    }
    
    # 🌟 모든 LLM의 결과를 모아둘 마스터 딕셔너리
    master_results_dict = {}

    for llm_name in LLM_MODELS:
        llm_path = os.path.join(BASE_RESULTS_DIR, llm_name, condition, "model_data.pkl")
        llm_data = load_pickle_safe(llm_path)
        
        if llm_data is None:
            print(f"   ❌ Data missing, skipping.")
            continue

        results_dict = {}
        for cat_name, key in target_keys.items():
            Y_llm = extract_flat_data(llm_data, valid_mask, key)
            df_reg = pd.DataFrame({'LLM': Y_llm})
            
            for ref in ALL_REFS:
                if ref in ref_data_dict:
                    df_reg[ref] = extract_flat_data(ref_data_dict[ref], valid_mask, key)

            df_reg.dropna(inplace=True)
            if len(df_reg) == 0: continue

            df_reg_std = df_reg.apply(zscore)
            available_refs = [r for r in ALL_REFS if r in df_reg_std.columns]
            
            X_all_with_const = sm.add_constant(df_reg_std[available_refs])
            if 'btom' not in df_reg_std.columns: continue

            # Block 1
            X_base = sm.add_constant(df_reg_std[['btom']])
            model_1 = sm.OLS(df_reg_std['LLM'], X_base).fit()
            
            # Block 2
            model_2 = sm.OLS(df_reg_std['LLM'], X_all_with_const).fit()
            delta_adj_r2 = model_2.rsquared_adj - model_1.rsquared_adj
            
            results_dict[cat_name] = {
                'params': model_2.params.drop('const'),
                'conf_int': model_2.conf_int().drop('const'),
                'pvalues': model_2.pvalues.drop('const'),
                'base_r2': model_1.rsquared_adj,
                'total_r2': model_2.rsquared_adj,
                'delta_r2': delta_adj_r2
            }

        if 'Desire' in results_dict and 'Belief' in results_dict:
            master_results_dict[llm_name] = results_dict
            print(f"   ✅ Done.")
        else:
            print(f"   ⚠️ Incomplete data.")

    # 모든 모델 처리가 끝난 후 단일 플롯으로 시각화
    if master_results_dict:
        # 결과를 저장할 공통 디렉토리 설정 (예: results/aggregate_plots)
        plot_save_dir = os.path.join(parent_dir, "results")
        draw_forest_plot(master_results_dict, condition, plot_save_dir)
    else:
        print("❌ No valid data collected to draw the batch plot.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hierarchical Regression on LLM ToM scores.")
    # parser.add_argument("--llm", type=str, required=True, help="LLM model name (e.g., gpt-4o)")
    parser.add_argument("--condition", type=str, default="vanilla", help="Experiment condition (e.g., vanilla)")
    parser.add_argument("--exclude_partial", action="store_true", help="Exclude Check-Partial groups")

    args = parser.parse_args()
    run_hierarchical_regression(args.condition, args.exclude_partial)