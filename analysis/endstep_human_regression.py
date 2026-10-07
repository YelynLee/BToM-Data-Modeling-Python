import os
import sys
import argparse
import pickle
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

from src.config import REFERENCE_PKL_DIR, BASE_RESULTS_DIR, HUMAN_PKL_PATH, get_group_indices

# LLM 논문 도판과 1:1 비교를 위해 동일한 결함 모델 세트 사용
LESION_MODELS = ['truebelief', 'nocost', 'motionheuristic'] 
ALL_REFS = ['btom'] + LESION_MODELS

def get_valid_indices(total_scenarios=78, exclude_partial=False):
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
    filtered_data = data_dict[key][:, valid_mask]
    return filtered_data.flatten()

def get_top_predictor_lesion(params, pvalues):
    valid_vars = [v for v in params.index if v != 'btom' and pvalues[v] < 0.05 and params[v] > 0]
    if not valid_vars: return None
    return max(valid_vars, key=lambda v: params[v])

def get_top_predictor_overall(params, pvalues):
    valid_vars = [v for v in params.index if pvalues[v] < 0.05 and params[v] > 0]
    if not valid_vars: return None
    return max(valid_vars, key=lambda v: params[v])

def set_dynamic_xlim(ax, ci_lower_list, ci_upper_list):
    if not ci_lower_list or not ci_upper_list: return
    x_min, x_max = min(ci_lower_list), max(ci_upper_list)
    margin = (x_max - x_min) * 0.15
    if margin == 0: margin = 0.1 
    ax.set_xlim(x_min - margin, x_max + margin)

def draw_stacked_r2_bar(ax, base_r2, delta_r2, title="Adj. $R^2$", show_ticks=False):
    """우측에 작게 붙는 Stacked Bar Chart 그리기 함수"""
    # R2 값이 음수일 경우 0으로 보정 (Adjusted R2의 특성 고려)
    base = max(0, base_r2)
    delta = max(0, delta_r2)
    unexplained = max(0, 1.0 - (base + delta))
    
    # 막대 폭 및 색상 설정
    width = 0.6
    color_base = '#E63946'    # BToM: 빨강
    color_delta = '#4A4E69'   # Lesions: 어두운 남색 계열
    color_unexpl = '#E5E5E5'  # Unexplained: 밝은 회색
    
    # 누적 막대 그리기
    p1 = ax.bar(" ", base, width, color=color_base, edgecolor='white')
    p2 = ax.bar(" ", delta, width, bottom=base, color=color_delta, edgecolor='white')
    p3 = ax.bar(" ", unexplained, width, bottom=base+delta, color=color_unexpl, edgecolor='white')
    
    # y축을 0~1 (0~100%)로 고정하고 우측으로 이동
    ax.set_ylim(0, 1.0)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=15)
    
    # 💡 가장 우측에 있는 막대에만 y축 퍼센트 표시
    if show_ticks:
        # Belief (우측 막대): 눈금과 % 라벨 표시
        ax.yaxis.tick_right()
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=10)
    else:
        # Desire (좌측 막대): 눈금 선과 라벨을 완벽히 제거
        ax.set_yticks([]) 
        ax.set_yticklabels([])
    
    # 텍스트 라벨 (비율이 5% 이상일 때만 박스 중앙에 텍스트 표시)
    if base > 0.05:
        ax.text(0, base/2, f"BToM\n{base*100:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=10)
    if delta > 0.05:
        ax.text(0, base + delta/2, f"Lesion\n{delta*100:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=10)

    # 테두리 가리기 (위, 아래, 왼쪽)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_visible(False)

def draw_human_forest_plot(results_dict, condition, save_dir):
    """Forest Plot과 Stacked R2 Bar Chart를 통합한 시각화"""
    # GridSpec을 사용하여 1x4 구조 생성 (Forest, Bar, Forest, Bar 비율 조정)
    fig = plt.figure(figsize=(14, 5.5))

    # 전체 공간을 좌측(Forest) 75% : 우측(Bar) 25% 비율로 나눔
    gs_main = gridspec.GridSpec(1, 2, width_ratios=[7.5, 2.5], wspace=0.15)
    
    # 좌측 공간을 다시 2개로 쪼개어 포레스트 플롯 2개 배치 (딱 붙게 wspace=0.05)
    gs_forest = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[0], wspace=0.05)
    ax_des = fig.add_subplot(gs_forest[0])
    ax_bel = fig.add_subplot(gs_forest[1], sharey=ax_des)
    
    # 우측 공간을 다시 2개로 쪼개어 누적 막대 2개 배치
    gs_bar = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[1], wspace=0.1)
    ax_des_r2 = fig.add_subplot(gs_bar[0])
    ax_bel_r2 = fig.add_subplot(gs_bar[1], sharey=ax_des_r2)

    color_btom_top = '#E63946' 
    color_des_top = '#F4A261'  
    color_bel_top = "#4681CE"  
    color_gray = '#B0B0B0'   
    
    ax_des.set_ylabel("HUMAN", fontsize=16, fontweight='bold', labelpad=15)
    
    variables = results_dict['Desire']['params'].index
    labels = [idx.capitalize() if idx != 'btom' else 'BToM' for idx in variables]
    y_pos = np.arange(len(variables))[::-1] 
    
    top_lesion_des = get_top_predictor_lesion(results_dict['Desire']['params'], results_dict['Desire']['pvalues'])
    top_overall_des = get_top_predictor_overall(results_dict['Desire']['params'], results_dict['Desire']['pvalues'])
    
    top_lesion_bel = get_top_predictor_lesion(results_dict['Belief']['params'], results_dict['Belief']['pvalues'])
    top_overall_bel = get_top_predictor_overall(results_dict['Belief']['params'], results_dict['Belief']['pvalues'])

    is_global_match = (top_overall_des == top_overall_bel) and (top_overall_des is not None)

    # ---------------------------------------------------------
    # 1. 왼쪽 덩어리: Desire & Belief Forest Plots
    # ---------------------------------------------------------
    des_ci_lower_all, des_ci_upper_all = [], []
    bel_ci_lower_all, bel_ci_upper_all = [], []

    for i, var_name in enumerate(variables):
        # Desire
        coef_d = results_dict['Desire']['params'][var_name]
        ci_l_d, ci_u_d = results_dict['Desire']['conf_int'].loc[var_name]
        pval_d = results_dict['Desire']['pvalues'][var_name]
        
        if var_name == 'btom':
            c_des = color_btom_top if top_overall_des == 'btom' else color_gray
        else:
            c_des = color_des_top if var_name == top_lesion_des else color_gray

        is_overall_top_d = (var_name == top_overall_des)
        edge_color_d = 'black' if (is_overall_top_d and is_global_match) else c_des
        edge_width_d = 2.0 if (is_overall_top_d and is_global_match) else 0.0

        ax_des.errorbar(coef_d, y_pos[i], xerr=[[coef_d - ci_l_d], [ci_u_d - coef_d]], 
                        fmt='o', color=c_des, markeredgecolor=edge_color_d, markeredgewidth=edge_width_d,
                        ecolor=c_des, elinewidth=2.5, capsize=5, markersize=9)
        
        if pval_d < 0.05:
            stars = "***" if pval_d < 0.001 else "**" if pval_d < 0.01 else "*"
            ax_des.text(ci_u_d + 0.02, y_pos[i], stars, color=c_des, va='center', fontweight='bold', fontsize=13)
        
        if is_overall_top_d:
            ax_des.text(coef_d, y_pos[i] - 0.28, f"{coef_d:.2f}", ha='center', va='top', color=c_des, fontweight='bold', fontsize=11)
            
        des_ci_lower_all.append(ci_l_d)
        des_ci_upper_all.append(ci_u_d)

        # Belief
        coef_b = results_dict['Belief']['params'][var_name]
        ci_l_b, ci_u_b = results_dict['Belief']['conf_int'].loc[var_name]
        pval_b = results_dict['Belief']['pvalues'][var_name]
        
        c_bel = color_btom_top if var_name == 'btom' and top_overall_bel == 'btom' else \
                color_bel_top if var_name == top_lesion_bel else color_gray

        is_overall_top_b = (var_name == top_overall_bel)
        edge_color_b = 'black' if (is_overall_top_b and is_global_match) else c_bel
        edge_width_b = 2.0 if (is_overall_top_b and is_global_match) else 0.0

        ax_bel.errorbar(coef_b, y_pos[i], xerr=[[coef_b - ci_l_b], [ci_u_b - coef_b]], 
                        fmt='o', color=c_bel, markeredgecolor=edge_color_b, markeredgewidth=edge_width_b,
                        ecolor=c_bel, elinewidth=2.5, capsize=5, markersize=9) 
                        
        if pval_b < 0.05:
            stars = "***" if pval_b < 0.001 else "**" if pval_b < 0.01 else "*"
            ax_bel.text(ci_u_b + 0.02, y_pos[i], stars, color=c_bel, va='center', fontweight='bold', fontsize=13)
        
        if is_overall_top_b:
            ax_bel.text(coef_b, y_pos[i] - 0.28, f"{coef_b:.2f}", ha='center', va='top', color=c_bel, fontweight='bold', fontsize=11)
            
        bel_ci_lower_all.append(ci_l_b)
        bel_ci_upper_all.append(ci_u_b)

    ax_des.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    set_dynamic_xlim(ax_des, des_ci_lower_all, des_ci_upper_all)
    
    ax_des.set_yticks(y_pos)
    ax_des.set_yticklabels(labels, fontsize=12, fontweight='bold')
    ax_des.set_ylim(min(y_pos) - 1, max(y_pos) + 1)
    ax_des.set_xlabel("Standardized Coefficient (β)", fontsize=11, fontweight='bold')
    ax_des.grid(axis='x', linestyle=':', alpha=0.6)
    ax_des.set_title("Desire Predictors (β)", fontsize=16, fontweight='bold', pad=15)

    ax_bel.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    set_dynamic_xlim(ax_bel, bel_ci_lower_all, bel_ci_upper_all)
    
    # ax_bel.set_yticks() 는 sharey=True 덕분에 생략하거나 안보이게 처리됨
    plt.setp(ax_bel.get_yticklabels(), visible=False)
    ax_bel.set_xlabel("Standardized Coefficient (β)", fontsize=11, fontweight='bold')
    ax_bel.grid(axis='x', linestyle=':', alpha=0.6)
    ax_bel.set_title("Belief Predictors (β)", fontsize=16, fontweight='bold', pad=15)

    # ---------------------------------------------------------
    # 2. 오른쪽 덩어리: Stacked R2 Bars
    # ---------------------------------------------------------
    draw_stacked_r2_bar(ax_des_r2, results_dict['Desire']['base_r2'], results_dict['Desire']['delta_r2'], 
                        title="Desire\nAdj. $R^2$", show_ticks=False)
    
    # 💡 우측 끝 막대에만 눈금(100%) 표시
    draw_stacked_r2_bar(ax_bel_r2, results_dict['Belief']['base_r2'], results_dict['Belief']['delta_r2'], 
                        title="Belief\nAdj. $R^2$", show_ticks=True)

    # ---------------------------------------------------------
    # 3. 전체 범례 및 마무리
    # ---------------------------------------------------------
    # legend_elements = [
    #     Line2D([0], [0], marker='o', color='w', markerfacecolor=color_btom_top, markersize=11, label='BToM (Absolute Top)'),
    #     Line2D([0], [0], marker='o', color='w', markerfacecolor=color_des_top, markersize=11, label='Desire (Top Lesion)'),
    #     Line2D([0], [0], marker='o', color='w', markerfacecolor=color_bel_top, markersize=11, label='Belief (Top Lesion)'),
    #     Line2D([0], [0], marker='o', color='w', markerfacecolor='#E0E0E0', markeredgecolor='black', markeredgewidth=2, markersize=11, label='Global Top Driver (Match)'),
    #     Line2D([0], [0], marker='o', color='w', markerfacecolor=color_gray, markersize=11, label='Not Primary / N.S.')
    # ]
    # fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.97), ncol=5, fontsize=11)

    fig.suptitle(f"Primary Drivers of Human Inferences (Condition: {condition})", 
                 fontsize=20, fontweight='bold', y=1.05)
    
    plt.subplots_adjust(wspace=0.1, top=0.85) 
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"human_forest_r2_plot_{condition}.png")
    
    plt.savefig(save_path, dpi=250, bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print("\n" + "="*90)
    print(f"✅ SUCCESS: Human grid plot saved to: {save_path}")
    print("="*90)

def run_human_flattened_regression(condition, exclude_partial):
    valid_mask = get_valid_indices(exclude_partial=exclude_partial)
    
    print("\n" + "="*90)
    print(f"🧠 Human Flattened Regression [Condition: {condition}]")
    print("="*90)

    human_data = load_pickle_safe(HUMAN_PKL_PATH)
    if human_data is None:
        print(f"❌ Error: Human data not found.")
        return

    ref_data_dict = {}
    for ref in ALL_REFS:
        path = os.path.join(REFERENCE_PKL_DIR, ref, f"{ref}_data.pkl")
        data = load_pickle_safe(path)
        if data is not None:
            ref_data_dict[ref] = data

    target_keys = {'Desire': 'des_inf_mean', 'Belief': 'bel_inf_mean_norm'}
    results_dict = {}

    for cat_name, key in target_keys.items():
        # 특정 선지가 아니라 전체 데이터를 Flatten
        Y_human = extract_flat_data(human_data, valid_mask, key)
        df_reg = pd.DataFrame({'HUMAN': Y_human})
        
        for ref in ALL_REFS:
            if ref in ref_data_dict:
                df_reg[ref] = extract_flat_data(ref_data_dict[ref], valid_mask, key)

        df_reg.dropna(inplace=True)
        if len(df_reg) == 0: continue

        df_reg_std = df_reg.apply(zscore)
        available_refs = [r for r in ALL_REFS if r in df_reg_std.columns]
        
        X_all_with_const = sm.add_constant(df_reg_std[available_refs])
        X_base = sm.add_constant(df_reg_std[['btom']])

        model_1 = sm.OLS(df_reg_std['HUMAN'], X_base).fit()
        model_2 = sm.OLS(df_reg_std['HUMAN'], X_all_with_const).fit()
        
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
        plot_save_dir = os.path.join(parent_dir, "results")
        draw_human_forest_plot(results_dict, condition, plot_save_dir)
    else:
        print("❌ Data insufficient to draw the plot.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Flattened Regression on Human data.")
    parser.add_argument("--condition", type=str, default="vanilla", help="Experiment condition")
    parser.add_argument("--exclude_partial", action="store_true", help="Exclude Check-Partial groups")

    args = parser.parse_args()
    run_human_flattened_regression(args.condition, args.exclude_partial)



# import os
# import sys
# import argparse
# import pickle
# import pandas as pd
# import numpy as np
# import statsmodels.api as sm
# from scipy.stats import zscore
# import warnings

# warnings.filterwarnings("ignore")

# # 1. 경로 설정
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)

# if parent_dir not in sys.path:
#     sys.path.append(parent_dir)

# # 💡 HUMAN_PKL_PATH를 추가로 임포트합니다.
# from src.config import REFERENCE_PKL_DIR, BASE_RESULTS_DIR, HUMAN_PKL_PATH, get_group_indices

# LESION_MODELS = ['truebelief', 'nocost', 'motionheuristic']
# ALL_REFS = ['btom'] + LESION_MODELS

# TARGET_MAPPING = {
#     'Desire': {0: 'Target K', 1: 'Target L', 2: 'Target M'},
#     'Belief': {0: 'Target L', 1: 'Target M', 2: 'Target Empty(N)'}
# }

# def get_valid_indices(total_scenarios=78, exclude_partial=False):
#     group_inds = get_group_indices(include_irrational=False)
#     num_groups = 5 if exclude_partial else 7
#     valid_scenarios_1based = []
#     for i in range(num_groups): 
#         valid_scenarios_1based.extend(group_inds[i])
#     valid_indices_0based = np.array(valid_scenarios_1based) - 1
#     valid_mask = np.zeros(total_scenarios, dtype=bool)
#     valid_mask[valid_indices_0based] = True
#     return valid_mask

# def load_pickle_safe(path):
#     if os.path.exists(path):
#         with open(path, 'rb') as f:
#             return pickle.load(f)
#     return None

# def extract_target_data(data_dict, valid_mask, key, target_idx):
#     target_data = data_dict[key][target_idx, :]
#     return target_data[valid_mask]

# def run_human_target_regression(exclude_partial):
#     mode_text = "EXCL. PARTIAL" if exclude_partial else "INCL. PARTIAL"
#     valid_mask = get_valid_indices(exclude_partial=exclude_partial)
    
#     print("\n" + "="*90)
#     print(f"🧠 Human Validation: Target-Specific Regression [Mode: {mode_text}]")
#     print("="*90)

#     # 1. Human 데이터 로드 (종속 변수 Y)
#     human_data = load_pickle_safe(HUMAN_PKL_PATH)
#     if human_data is None:
#         print(f"❌ Error: Human data not found at {HUMAN_PKL_PATH}")
#         return
#     else:
#         print("   ✅ Human empirical data loaded successfully.")

#     # 2. Reference 모델 데이터 로드 (독립 변수 X)
#     ref_data_dict = {}
#     for ref in ALL_REFS:
#         path = os.path.join(REFERENCE_PKL_DIR, ref, f"{ref}_data.pkl")
#         data = load_pickle_safe(path)
#         if data is not None:
#             ref_data_dict[ref] = data

#     target_keys = {'Desire': 'des_inf_mean', 'Belief': 'bel_inf_mean_norm'}

#     for cat_name, key in target_keys.items():
#         print(f"\n\n{'='*30} [ {cat_name.upper()} ] {'='*30}")
        
#         for target_idx, target_name in TARGET_MAPPING[cat_name].items():
#             print(f"\n🔍 Analyzing: {target_name}")
#             print("-" * 50)
            
#             # Y 변수를 Human 데이터로 설정
#             Y_human = extract_target_data(human_data, valid_mask, key, target_idx)
#             df_reg = pd.DataFrame({'HUMAN': Y_human})
            
#             for ref in ALL_REFS:
#                 if ref in ref_data_dict:
#                     df_reg[ref] = extract_target_data(ref_data_dict[ref], valid_mask, key, target_idx)

#             df_reg.dropna(inplace=True)
#             if len(df_reg) == 0: continue

#             # 상수 변수(분산이 0인 열) 제거
#             constant_cols = [col for col in df_reg.columns if df_reg[col].nunique() <= 1]
#             if constant_cols:
#                 df_reg.drop(columns=constant_cols, inplace=True)

#             if 'btom' not in df_reg.columns: continue

#             # 표준화 (Z-score)
#             df_reg_std = df_reg.apply(zscore)
#             available_refs = [r for r in ALL_REFS if r in df_reg_std.columns]
            
#             X_all = df_reg_std[available_refs]
#             X_all_with_const = sm.add_constant(X_all)
            
#             # 회귀분석 수행
#             X_base = sm.add_constant(df_reg_std[['btom']])
#             model_1 = sm.OLS(df_reg_std['HUMAN'], X_base).fit()
#             model_2 = sm.OLS(df_reg_std['HUMAN'], X_all_with_const).fit()
            
#             delta_adj_r2 = model_2.rsquared_adj - model_1.rsquared_adj
            
#             print(f"   ▶ Baseline (BToM Only) Adj. R² : {model_1.rsquared_adj:.4f}")
#             print(f"   ▶ Full Model (Lesions) Adj. R² : {model_2.rsquared_adj:.4f}  (Δ Adj. R² = {delta_adj_r2:.4f})")
            
#             print(f"\n   [ Human Coefficients for {target_name} ]")
#             print(f"   {'Predictor':<15} | {'Beta (β)':<10} | {'p-value':<10}")
            
#             for var in available_refs:
#                 coef = model_2.params[var]
#                 pval = model_2.pvalues[var]
#                 stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "n.s."
                
#                 # 💡 인간이 BToM을 잘 따르는지 확인하기 위해 BToM에도 💡 하이라이트 부여
#                 highlight = "💡" if pval < 0.05 and coef > 0.05 else "  "
#                 print(f" {highlight} {var:<13} | {coef:>8.4f} {stars:<4} | {pval:.4f}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Run Human Validation Target-Specific Regression.")
#     parser.add_argument("--exclude_partial", action="store_true", help="Exclude Check-Partial groups")

#     args = parser.parse_args()
#     run_human_target_regression(args.exclude_partial)