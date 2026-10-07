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

import warnings
warnings.filterwarnings("ignore")

# 1. 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.config import REFERENCE_PKL_DIR, BASE_RESULTS_DIR, get_group_indices

WORSE_MODELS = ['gpt-4o', 'gpt-5.4', 'o4-mini', 'gemini-2.5-flash', 'deepseek-chat', 'claude-sonnet-4-6']
BETTER_MODELS = ['gemini-2.5-pro', 'deepseek-reasoner', 'deepseek-v4-pro', 'claude-opus-4-6']
LLM_MODELS = WORSE_MODELS + BETTER_MODELS

ALL_REFS = ['btom', 'truebelief', 'nocost', 'motionheuristic'] # hindsight 제외 버전

COLOR_MAP = {
    'btom': '#E63946',             # 빨강
    'truebelief': '#F5A623',       # 주황
    'nocost': "#DC91EB",           # 보라
    'motionheuristic': "#B8969A",  # 갈색
    # 'shared_btom': '#457B9D',      # 🔵 남색 (BToM ∩ Lesion: 합리성의 착시)
    # 'shared_lesion': '#1D3557',    # 🔷 옅은 남색 (Lesion ∩ Lesion: 편향의 뒤엉킴)
    # 'unexplained': '#E5E5E5'       # ⚪ 회색 (설명되지 않은 영역, 100% 채우기용)
}

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

def calculate_vpa(df_reg_std, target_col, predictors):
    """
    전체 모델의 설명력을 바탕으로 각 Predictor가 단독으로 기여하는
    '순수 고유 설명력(Unique Variance Explained)'만을 계산하여 반환합니다.
    """
    # 1. Full Model 피팅 (전체 설명력)
    X_all = sm.add_constant(df_reg_std[predictors])
    r2_total = max(0, sm.OLS(df_reg_std[target_col], X_all).fit().rsquared_adj)
    
    # 2. Full Model 내 각 변수별 고유 분산 산출
    unique_vars = {}
    for var in predictors:
        others = [p for p in predictors if p != var]
        X_others = sm.add_constant(df_reg_std[others])
        r2_others = max(0, sm.OLS(df_reg_std[target_col], X_others).fit().rsquared_adj)
        unique_vars[var] = max(0, r2_total - r2_others)
        
    # # 3. Lesion-only Model 피팅 (BToM을 완전히 배제한 하위 집합 모델)
    # lesions = [p for p in predictors if p != 'btom']
    # X_lesions = sm.add_constant(df_reg_std[lesions])
    # r2_lesions = max(0, sm.OLS(df_reg_std[target_col], X_lesions).fit().rsquared_adj)
    
    # # 4. Lesion-only 하위 모델 내부의 고유 분산 구하기
    # unique_lesions_only = {}
    # for les in lesions:
    #     other_lesions = [l for l in lesions if l != les]
    #     X_other_les = sm.add_constant(df_reg_std[other_lesions])
    #     r2_other_les = max(0, sm.OLS(df_reg_std[target_col], X_other_les).fit().rsquared_adj)
    #     unique_lesions_only[les] = max(0, r2_lesions - r2_other_les)
        
    # # 5. [핵심 분리] 
    # # 편향들끼리만 순수하게 공유하는 분산 (Shared Lesion-only)
    # shared_lesion_only = max(0, r2_lesions - sum(unique_lesions_only.values()))
    
    # # 전체 공통 분산 구하기
    # total_shared = max(0, r2_total - sum(unique_vars.values()))
    
    # # BToM과 결합된 가짜 합리성 공통 분산 = 전체 공통 분산 - 순수 편향 공통 분산
    # shared_btom_involved = max(0, total_shared - shared_lesion_only)
    
    # # 100% 중 설명되지 않은 나머지 잔차 영역
    # unexplained = max(0, 1.0 - r2_total)
    
    return {
        'total_r2': r2_total,
        'unique': unique_vars,
        # 'shared_btom': shared_btom_involved,
        # 'shared_lesion': shared_lesion_only,
        # 'unexplained': unexplained
    }

# =========================================================================
# 🌟 [Step 1] 전체 4변수 VPA 시각화 함수 (Paired Bar)
# =========================================================================
def draw_vpa_grouped_bars(master_results, condition, save_dir):
    n_models = len(LLM_MODELS)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 2 * n_models), sharey=True)
    ax_des, ax_bel = axes[0], axes[1]

    # Y축 위치 설정 (위에서부터 아래로 읽도록 역순 정렬)
    y_positions = np.arange(n_models)[::-1]

    # 4개의 막대를 그리기 위한 그룹 내 오프셋 및 두께 설정
    bar_height = 0.12
    bar_spacing = 0.14

    # 시각적 스택 순서 정의
    # draw_order = ['shared_lesion', 'shared_btom', 'btom', 'truebelief', 'nocost', 'motionheuristic', 'unexplained']
    draw_order = ['btom', 'truebelief', 'nocost', 'motionheuristic']
    offsets = [1.5 * bar_spacing, 0.5 * bar_spacing, -0.5 * bar_spacing, -1.5 * bar_spacing]

    for idx, llm in enumerate(LLM_MODELS):
        base_y = y_positions[idx]

        # 데이터가 없는 경우의 처리
        if llm not in master_results:
            ax_des.text(0.1, base_y, "Data Missing", color='gray', style='italic')
            ax_bel.text(0.1, base_y, "Data Missing", color='gray', style='italic')
            continue

        vpa_des = master_results[llm]['Desire']['full_vpa']['unique']
        vpa_bel = master_results[llm]['Belief']['full_vpa']['unique']

        # 🌟 1. 현재 LLM에서 가장 높은 비율(최댓값) 찾기
        max_des_val = max([vpa_des.get(ref, 0) for ref in draw_order]) if vpa_des else 0
        max_bel_val = max([vpa_bel.get(ref, 0) for ref in draw_order]) if vpa_bel else 0

        for i, ref in enumerate(draw_order):
            val_des = vpa_des.get(ref, 0)
            val_bel = vpa_bel.get(ref, 0)
            
            # Desire 막대
            is_max_des = (val_des == max_des_val and val_des > 0)
            
            # 최댓값이면 검은색 굵은 테두리, 아니면 얇은 흰색 테두리
            edge_c_des = 'black' if is_max_des else 'white'
            lw_des = 2.2 if is_max_des else 0.5

            ax_des.barh(base_y + offsets[i], val_des, height=bar_height, 
                        color=COLOR_MAP[ref], edgecolor=edge_c_des, 
                        linewidth=lw_des, alpha=0.9)
            
            # 수치 텍스트 (1% 이상일 때만 표기)
            if val_des >= 0.01:
                # 🌟 최댓값일 경우 폰트 강조 로직
                f_weight = 'bold' if is_max_des else 'normal'
                f_size = 11 if is_max_des else 9

                ax_des.text(val_des + 0.01, base_y + offsets[i], f"{val_des*100:.1f}%", 
                            va='center', fontsize=f_size, fontweight=f_weight, color='black')

            # Belief 막대
            is_max_bel = (val_bel == max_bel_val and val_bel > 0)
            
            # 최댓값이면 검은색 굵은 테두리, 아니면 얇은 흰색 테두리
            edge_c_bel = 'black' if is_max_bel else 'white'
            lw_bel = 2.2 if is_max_bel else 0.5

            ax_bel.barh(base_y + offsets[i], val_bel, height=bar_height, 
                        color=COLOR_MAP[ref], edgecolor=edge_c_bel,
                        linewidth=lw_bel, alpha=0.9)
            
            if val_bel >= 0.01:
                # 🌟 최댓값일 경우 폰트 강조 로직
                f_weight = 'bold' if is_max_bel else 'normal'
                f_size = 11 if is_max_bel else 9

                ax_bel.text(val_bel + 0.01, base_y + offsets[i], f"{val_bel*100:.1f}%", 
                            va='center', fontsize=f_size, fontweight=f_weight, color='black')

    # 🌟 Worse Models와 Better Models 사이를 시각적으로 구분하는 점선 추가
    separator_idx = len(WORSE_MODELS) - 0.5
    sep_y = y_positions[0] - separator_idx
    for ax in [ax_des, ax_bel]:
        ax.axhline(sep_y, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)

    # 축 및 레이블 꾸미기
    for ax, title in zip([ax_des, ax_bel], ["Desire: Unique Variance", "Belief: Unique Variance"]):
        ax.set_yticks(y_positions)
        ax.set_yticklabels(LLM_MODELS, fontsize=14, fontweight='bold')
        ax.set_xlabel("Adjusted $R^2$", fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # X축의 상한선을 데이터의 최대값에 맞게 약간의 여유를 둠 (보통 Unique는 0.5를 넘기 힘드므로 0.6 정도로 세팅)
        ax.set_xlim(0, 0.6)

    # 범례 설정
    legend_elements = [
        # Line2D([0], [0], marker='s', color='w', markerfacecolor=COLOR_MAP['shared_lesion'], markersize=14, label='Lesion ∩ Lesion'),
        # Line2D([0], [0], marker='s', color='w', markerfacecolor=COLOR_MAP['shared_btom'], markersize=14, label='BToM ∩ Lesions'),
        # Line2D([0], [0], marker='s', color='w', markerfacecolor="#91BBEA", markersize=14, label='Human'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLOR_MAP['btom'], markersize=14, label='BToM'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLOR_MAP['truebelief'], markersize=14, label='TrueBelief'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLOR_MAP['nocost'], markersize=14, label='NoCost'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLOR_MAP['motionheuristic'], markersize=14, label='MotionHeur'),
        # Line2D([0], [0], marker='s', color='w', markerfacecolor=COLOR_MAP['unexplained'], markersize=14, label='Unexplained')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.01), ncol=4, fontsize=12)
    fig.suptitle(f"End-Step VPA: Unique Cognitive Model Contributions (Condition: {condition})", fontsize=22, fontweight='bold', y=1.06)
    
    plt.tight_layout(w_pad=5)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"vpa_endstep_extended_{condition}.png")
    plt.savefig(save_path, dpi=250, bbox_inches='tight')
    plt.close()
    print(f"\n📊 Unique VPA Stacked Bars saved to: {save_path}")

def run_vpa_analysis(condition, exclude_partial):
    mode_text = "EXCL. PARTIAL" if exclude_partial else "INCL. PARTIAL"
    valid_mask = get_valid_indices(exclude_partial=exclude_partial)
    
    print("\n" + "="*90)
    print(f"🚀 VPA Analysis [ALL MODELS | {condition} | {mode_text}]")
    print("="*90)

    ref_data_dict = {}
    for ref in ALL_REFS:
        path = os.path.join(REFERENCE_PKL_DIR, ref, f"{ref}_data.pkl")
        data = load_pickle_safe(path)
        if data is not None:
            ref_data_dict[ref] = data

    target_keys = {
        'Desire': 'des_inf_mean',
        'Belief': 'bel_inf_mean_norm'
    }
    
    master_results_dict = {}

    for llm_name in LLM_MODELS:
        llm_path = os.path.join(BASE_RESULTS_DIR, llm_name, condition, "model_data.pkl")
        llm_data = load_pickle_safe(llm_path)
        
        if llm_data is None:
            continue

        print(f"\n⏳ VPA Processing: {llm_name}...")
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
            
            # 1. Step 1: 전체 변수로 VPA
            full_vpa_res = calculate_vpa(df_reg_std, 'LLM', available_refs)
            
            # lesion_uniques = {k: v for k, v in full_vpa_res['unique'].items() if k != 'btom'}
            # top_lesion = max(lesion_uniques, key=lesion_uniques.get) if lesion_uniques else None
            
            # # 2. Step 2: 1:1 VPA 재계산
            # if top_lesion and 'btom' in available_refs:
            #     # 1:1일 때는 기존 calculate_vpa(모든 경우의 공통 분산을 합침)와 메커니즘이 동일함
            #     X_p = sm.add_constant(df_reg_std[['btom', top_lesion]])
            #     r2_p = max(0, sm.OLS(df_reg_std['LLM'], X_p).fit().rsquared_adj)
                
            #     u_b = max(0, r2_p - max(0, sm.OLS(df_reg_std['LLM'], sm.add_constant(df_reg_std[[top_lesion]])).fit().rsquared_adj))
            #     u_l = max(0, r2_p - max(0, sm.OLS(df_reg_std['LLM'], sm.add_constant(df_reg_std[['btom']])).fit().rsquared_adj))
            #     sh_p = max(0, r2_p - u_b - u_l)
                
            #     pair_vpa_res = {'total_r2': r2_p, 'unique': {'btom': u_b, top_lesion: u_l}, 'shared': sh_p}
            # else:
            #     pair_vpa_res = None
                
            # 3. 양쪽 결과를 모두 저장
            results_dict[cat_name] = {
                'full_vpa': full_vpa_res
                # 'pairwise_vpa': pair_vpa_res,
                # 'top_lesion': top_lesion
            }

        if 'Desire' in results_dict and 'Belief' in results_dict:
            master_results_dict[llm_name] = results_dict

    if master_results_dict:
        plot_save_dir = os.path.join(parent_dir, "results")
        draw_vpa_grouped_bars(master_results_dict, condition, plot_save_dir)
    else:
        print("❌ No valid data collected to draw the VPA plots.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Hybrid VPA on LLM ToM scores.")
    parser.add_argument("--condition", type=str, default="vanilla", help="Experiment condition")
    parser.add_argument("--exclude_partial", action="store_true", help="Exclude Check-Partial groups")

    args = parser.parse_args()
    run_vpa_analysis(args.condition, args.exclude_partial)