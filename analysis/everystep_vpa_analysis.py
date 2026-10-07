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

from src.config import BASE_RESULTS_DIR
from src.prepare_everystep import load_reference_everystep
from analysis.plot_everystep import get_phase_index

# =========================================================================
# 🌟 [설정] 모델, 레퍼런스, 색상, 그룹 정의
# =========================================================================
WORSE_MODELS = ['gpt-5.4', 'o4-mini', 'gemini-2.5-flash', 'claude-sonnet-4-6'] # gpt-4o, deepseek-chat은 데이터 없음
BETTER_MODELS = ['gemini-2.5-pro', 'deepseek-reasoner', 'deepseek-v4-pro', 'claude-opus-4-6']
LLM_MODELS = WORSE_MODELS + BETTER_MODELS 

ALL_REFS = ['btom', 'truebelief', 'nocost', 'motionheuristic', 'hindsight']

PREDICTORS = {
    'Desire': ['btom', 'truebelief', 'nocost', 'motionheuristic'],
    'Belief': ['btom', 'truebelief', 'nocost', 'motionheuristic', 'hindsight']
}

COLOR_MAP = {
    'btom': '#E63946',             
    'truebelief': '#F5A623',       
    'nocost': "#DC91EB",           
    'motionheuristic': "#B8969A",
    'hindsight': "#747ED2"   
}

# PATH_GROUPS = {
#     'NoCheck': [3, 5],
#     'Check-Stay': [2],
#     'Check-Partial': [6, 7],
#     'Check-GoBack': [1, 4]
# }

# 💡 [NEW] Overall 통합 플롯을 위해 전체 그룹 인덱스 정의
ALL_GROUPS = [1, 2, 3, 4, 5, 6, 7]

TARGETS = {
    'Desire': ['desire_K', 'desire_L', 'desire_M'], 
    'Belief': ['belief_L', 'belief_M', 'belief_Empty']
}

# =========================================================================
# 🌟 [Step 1] 데이터 로드 및 병합 로직
# =========================================================================
def load_llm_everystep(llm_name, condition):
    csv_path = os.path.join(BASE_RESULTS_DIR, llm_name, condition, "everystep", "everystep_valid_only.csv")
    if not os.path.exists(csv_path): return None
    
    df = pd.read_csv(csv_path)
    score_cols = TARGETS['Desire'] + TARGETS['Belief']

    # 💡 phase를 포함하여 그룹화하고, phase_index를 컬럼으로 추가
    if 'phase' not in df.columns:
        print(f"⚠️ Warning: 'phase' column missing in {llm_name} data. Cumulative logic requires this.")
        return None

    df_mean = df.groupby(['scenario_id', 'group_id', 'phase', 'time_step'])[score_cols].mean().reset_index()
    df_mean['phase_idx'] = df_mean.apply(lambda row: get_phase_index(row['group_id'], row['phase']), axis=1)
    
    # Unknown Phase 제거
    df_mean = df_mean[df_mean['phase_idx'] != 9]

    # Belief 정규화
    belief_cols = TARGETS['Belief']
    df_mean_shifted = np.maximum(df_mean[belief_cols] - 1, 0)
    bel_sum = df_mean_shifted.sum(axis=1).replace(0, 1.0)
    df_mean[belief_cols] = df_mean_shifted.div(bel_sum, axis=0)
    
    return df_mean

def build_regression_df(df_target, ref_dfs, target_cols, current_refs):
    """특정 Path Group의 데이터를 Melt하여 하나의 평탄화(Flatten)된 회귀 데이터프레임으로 변환"""
    # df_llm_path = df_llm[df_llm['group_id'].isin(group_ids)]
    if len(df_target) == 0: return None

    df_reg = df_target.melt(id_vars=['scenario_id', 'time_step'], value_vars=target_cols, 
                              var_name='target_type', value_name='LLM')
    
    # 💡 전체가 아닌 current_refs(4개 또는 5개)만 병합
    for ref in current_refs:
        if ref in ref_dfs:
            # 💡 [핵심] 레퍼런스 모델도 df_target과 동일한(누적된) scenario_id 및 time_step만 조인됨
            df_ref_melt = ref_dfs[ref].melt(id_vars=['scenario_id', 'time_step'], value_vars=target_cols, 
                                           var_name='target_type', value_name=ref)
            df_reg = pd.merge(df_reg, df_ref_melt, on=['scenario_id', 'time_step', 'target_type'], how='left')

    df_reg_clean = df_reg.drop(columns=['scenario_id', 'time_step', 'target_type']).dropna()
    if len(df_reg_clean) < 10: return None # 데이터가 너무 적으면 회귀 불가

    # 상수 컬럼(Variance == 0) 제거 (회귀분석 에러 방지)
    constant_cols = [col for col in df_reg_clean.columns if df_reg_clean[col].nunique() <= 1]
    if constant_cols: df_reg_clean.drop(columns=constant_cols, inplace=True)

    if 'btom' not in df_reg_clean.columns or 'LLM' not in df_reg_clean.columns: return None

    return df_reg_clean.apply(zscore)

# =========================================================================
# 🌟 [Step 2] 순수 고유 설명력(Unique R²) 계산
# =========================================================================
def calculate_unique_vpa(df_reg_std, target_col, predictors):
    try:
        X_all = sm.add_constant(df_reg_std[predictors])
        r2_total = max(0, sm.OLS(df_reg_std[target_col], X_all).fit().rsquared_adj)
        
        unique_vars = {}
        for var in predictors:
            others = [p for p in predictors if p != var]
            if not others: continue
            X_others = sm.add_constant(df_reg_std[others])
            r2_others = max(0, sm.OLS(df_reg_std[target_col], X_others).fit().rsquared_adj)
            unique_vars[var] = max(0, r2_total - r2_others)
            
        return {'total_r2': r2_total, 'unique': unique_vars}
    except Exception as e:
        return {'total_r2': 0, 'unique': {}}

# =========================================================================
# 🌟 [Step 3] 1x2 통합 VPA 시각화 (동적 오프셋 적용)
# =========================================================================
def draw_stepwise_overall_vpa(master_results, condition, save_dir):
    """Ribbon 값은 따로 저장해두고, 전체 데이터를 합쳐서 보여주는 메인 플롯"""
    n_llms = len(LLM_MODELS)
    
    # 캔버스 크기: 가로는 그룹 수에 비례, 세로는 LLM 수에 비례
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 2 * n_llms), sharey=True)
    plt.subplots_adjust(wspace=0.1) 
    ax_des, ax_bel = axes[0], axes[1]
    
    y_positions = np.arange(n_llms)[::-1]

    bar_height = 0.12
    bar_spacing = 0.14

    # Desire와 Belief의 컬럼별 매핑
    for metric, ax in zip(['Desire', 'Belief'], [ax_des, ax_bel]):
        draw_order = PREDICTORS[metric]
        n_refs = len(draw_order)
        
        # 💡 [핵심] 4개 vs 5개 막대에 대한 동적 위치 계산
        start_offset = (n_refs - 1) / 2.0
        offsets = [(start_offset - i) * bar_spacing for i in range(n_refs)]
        
        for row_idx, llm in enumerate(LLM_MODELS):
            base_y = y_positions[row_idx]
            
            # 'Overall' 그룹의 데이터를 가져옵니다.
            if llm not in master_results or 'Overall' not in master_results[llm]:
                ax.text(0.1, base_y, "Data Missing", color='gray', style='italic')
                continue
                
            vpa_res = master_results[llm]['Overall'][metric]['unique']
            if not vpa_res: continue
            
            max_val = max([vpa_res.get(ref, 0) for ref in draw_order])
            
            for i, ref in enumerate(draw_order):
                val = vpa_res.get(ref, 0)
                is_max = (val == max_val and val > 0)
                
                # 💡 [NEW] 1등 막대에 검은색 굵은 하이라이트
                edge_c = 'black' if is_max else 'white'
                lw = 1.8 if is_max else 0.5
                
                ax.barh(base_y + offsets[i], val, height=bar_height, 
                        color=COLOR_MAP[ref], edgecolor=edge_c, 
                        linewidth=lw, alpha=0.9)
                        
                if val >= 0.01:
                    f_weight = 'bold' if is_max else 'normal'
                    f_size = 11 if is_max else 9
                    ax.text(val + 0.01, base_y + offsets[i], f"{val*100:.1f}%", 
                            va='center', fontsize=f_size, fontweight=f_weight, color='black')

        # 🌟 구분선 및 스타일링 (축은 각 패싯마다 설정)
        separator_idx = len(WORSE_MODELS) - 0.5
        sep_y = y_positions[0] - separator_idx

        for ax, title in zip([ax_des, ax_bel], ["Desire: Unique Variance", "Belief: Unique Variance"]):
            ax.axhline(sep_y, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
            
            if ax == ax_des:
                ax.set_yticks(y_positions)
                ax.set_yticklabels(LLM_MODELS, fontsize=13, fontweight='bold')
                
            ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
            ax.set_xlim(0, 0.6)
            ax.set_xlabel("Adjusted $R^2$", fontsize=12, fontweight='bold')
            ax.grid(axis='x', linestyle='--', alpha=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLOR_MAP['btom'], markersize=14, label='BToM'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLOR_MAP['truebelief'], markersize=14, label='TrueBelief'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLOR_MAP['nocost'], markersize=14, label='NoCost'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLOR_MAP['motionheuristic'], markersize=14, label='MotionHeur'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLOR_MAP['hindsight'], markersize=14, label='Hindsight')
    ]

    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.01), ncol=4, fontsize=12)
    fig.suptitle(f"Everystep VPA: Unique Cognitive Model Contributions (Cond: {condition})", 
                 fontsize=22, fontweight='bold', y=1.07)
    
    plt.tight_layout(w_pad=5)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"vpa_everystep_{condition}.png")
    plt.savefig(save_path, dpi=250, bbox_inches='tight')
    plt.close()
    print(f"\n📊 Overall Everystep VPA plot saved to: {save_path}")

# =========================================================================
# 🌟 [Main] 분석 실행 및 데이터 저장
# =========================================================================
def run_stepwise_vpa(condition):
    print("\n" + "="*90)
    print(f"🚀 Running Stepwise VPA Analysis [ALL MODELS | {condition}]")
    print("="*90)

    # 레퍼런스 모델 Everystep 데이터 로드
    ref_dfs = {}
    for ref in ALL_REFS:
        df_ref = load_reference_everystep(ref)
        if df_ref is not None:
            ref_dfs[ref] = df_ref

    master_results = {}

    for llm_name in LLM_MODELS:
        df_llm = load_llm_everystep(llm_name, condition)
        if df_llm is None: continue
        
        print(f"\n⏳ Processing: {llm_name}...")
        
        # 💡 [NEW 구조] Overall 결과와 Cumulative 결과를 분리하여 저장
        master_results[llm_name] = {'Overall': {}, 'Cumulative': {}}

        # ---------------------------------------------------------
        # 1. Overall VPA 계산 (전체 시나리오, 전체 타임스텝 통짜 병합)
        # ---------------------------------------------------------    
        for cat_name, target_cols in TARGETS.items():
            current_refs = PREDICTORS[cat_name]
            df_reg_std = build_regression_df(df_llm, ref_dfs, target_cols, current_refs)
                
            if df_reg_std is not None:
                available_refs = [r for r in current_refs if r in df_reg_std.columns]
                master_results[llm_name]['Overall'][cat_name] = calculate_unique_vpa(df_reg_std, 'LLM', available_refs)
            else:
                master_results[llm_name]['Overall'][cat_name] = {'unique': {}}

        # ---------------------------------------------------------
        # 2. 💡 [핵심] Cumulative VPA 계산 (그룹별 -> Phase 누적별)
        # ---------------------------------------------------------
        for group_id in ALL_GROUPS:
            master_results[llm_name]['Cumulative'][group_id] = {}
            
            # 해당 그룹의 데이터만 추출
            df_group = df_llm[df_llm['group_id'] == group_id]
            if len(df_group) == 0: continue
            
            # 해당 그룹이 도달한 최대 Phase 인덱스 파악
            max_phase = int(df_group['phase_idx'].max())
            
            # 0부터 max_phase까지 횡단하며 데이터를 누적
            for current_phase in range(max_phase + 1):
                master_results[llm_name]['Cumulative'][group_id][current_phase] = {}
                
                # 핵심: 0부터 현재 Phase까지의 데이터를 누적(Slice)
                df_accumulated = df_group[df_group['phase_idx'] <= current_phase]
                
                for cat_name, target_cols in TARGETS.items():
                    current_refs = PREDICTORS[cat_name]
                    df_reg_std = build_regression_df(df_accumulated, ref_dfs, target_cols, current_refs)
                    
                    if df_reg_std is not None:
                        available_refs = [r for r in current_refs if r in df_reg_std.columns]
                        vpa_res = calculate_unique_vpa(df_reg_std, 'LLM', available_refs)
                    else:
                        vpa_res = {'unique': {}}
                        
                    master_results[llm_name]['Cumulative'][group_id][current_phase][cat_name] = vpa_res

    plot_save_dir = os.path.join(parent_dir, "results")

    # 1. Overall 데이터를 이용해 메인 플롯 그리기
    draw_stepwise_overall_vpa(master_results, condition, plot_save_dir)
    
    # 2. 전체 연산 결과를 Pickle로 저장 (plot_everystep.py 에서 꺼내 쓰기 위함)
    pkl_save_path = os.path.join(plot_save_dir, f"everystep_cumulative_vpa_results_{condition}.pkl")
    with open(pkl_save_path, 'wb') as f:
        pickle.dump(master_results, f)
    print(f"💾 All Stepwise VPA results successfully saved to: {pkl_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stepwise VPA Analysis.")
    parser.add_argument("--condition", type=str, default="vanilla", help="Experiment condition")
    args = parser.parse_args()
    
    run_stepwise_vpa(args.condition)