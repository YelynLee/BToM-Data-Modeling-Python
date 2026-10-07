import os
import sys
import argparse
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings("ignore")

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.config import BASE_RESULTS_DIR, BEHAVIOR_GROUPS

BETTER_MODELS = ['gemini-2.5-pro', 'deepseek-reasoner', 'deepseek-v4-pro', 'claude-opus-4-6']

def load_normal_data(model_name, condition):
    """End-step (Normal) 데이터 로드: model_data.pkl에서 시나리오별 평균값 추출"""
    pkl_path = os.path.join(BASE_RESULTS_DIR, model_name, condition, "model_data.pkl")
    if not os.path.exists(pkl_path):
        print(f"❌ Error: Normal mode data not found at {pkl_path}")
        return None, None

    normal_data = pd.read_pickle(pkl_path)
    
    # shape: (3, 78) -> 열(Column)이 scenario_id (0-based index)
    des_mean = normal_data.get('des_inf_mean')
    bel_mean = normal_data.get('bel_inf_mean_norm')
    
    return des_mean, bel_mean

def load_everystep_final_data(model_name, condition):
    """Every-step 데이터 로드: 각 시나리오의 '마지막 time_step' 값만 추출"""
    csv_path = os.path.join(BASE_RESULTS_DIR, model_name, condition, "everystep", "everystep_valid_only.csv")
    if not os.path.exists(csv_path):
        print(f"❌ Error: Everystep data not found at {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    
    # 피험자 전체의 평균을 구함 (scenario_id, group_id, time_step 기준)
    score_cols = ['desire_K', 'desire_L', 'desire_M', 'belief_L', 'belief_M', 'belief_Empty']
    df_mean = df.groupby(['scenario_id', 'group_id', 'time_step'])[score_cols].mean().reset_index()

    # Belief 정규화 (1~7 -> 0~6 -> 합계로 나누기)
    belief_cols = ['belief_L', 'belief_M', 'belief_Empty']
    df_mean_shifted = np.maximum(df_mean[belief_cols] - 1, 0)
    bel_sum = df_mean_shifted.sum(axis=1).replace(0, 1.0)
    df_mean[belief_cols] = df_mean_shifted.div(bel_sum, axis=0)

    # 각 시나리오별로 time_step이 가장 큰(마지막) 행만 추출
    # sort_values 후 drop_duplicates(keep='last') 사용
    df_final_step = df_mean.sort_values('time_step').drop_duplicates(subset=['scenario_id', 'group_id'], keep='last')
    
    return df_final_step.sort_values('scenario_id').reset_index(drop=True)

def run_correlation_analysis(model_name, condition):
    print("\n" + "="*80)
    print(f"📊 End-step vs Every-step Final Response Correlation | Model: [{model_name}]")
    print("="*80)

    # 매칭 및 상관계수 계산을 위한 리스트 초기화
    groups = range(1, 8)
    desire_cols = ['desire_K', 'desire_L', 'desire_M']
    belief_cols = ['belief_L', 'belief_M', 'belief_Empty']

    # =========================================================================
    # [Target Model Analysis] - Section 1 & 2
    # =========================================================================

    # 1. 데이터 로드
    norm_des, norm_bel = load_normal_data(model_name, condition)
    df_every_final = load_everystep_final_data(model_name, condition)

    if norm_des is None or df_every_final is None:
        return

    # 2. 매칭 및 상관계수 계산을 위한 리스트 초기화
    groups = range(1, 8)
    desire_cols = ['desire_K', 'desire_L', 'desire_M']
    belief_cols = ['belief_L', 'belief_M', 'belief_Empty']

    all_norm_des, all_every_des = [], []
    all_norm_bel, all_every_bel = [], []

    print("\n[Section 1] Correlation by Path Group")
    print("-" * 60)
    print(f"{'Group ID':<15} | {'Desire Correlation (r)':<20} | {'Belief Correlation (r)':<20}")
    print("-" * 60)

    for g_id in groups:
        # 현재 그룹에 속하는 시나리오 ID 추출
        group_df = df_every_final[df_every_final['group_id'] == g_id]
        if group_df.empty:
            continue
            
        scenarios = group_df['scenario_id'].values
        
        grp_norm_des, gpr_every_des = [], []
        grp_norm_bel, gpr_every_bel = [], []

        for sc_id in scenarios:
            # sc_id는 1-based, norm_des는 0-based index
            n_d = norm_des[:, sc_id - 1]
            n_b = norm_bel[:, sc_id - 1]
            
            # Everystep의 마지막 값 추출
            sc_row = group_df[group_df['scenario_id'] == sc_id].iloc[0]
            e_d = sc_row[desire_cols].values.astype(float)
            e_b = sc_row[belief_cols].values.astype(float)

            # 그룹 단위 배열에 추가
            grp_norm_des.extend(n_d)
            gpr_every_des.extend(e_d)
            grp_norm_bel.extend(n_b)
            gpr_every_bel.extend(e_b)

            # 전체 단위 배열에 추가
            all_norm_des.extend(n_d)
            all_every_des.extend(e_d)
            all_norm_bel.extend(n_b)
            all_every_bel.extend(e_b)

        # 💡 그룹별 상관계수 계산 (Pearson r)
        r_des = pearsonr(grp_norm_des, gpr_every_des)[0] if len(grp_norm_des) > 1 else np.nan
        r_bel = pearsonr(grp_norm_bel, gpr_every_bel)[0] if len(grp_norm_bel) > 1 else np.nan
        
        group_name = BEHAVIOR_GROUPS.get(g_id, f"Group {g_id}")
        print(f"{group_name:<15} | r = {r_des:.4f}             | r = {r_bel:.4f}")

    print("-" * 60)

    # 3. 전체 시나리오 통틀어서 상관계수 계산
    r_des_all = pearsonr(all_norm_des, all_every_des)[0] if len(all_norm_des) > 1 else np.nan
    r_bel_all = pearsonr(all_norm_bel, all_every_bel)[0] if len(all_norm_bel) > 1 else np.nan

    print("\n[Section 2] Overall Correlation (All Scenarios)")
    print("-" * 60)
    print(f"{'Overall (N=78)':<15} | r = {r_des_all:.4f}             | r = {r_bel_all:.4f}")

    # =========================================================================
    # [Better Models Aggregate Analysis] - Section 3 & 4
    # =========================================================================
    agg_group_data = {g: {'n_d': [], 'e_d': [], 'n_b': [], 'e_b': []} for g in groups}
    agg_all_n_d, agg_all_e_d, agg_all_n_b, agg_all_e_b = [], [], [], []
    valid_models_count = 0

    for b_model in BETTER_MODELS:
        n_d_mat, n_b_mat = load_normal_data(b_model, condition)
        df_ev = load_everystep_final_data(b_model, condition)
        
        if n_d_mat is None or df_ev is None:
            continue
            
        valid_models_count += 1

        for g_id in groups:
            group_df = df_ev[df_ev['group_id'] == g_id]
            if group_df.empty: continue

            for sc_id in group_df['scenario_id'].values:
                n_d = n_d_mat[:, sc_id - 1]
                n_b = n_b_mat[:, sc_id - 1]

                sc_row = group_df[group_df['scenario_id'] == sc_id].iloc[0]
                e_d = sc_row[desire_cols].values.astype(float)
                e_b = sc_row[belief_cols].values.astype(float)

                agg_group_data[g_id]['n_d'].extend(n_d)
                agg_group_data[g_id]['e_d'].extend(e_d)
                agg_group_data[g_id]['n_b'].extend(n_b)
                agg_group_data[g_id]['e_b'].extend(e_b)

                agg_all_n_d.extend(n_d)
                agg_all_e_d.extend(e_d)
                agg_all_n_b.extend(n_b)
                agg_all_e_b.extend(e_b)

    if valid_models_count > 0:
        print(f"\n\n[Section 3] Correlation by Path Group (Aggregate: {valid_models_count} Better Models)")
        print("-" * 65)
        print(f"{'Group ID':<18} | {'Desire Correlation (r)':<20} | {'Belief Correlation (r)':<20}")
        print("-" * 65)

        for g_id in groups:
            grp_data = agg_group_data[g_id]
            if len(grp_data['n_d']) < 2:
                continue

            r_des = pearsonr(grp_data['n_d'], grp_data['e_d'])[0]
            r_bel = pearsonr(grp_data['n_b'], grp_data['e_b'])[0]
            
            group_name = BEHAVIOR_GROUPS.get(g_id, f"Group {g_id}")
            print(f"{group_name:<18} | r = {r_des:.4f}             | r = {r_bel:.4f}")

        print("-" * 65)

        if len(agg_all_n_d) > 1:
            r_des_all = pearsonr(agg_all_n_d, agg_all_e_d)[0]
            r_bel_all = pearsonr(agg_all_n_b, agg_all_e_b)[0]

            print(f"\n[Section 4] Overall Correlation (Aggregate: {valid_models_count} Better Models)")
            print("-" * 65)
            print(f"{'Overall Aggregate':<18} | r = {r_des_all:.4f}             | r = {r_bel_all:.4f}")
    else:
        print("\n⚠️ Warning: No valid Better Models data found. Skipping Section 3 & 4.")

    print("=" * 85 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate correlation between Normal and Everystep final responses.")
    parser.add_argument("--model", type=str, required=True, help="Target model (e.g., claude-opus-4-6)")
    parser.add_argument("--condition", type=str, default="vanilla", help="Experiment condition")
    
    args = parser.parse_args()
    
    run_correlation_analysis(args.model, args.condition)