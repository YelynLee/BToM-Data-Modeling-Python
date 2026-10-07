import os
import sys
import argparse
import pickle
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import zscore
import warnings

warnings.filterwarnings("ignore")

# 1. 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.config import REFERENCE_PKL_DIR, BASE_RESULTS_DIR, get_group_indices

# 그룹 정의
GROUPS = {
    "BETTER_MODELS": ['gemini-2.5-pro', 'deepseek-reasoner', 'claude-opus-4-6'],
    "WORSE_MODELS": ['gpt-5.4', 'o4-mini', 'gemini-2.5-flash', 'claude-sonnet-4-6']
}

LESION_MODELS = ['truebelief', 'nocost', 'hindsight']
ALL_REFS = ['btom'] + LESION_MODELS

TARGET_MAPPING = {
    'Desire': {0: 'Target K', 1: 'Target L', 2: 'Target M'},
    'Belief': {0: 'Target L', 1: 'Target M', 2: 'Target Empty(N)'}
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

def get_averaged_llm_data(model_list, condition):
    """지정된 모델 리스트의 데이터를 불러와 산술 평균을 계산"""
    des_list = []
    bel_list = []
    loaded_models = []
    
    for m in model_list:
        path = os.path.join(BASE_RESULTS_DIR, m, condition, "model_data.pkl")
        data = load_pickle_safe(path)
        if data is not None:
            des_list.append(data['des_inf_mean'])
            bel_list.append(data['bel_inf_mean_norm'])
            loaded_models.append(m)
        else:
            print(f"      ⚠️ Warning: Data missing for {m}. Excluded from average.")
            
    if not loaded_models:
        return None, []
        
    avg_data = {
        # axis=0 기준으로 평균 (3, 78) 행렬 유지
        'des_inf_mean': np.mean(des_list, axis=0),
        'bel_inf_mean_norm': np.mean(bel_list, axis=0)
    }
    return avg_data, loaded_models

def extract_target_data(data_dict, valid_mask, key, target_idx):
    target_data = data_dict[key][target_idx, :]
    return target_data[valid_mask]

def run_grouped_target_regression(condition, exclude_partial):
    mode_text = "EXCL. PARTIAL" if exclude_partial else "INCL. PARTIAL"
    valid_mask = get_valid_indices(exclude_partial=exclude_partial)
    
    print("\n" + "="*90)
    print(f"👥 Group-Averaged Target-Specific Regression: [{condition.upper()} | {mode_text}]")
    print("="*90)

    # Reference 데이터 로드
    ref_data_dict = {}
    for ref in ALL_REFS:
        path = os.path.join(REFERENCE_PKL_DIR, ref, f"{ref}_data.pkl")
        data = load_pickle_safe(path)
        if data is not None:
            ref_data_dict[ref] = data

    target_keys = {'Desire': 'des_inf_mean', 'Belief': 'bel_inf_mean_norm'}

    # 💡 그룹 단위로 순회 (BETTER -> WORSE)
    for group_name, model_list in GROUPS.items():
        print(f"\n\n{'#'*80}")
        print(f"🟢 ANALYZING GROUP: {group_name}")
        print(f"{'#'*80}")
        
        avg_llm_data, loaded_models = get_averaged_llm_data(model_list, condition)
        
        if avg_llm_data is None:
            print(f"❌ Error: No valid data found for group {group_name}")
            continue
            
        print(f"   ✓ Averaged across {len(loaded_models)} models: {', '.join(loaded_models)}")

        for cat_name, key in target_keys.items():
            print(f"\n\n{'='*30} [ {cat_name.upper()} ] {'='*30}")
            
            for target_idx, target_name in TARGET_MAPPING[cat_name].items():
                print(f"\n🔍 Analyzing: {target_name}")
                print("-" * 50)
                
                Y_llm = extract_target_data(avg_llm_data, valid_mask, key, target_idx)
                df_reg = pd.DataFrame({'LLM_AVG': Y_llm})
                
                for ref in ALL_REFS:
                    if ref in ref_data_dict:
                        df_reg[ref] = extract_target_data(ref_data_dict[ref], valid_mask, key, target_idx)

                df_reg.dropna(inplace=True)
                if len(df_reg) == 0: continue

                constant_cols = [col for col in df_reg.columns if df_reg[col].nunique() <= 1]
                if constant_cols:
                    df_reg.drop(columns=constant_cols, inplace=True)

                if 'btom' not in df_reg.columns: continue

                df_reg_std = df_reg.apply(zscore)
                available_refs = [r for r in ALL_REFS if r in df_reg_std.columns]
                
                X_all = df_reg_std[available_refs]
                X_all_with_const = sm.add_constant(X_all)
                
                # 회귀분석 수행
                X_base = sm.add_constant(df_reg_std[['btom']])
                model_1 = sm.OLS(df_reg_std['LLM_AVG'], X_base).fit()
                model_2 = sm.OLS(df_reg_std['LLM_AVG'], X_all_with_const).fit()
                
                delta_adj_r2 = model_2.rsquared_adj - model_1.rsquared_adj
                
                print(f"   ▶ Baseline (BToM Only) Adj. R² : {model_1.rsquared_adj:.4f}")
                print(f"   ▶ Full Model (Lesions) Adj. R² : {model_2.rsquared_adj:.4f}  (Δ Adj. R² = {delta_adj_r2:.4f})")
                
                print(f"\n   [ Coefficients for {target_name} ]")
                print(f"   {'Predictor':<15} | {'Beta (β)':<10} | {'p-value':<10}")
                
                for var in available_refs:
                    coef = model_2.params[var]
                    pval = model_2.pvalues[var]
                    stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "n.s."
                    
                    highlight = "💡" if pval < 0.05 and coef > 0.05 and var != 'btom' else "  "
                    print(f" {highlight} {var:<13} | {coef:>8.4f} {stars:<4} | {pval:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Group-Averaged Target-Specific Regression.")
    parser.add_argument("--condition", type=str, default="vanilla", help="Experiment condition")
    parser.add_argument("--exclude_partial", action="store_true", help="Exclude Check-Partial groups")

    args = parser.parse_args()
    run_grouped_target_regression(args.condition, args.exclude_partial)