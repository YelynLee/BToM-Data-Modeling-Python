import os
import sys
import pickle
import scipy.io
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# 1. 현재 스크립트(analysis 폴더)의 상위 경로를 파이썬 탐색 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) # 상위 폴더 (프로젝트 루트)

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.config import REFERENCE_MAT_PATH, BASE_RESULTS_DIR, HUMAN_PKL_PATH

# 평가할 모델 및 조건 리스트 정의
COG_MODELS = ['btom', 'truebelief', 'nocost', 'motionheuristic', 'hindsight']
LLM_MODELS = ['gpt-4o', 'o4-mini', 'gpt-5.4', 'gemini-2.5-flash', 'gemini-2.5-pro',
              'deepseek-chat', 'deepseek-reasoner', 'claude-sonnet-4-6', 'claude-opus-4-6']
CONDITIONS = ['vanilla', 'oneshot']

# BToM 원본 타겟 파일 경로 (모든 Beta를 순회하기 위해 MAT 유지)
TARGET_BTOM_MAT = os.path.join(REFERENCE_MAT_PATH, "btom_results_complete.mat")

def safe_pearsonr(x, y):
    """NaN이나 분산이 0인 상수가 섞여 있을 때의 상관계수 계산 에러를 방지합니다."""
    valid = ~np.isnan(x) & ~np.isnan(y)
    if np.sum(valid) < 2:
        return np.nan
    if np.std(x[valid]) == 0 or np.std(y[valid]) == 0:
        return np.nan
    r, _ = pearsonr(x[valid], y[valid])
    return r

def compare_predictions(pred_des, pred_bel, target_des, target_bel, target_betas=None):
    """
    하나의 예측값을 타겟(Human PKL 또는 BToM MAT 행렬)과 비교합니다.
    target_betas가 주어지면(3차원), 가장 상관계수가 높은 beta를 찾습니다.
    """
    if target_des.ndim == 3 and target_betas is not None: 
        max_r = -1.0
        best_beta = np.nan
        best_r_des, best_r_bel = np.nan, np.nan
        
        for idx, beta in enumerate(target_betas):
            t_des_flat = target_des[:, :, idx].flatten()
            t_bel_flat = target_bel[:, :, idx].flatten()
            
            r_des = safe_pearsonr(pred_des, t_des_flat)
            r_bel = safe_pearsonr(pred_bel, t_bel_flat)
            avg_r = (np.nan_to_num(r_des) + np.nan_to_num(r_bel)) / 2
            
            if avg_r > max_r:
                max_r = avg_r
                best_beta = beta
                best_r_des = r_des
                best_r_bel = r_bel
        return best_beta, best_r_des, best_r_bel, max_r
    else: 
        r_des = safe_pearsonr(pred_des, target_des.flatten())
        r_bel = safe_pearsonr(pred_bel, target_bel.flatten())
        avg_r = (np.nan_to_num(r_des) + np.nan_to_num(r_bel)) / 2
        return np.nan, r_des, r_bel, avg_r

def run_all_evaluations():
    print("🚀 모든 모델에 대한 Best Fit 분석을 시작합니다...\n")
    results_list = []

    # 1. BToM 데이터 로드 (MAT 파일 - 20개의 Beta 값을 모두 얻기 위함)
    btom_loaded = False
    if os.path.exists(TARGET_BTOM_MAT):
        btom_mat = scipy.io.loadmat(TARGET_BTOM_MAT, squeeze_me=True, struct_as_record=False)
        btom_des_all = btom_mat['desire_model']
        btom_bel_all = btom_mat['belief_model']
        btom_betas = np.atleast_1d(btom_mat.get('beta_score_values', []))
        btom_loaded = True
    else:
        print(f"⚠️ 경고: BToM MAT 파일이 없습니다 ({TARGET_BTOM_MAT}).")

    # 2. Human 데이터 로드 (PKL 파일 - data_processor.py 결과물 활용)
    human_loaded = False
    if os.path.exists(HUMAN_PKL_PATH):
        with open(HUMAN_PKL_PATH, 'rb') as f:
            human_data = pickle.load(f)
        human_des = human_data['des_inf_mean']
        human_bel = human_data['bel_inf_mean_norm']
        human_loaded = True
        print(f"✅ Human 데이터 로드 완료 (PKL): {HUMAN_PKL_PATH}")
    else:
        print(f"⚠️ 경고: Human PKL 데이터가 없습니다 ({HUMAN_PKL_PATH}).")
        human_des, human_bel = np.full((3, 78), np.nan), np.full((3, 78), np.nan)

    # ==========================================
    # A. 인지 모델(Cognitive Models) 평가 (MAT 사용)
    # ==========================================
    for model in COG_MODELS:
        row = {
            'Model': model, 'Condition': 'N/A', 'Type': 'Cognitive Model',
            'Best_Beta_vs_Human': np.nan, 'Human_r_Avg': np.nan,
            'Best_Beta_vs_BToM': np.nan, 'BToM_r_Avg': np.nan,
            'BToM_r_at_Beta2.5': np.nan  # 새롭게 추가된 기준점 컬럼
        }
        
        mat_path = os.path.join(REFERENCE_MAT_PATH, f"{model}_results_complete.mat")
        if os.path.exists(mat_path):
            mat_data = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
            cog_des = mat_data.get('desire_model', np.full((3, 78), np.nan))
            cog_bel = mat_data.get('belief_model', np.full((3, 78), np.nan))
            
            # MotionHeuristic은 beta 값이 없을 수 있음
            cog_betas = mat_data.get('beta_score_values', None)
            if cog_betas is not None:
                cog_betas = np.atleast_1d(cog_betas)
            
            if human_loaded:
                best_h_beta, _, _, h_avg_r = compare_predictions(
                    human_des.flatten(), human_bel.flatten(), cog_des, cog_bel, cog_betas)
                row['Best_Beta_vs_Human'] = best_h_beta
                row['Human_r_Avg'] = h_avg_r
            
            if model == 'btom':
                row['Best_Beta_vs_BToM'] = 1.0 
                row['BToM_r_Avg'] = 1.0
                row['BToM_r_at_Beta2.5'] = 1.0
        
        results_list.append(row)

    # ==========================================
    # B. 대규모 언어 모델(LLMs) 평가 (PKL 사용)
    # ==========================================
    for model in LLM_MODELS:
        for cond in CONDITIONS:
            row = {
                'Model': model, 'Condition': cond, 'Type': 'LLM',
                'Best_Beta_vs_Human': np.nan, 'Human_r_Avg': np.nan,
                'Best_Beta_vs_BToM': np.nan, 'BToM_r_Avg': np.nan,
                'BToM_r_at_Beta2.5': np.nan  # 새롭게 추가된 기준점 컬럼
            }
            
            llm_pkl_path = os.path.join(BASE_RESULTS_DIR, model, cond, "model_data.pkl")
            if os.path.exists(llm_pkl_path):
                with open(llm_pkl_path, 'rb') as f:
                    llm_data = pickle.load(f)
                
                llm_des = llm_data['des_inf_mean'].flatten()
                llm_bel = llm_data['bel_inf_mean_norm'].flatten()

                # 1. vs Human (PKL 데이터간 비교)
                if human_loaded:
                    _, _, _, h_avg_r = compare_predictions(llm_des, llm_bel, human_des, human_bel)
                    row['Human_r_Avg'] = h_avg_r

                # 2. vs BToM (LLM PKL 데이터와 BToM MAT 데이터간 비교)
                if btom_loaded:
                    # 최고 상관계수 탐색
                    best_b_beta, _, _, b_avg_r = compare_predictions(
                        llm_des, llm_bel, btom_des_all, btom_bel_all, btom_betas)
                    row['Best_Beta_vs_BToM'] = best_b_beta
                    row['BToM_r_Avg'] = b_avg_r

                    # Beta=2.5 에서의 상관계수를 강제로 추출
                    target_beta = 2.5
                    idx_2_5 = np.argmin(np.abs(btom_betas - target_beta))
                    
                    btom_des_2_5 = btom_des_all[:, :, idx_2_5].flatten()
                    btom_bel_2_5 = btom_bel_all[:, :, idx_2_5].flatten()
                    
                    r_des_2_5 = safe_pearsonr(llm_des, btom_des_2_5)
                    r_bel_2_5 = safe_pearsonr(llm_bel, btom_bel_2_5)
                    r_avg_2_5 = (np.nan_to_num(r_des_2_5) + np.nan_to_num(r_bel_2_5)) / 2
                    
                    row['BToM_r_at_Beta2.5'] = r_avg_2_5
                    
            results_list.append(row)

    # ==========================================
    # C. 결과를 DataFrame으로 변환 및 CSV 저장
    # ==========================================
    df = pd.DataFrame(results_list)
    df = df.round(4) # 소수점 4자리까지 표기
    
    csv_path = os.path.join(parent_dir, "results", "model_beta_evaluation.csv")
    df.to_csv(csv_path, index=False, na_rep='NaN')
    
    print("\n✅ 모든 모델 분석이 완료되었습니다!")
    print(f"📊 결과가 저장된 경로: {csv_path}\n")
    print(df.dropna(subset=['BToM_r_Avg', 'Human_r_Avg'], how='all').to_string(index=False))

if __name__ == "__main__":
    run_all_evaluations()