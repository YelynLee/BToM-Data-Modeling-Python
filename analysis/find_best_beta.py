import os
import argparse
import sys
import pickle
import scipy.io
import numpy as np
from scipy.stats import pearsonr

# 1. 현재 스크립트(analysis 폴더)의 상위 경로를 파이썬 탐색 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) # 상위 폴더 (프로젝트 루트)

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.config import BTOM_MAT_PATH, BASE_RESULTS_DIR

def find_best_beta_for_model(llm_pkl_path, btom_mat_path=BTOM_MAT_PATH):
    """
    LLM의 데이터(Pickle)와 BToM 모델 데이터(MAT)를 비교하여,
    LLM과 가장 높은 상관관계를 가지는 최적의 beta 값을 찾습니다.
    """
    if not os.path.exists(llm_pkl_path):
        print(f"❌ Error: LLM 데이터 파일이 없습니다 -> {llm_pkl_path}")
        return
    
    if not os.path.exists(btom_mat_path):
        print(f"❌ Error: BToM MAT 파일이 없습니다 -> {btom_mat_path}")
        return

    # 1. LLM 데이터 로드
    with open(llm_pkl_path, 'rb') as f:
        llm_data = pickle.load(f)
        
    # 평탄화 (Flatten) 및 NaN 제거용 마스크 준비
    # shape: (3, 78) -> (234,)
    llm_des = llm_data['des_inf_mean'].flatten()
    llm_bel = llm_data['bel_inf_mean_norm'].flatten()

    # 2. BToM 데이터 로드
    mat_data = scipy.io.loadmat(btom_mat_path, squeeze_me=True, struct_as_record=False)
    beta_values = np.atleast_1d(mat_data['beta_score_values'])
    
    print(f"\n🔍 '{os.path.basename(os.path.dirname(llm_pkl_path))}' 조건의 최적 Beta 탐색 시작...")
    print("-" * 50)
    print(f"{'Beta':<10} | {'Desire (r)':<15} | {'Belief (r)':<15} | {'Average (r)':<15}")
    print("-" * 50)

    best_beta = None
    max_avg_r = -1.0
    results = []

    # 3. 20개의 Beta 값을 순회하며 상관계수 계산
    for idx, beta in enumerate(beta_values):
        # BToM의 특정 beta 값 데이터 추출 및 평탄화
        btom_des = mat_data['desire_model'][:, :, idx].flatten()
        btom_bel = mat_data['belief_model'][:, :, idx].flatten()

        # NaN 값이 있으면 상관계수 계산 시 에러가 나므로 유효한 인덱스만 추출
        valid_des = ~np.isnan(llm_des) & ~np.isnan(btom_des)
        valid_bel = ~np.isnan(llm_bel) & ~np.isnan(btom_bel)

        # Pearson 상관계수 계산
        r_des, _ = pearsonr(llm_des[valid_des], btom_des[valid_des])
        r_bel, _ = pearsonr(llm_bel[valid_bel], btom_bel[valid_bel])
        
        avg_r = (r_des + r_bel) / 2
        results.append((beta, r_des, r_bel, avg_r))
        
        print(f"{beta:<10.1f} | {r_des:<15.4f} | {r_bel:<15.4f} | {avg_r:<15.4f}")

        # 최고 상관계수 업데이트
        if avg_r > max_avg_r:
            max_avg_r = avg_r
            best_beta = beta

    print("-" * 50)
    print(f"🏆 LLM 데이터와 가장 잘 맞는 최적의 Beta: {best_beta} (Average r = {max_avg_r:.4f})")
    
    # 인간의 베스트 Beta(2.5)와의 비교 코멘트
    if best_beta == 2.5:
        print("💡 분석: LLM은 인간과 동일한 수준의 합리성(Beta=2.5)을 타인에게 기대하고 있습니다.")
    elif best_beta > 2.5:
        print("💡 분석: LLM은 인간보다 타인을 더 기계적이고 완벽한 합리적 존재로 과대평가하고 있습니다 (Hyper-rational).")
    else:
        print("💡 분석: LLM은 타인의 행동에 인간보다 더 많은 노이즈나 비합리성이 있다고 판단하고 있습니다.")
        
    return best_beta

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the best BToM Beta score for a given LLM result")
    
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., gpt-4o, gemini-2.5-flash)")
    parser.add_argument("--condition", type=str, required=True, help="Condition (e.g., vanilla, reasoning, oneshot)")
    parser.add_argument("--mode", type=str, default="normal", choices=["normal", "everystep"], help="Analysis mode")
    
    args = parser.parse_args()
    
    # 경로 자동 조합 로직
    if args.mode == "everystep":
        llm_pkl_path = os.path.join(BASE_RESULTS_DIR, args.model, args.condition, "everystep", "model_data.pkl")
    else:
        llm_pkl_path = os.path.join(BASE_RESULTS_DIR, args.model, args.condition, "model_data.pkl")
        
    find_best_beta_for_model(llm_pkl_path)