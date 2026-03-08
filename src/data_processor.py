import os
import argparse
import glob
import pickle
import numpy as np
import pandas as pd
import scipy.io
from src.config import get_group_indices, HUMAN_MAT_PATH, HUMAN_PKL_PATH, REFERENCE_MAT_PATH, REFERENCE_PKL_DIR, BASE_RESULTS_DIR
from src.utils import inspect_pickle_data

# =============================================================================
# 1. Human Data Processing (MAT -> PKL)
# =============================================================================
def convert_human_mat_to_pickle(mat_path=HUMAN_MAT_PATH, output_path=HUMAN_PKL_PATH):
    """
    MATLAB 원본 데이터를 로드하여 Python 분석용 Pickle로 변환
    """
    # 1. MATLAB 파일 로드
    if not os.path.exists(mat_path):
        print(f"❌ Error: MATLAB file not found at {mat_path}")
        return
    
    print(f"📂 Loading MATLAB file: {mat_path}...")

    # squeeze_me=True: MATLAB의 불필요한 차원(예: [[1]])을 스칼라로 펴줌
    # struct_as_record=False: MATLAB 구조체를 Python 객체처럼 다루게 해줌
    mat_data = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    # 2. 데이터 추출 및 매핑
    # Transformer 데이터와 키 이름을 100% 동일하게 맞춥니다.

    # 논문 분석용 Group Index (Irrational 제외)
    groups_raw = get_group_indices(include_irrational=False)

    human_data = {
        # --- 3D Raw Data (Rating x Condition x Subject) ---
        'des_inf': mat_data['des_inf'],   # (3, 78, 16)
        'bel_inf': mat_data['bel_inf'],   # (3, 78, 16)
        
        # --- Scenario Statistics (Rating x Condition) ---
        'des_inf_mean': mat_data['des_inf_mean'],           # (3, 78)
        'bel_inf_mean_norm': mat_data['bel_inf_mean_norm'], # (3, 78) - 정규화된 값
        
        'des_inf_se': mat_data['des_inf_se'],               # (3, 78)
        'bel_inf_se': mat_data['bel_inf_se'],               # (3, 78)
        
        # --- Group Statistics (Rating x Group) ---
        'des_inf_group_mean': mat_data['des_inf_group_mean'], # (3, 7)
        'bel_inf_group_mean': mat_data['bel_inf_group_mean'], # (3, 7)
        
        'des_inf_group_sd': mat_data['des_inf_group_se'],     # (3, 7)
        'bel_inf_group_sd': mat_data['bel_inf_group_se'],     # (3, 7)
        
        # --- Meta Info ---
        'group_mapping': groups_raw
    }

    # 3. Pickle로 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(human_data, f)

    print(f"✅ Human Data Conversion Complete! Saved to: {output_path}")


# =============================================================================
# 1.5 Reference Models Processing (MAT -> PKL)
# handles BToM, NoCost, TrueBelief, and MotionHeuristic
# =============================================================================
def convert_reference_model_to_pickle(mat_path, output_path, model_type="btom", 
                               target_beta=2.5, n_dummy_subj=16):
    """
    MATLAB/R 결과를 로드하여, 특정 Beta 값의 결과를 Python 분석용 Pickle로 변환.
    기존 Human/LLM 플롯 코드와의 호환성을 위해 동일한 데이터를 n_dummy_subj 만큼 복제함.
    model_type이 'motionheuristic'일 경우 beta 인덱싱 과정을 생략함.
    """
    if not os.path.exists(mat_path):
        print(f"❌ Error: BToM MATLAB file not found at {mat_path}")
        return
    
    print(f"📂 Loading {model_type.upper()} MAT file: {mat_path}...")
    mat_data = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    # 1. 데이터 추출 및 차원 맞추기
    if model_type.lower() == "motionheuristic":
        # MotionHeuristic은 beta 파라미터가 없으므로 shape이 (3, 78) 또는 (3, 7)임
        print("💡 MotionHeuristic model detected: Skipping beta selection.")
        des_inf_raw = mat_data['desire_model']
        bel_inf_raw = mat_data['belief_model']
        actual_beta = None # 적용되지 않음
    else:
        # BToM, NoCost, TrueBelief는 beta_score_values [0.5, 1.0, 1.5, ..., 10.0]에 따라 (3, 78, 20) 형태임
        beta_values = np.atleast_1d(mat_data['beta_score_values'])
        beta_idx = np.argmin(np.abs(beta_values - target_beta))
        actual_beta = beta_values[beta_idx]
        print(f"🎯 Selected Beta Score: {actual_beta} (Index: {beta_idx})")
        
        # mat_data['desire_model']의 shape는 (3, 78, 20)
        # -> (3, 78, 1) - 피험자 1명처럼 취급
        des_inf_raw = mat_data['desire_model'][:, :, beta_idx]
        bel_inf_raw = mat_data['belief_model'][:, :, beta_idx]

    # 2. 플롯 코드 호환성을 위한 데이터 복제 (3, 78) -> (3, 78, 16)
    # 똑같은 로봇(모델) 16마리가 똑같은 대답을 했다고 가정
    des_inf = np.repeat(des_inf_raw[:, :, np.newaxis], n_dummy_subj, axis=2)
    bel_inf = np.repeat(bel_inf_raw[:, :, np.newaxis], n_dummy_subj, axis=2)

    # 3. 통계 계산 (복제했으므로 기존과 동일한 np.nanmean, np.nanstd 로직 사용 가능)
    des_inf_mean = np.nanmean(des_inf, axis=2)
    bel_inf_mean = np.nanmean(bel_inf, axis=2)

    bel_sum = np.nansum(bel_inf_mean, axis=0)
    bel_sum[bel_sum == 0] = 1.0 
    bel_inf_mean_norm = bel_inf_mean / bel_sum[np.newaxis, :]

    # 표준 편차/오차 (값이 다 똑같으므로 자동으로 0이 됨)
    n_valid_des = np.sum(~np.isnan(des_inf), axis=2)
    n_valid_bel = np.sum(~np.isnan(bel_inf), axis=2)
    n_valid_des[n_valid_des == 0] = 1
    n_valid_bel[n_valid_bel == 0] = 1
    
    des_inf_se = np.nanstd(des_inf, axis=2, ddof=1) / np.sqrt(n_valid_des)
    bel_inf_se = np.nanstd(bel_inf, axis=2, ddof=1) / np.sqrt(n_valid_bel)

    # 4. 그룹별 통계 계산
    groups_raw = get_group_indices(include_irrational=False)
    group_inds = [[sid - 1 for sid in group] for group in groups_raw]
    n_group = len(group_inds)
    
    des_inf_group_mean = np.zeros((3, n_group))
    bel_inf_group_mean = np.zeros((3, n_group))
    des_inf_group_sd = np.zeros((3, n_group))
    bel_inf_group_sd = np.zeros((3, n_group))
    
    for gi in range(n_group):
        g_idxs = group_inds[gi]
        des_inf_group_mean[:, gi] = np.nanmean(des_inf_mean[:, g_idxs], axis=1)
        bel_inf_group_mean[:, gi] = np.nanmean(bel_inf_mean_norm[:, g_idxs], axis=1)

        # 그룹 안에서의 분산 (시나리오 간의 차이)
        des_inf_group_sd[:, gi] = np.nanstd(des_inf_mean[:, g_idxs], axis=1, ddof=1)
        bel_inf_group_sd[:, gi] = np.nanstd(bel_inf_mean_norm[:, g_idxs], axis=1, ddof=1)

    # 5. 최종 딕셔너리 구성 (LLM, Human과 100% 동일한 키)
    model_data = {
        'des_inf': des_inf,
        'bel_inf': bel_inf,
        'des_inf_mean': des_inf_mean,
        'bel_inf_mean_norm': bel_inf_mean_norm,
        'des_inf_se': des_inf_se,
        'bel_inf_se': bel_inf_se,
        'des_inf_group_mean': des_inf_group_mean,
        'bel_inf_group_mean': bel_inf_group_mean,
        'des_inf_group_sd': des_inf_group_sd,
        'bel_inf_group_sd': bel_inf_group_sd,
        'group_mapping': groups_raw,
        'model_type': model_type, # 참고용으로 추가
        'beta_score': actual_beta # 참고용으로 추가
    }

    # 6. 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"✅ {model_type.upper()} Data Conversion Complete! Saved to: {output_path}")


# =============================================================================
# 2. Transformer Result Processing (CSV -> PKL)
# =============================================================================
def process_model_results(target_dir, mode='normal'):
    """
    지정된 폴더(target_dir) 내의 subject_*.csv 파일들을 읽어서 model_data.pkl 파일로 자동 저장함.
    mode='everystep'일 경우, 누적된 증거를 바탕으로 한 최종 사후 판단을 얻기 위해
    Belief와 Desire 모두 마지막 스텝(t=End)의 데이터를 추출하여 Human Data와 구조를 맞춤.
    
    Args:
        target_dir (str): CSV 파일들이 있는 경로 (예: results/gpt-4o/reasoning)
    """
    output_filename = os.path.join(target_dir, "model_data.pkl")
    
    # 1. CSV 파일 목록 스캔
    csv_files = sorted(glob.glob(os.path.join(target_dir, "subject_*.csv")))
    
    if not csv_files:
        print(f"[Warning] No CSV files found in {target_dir}. Skipping processing.")
        return

    print(f"Processing {len(csv_files)} subjects in '{target_dir}' (Mode: {mode})...")

    # 2. 데이터 차원 설정
    # Desire: K, L, M (3개)
    # Belief: L, M, Empty (3개) -> Human은 3개였으나, 모델은 Empty가 포함됨
    n_rating_des = 3 
    n_rating_bel = 3
    n_cond = 78 # 전체 시나리오 개수 (Irrational 포함해서 로드)
    n_subj = len(csv_files)

    # 데이터 담을 배열 초기화 (Rating x Condition x Subject)
    # Human data 구조인 (3, 78, 16)과 유사하게 맞춤
    des_inf = np.full((n_rating_des, n_cond, n_subj), np.nan)
    bel_inf = np.full((n_rating_bel, n_cond, n_subj), np.nan)

    # 3. 데이터 로드 및 Matrix 변환
    for subj_idx, file_path in enumerate(csv_files):
        df = pd.read_csv(file_path)
        
        # 1부터 78까지 시나리오 ID를 순회하며 정확한 자리에 데이터 삽입 (누락 방지)
        for sc_idx in range(n_cond):
            sc_id = sc_idx + 1 # 1-based scenario_id
            
            sc_df = df[df['scenario_id'] == sc_id]
            if sc_df.empty:
                continue # API 실패 등으로 데이터가 없으면 NaN 유지
                
            # Mode에 따른 데이터 추출
            if mode == 'everystep':
                sc_df = sc_df.sort_values('time_step')

            # Desire와 Belief 모두 최종 판단(사후 추론) 결과를 가져옴
            row_final = sc_df.iloc[-1]

            # Desire 데이터 채우기 (K, L, M)
            if 'desire_K' in row_final:
                des_inf[0, sc_idx, subj_idx] = row_final['desire_K']
                des_inf[1, sc_idx, subj_idx] = row_final['desire_L']
                des_inf[2, sc_idx, subj_idx] = row_final['desire_M']

            # Belief 데이터 채우기 (L, M, Empty)
            if 'belief_L' in row_final:
                bel_inf[0, sc_idx, subj_idx] = row_final['belief_L']
                bel_inf[1, sc_idx, subj_idx] = row_final['belief_M']
                bel_inf[2, sc_idx, subj_idx] = row_final['belief_Empty']

    # 4. 통계 계산 (Mean, SE, Norm)
    # shape: (Rating, Condition) -> subject을 모두 합쳤으므로
    des_inf_mean = np.nanmean(des_inf, axis=2)
    bel_inf_mean = np.nanmean(bel_inf, axis=2)

    # Normalize Belief
    # Human data처럼 각 시나리오별 합이 1이 되도록 정규화 (선택 사항이나 비교를 위해 수행)
    # 모델은 1~7 척도이므로, 합계로 나누어 확률 분포처럼 만듦
    bel_sum = np.nansum(bel_inf_mean, axis=0) # (78,)
    # 0으로 나누기 방지
    bel_sum[bel_sum == 0] = 1.0 
    bel_inf_mean_norm = bel_inf_mean / bel_sum[np.newaxis, :]

    # Standard Error
    # Count valid (non-NaN) subjects per condition
    n_valid_des = np.sum(~np.isnan(des_inf), axis=2)
    n_valid_bel = np.sum(~np.isnan(bel_inf), axis=2)
    # 0으로 나누기 방지
    n_valid_des[n_valid_des == 0] = 1
    n_valid_bel[n_valid_bel == 0] = 1
    # ddof=1 (표본표준편차)
    des_inf_se = np.nanstd(des_inf, axis=2, ddof=1) / np.sqrt(n_valid_des)
    bel_inf_se = np.nanstd(bel_inf, axis=2, ddof=1) / np.sqrt(n_valid_bel)

    # 5. 그룹별 통계
    groups_raw = get_group_indices(include_irrational=False)
    # 0-based index 변환
    group_inds = [[sid - 1 for sid in group] for group in groups_raw]
    n_group = len(group_inds)
    
    des_inf_group_mean = np.zeros((n_rating_des, n_group))
    bel_inf_group_mean = np.zeros((n_rating_bel, n_group))
    des_inf_group_sd = np.zeros((n_rating_des, n_group))
    bel_inf_group_sd = np.zeros((n_rating_bel, n_group))

    for gi in range(n_group):
        g_idxs = group_inds[gi]
        # 해당 그룹에 속한 시나리오들의 평균
        des_inf_group_mean[:, gi] = np.nanmean(des_inf_mean[:, g_idxs], axis=1)
        bel_inf_group_mean[:, gi] = np.nanmean(bel_inf_mean_norm[:, g_idxs], axis=1)
        # 해당 그룹 내 분산 (SD)
        des_inf_group_sd[:, gi] = np.nanstd(des_inf_mean[:, g_idxs], axis=1, ddof=1)
        bel_inf_group_sd[:, gi] = np.nanstd(bel_inf_mean_norm[:, g_idxs], axis=1, ddof=1)

    # 6. 저장
    # human_data.pkl과 키(Key) 이름을 동일하게 맞춤
    model_data = {
        'des_inf': des_inf,
        'bel_inf': bel_inf,
        'des_inf_mean': des_inf_mean,
        'bel_inf_mean_norm': bel_inf_mean_norm,
        'des_inf_se': des_inf_se,
        'bel_inf_se': bel_inf_se,
        'des_inf_group_mean': des_inf_group_mean,
        'bel_inf_group_mean': bel_inf_group_mean,
        'des_inf_group_sd': des_inf_group_sd,
        'bel_inf_group_sd': bel_inf_group_sd,
        'group_mapping': groups_raw
    }

    with open(output_filename, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"✅ Processed Data Saved: {output_filename}")

if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description="Process raw CSV results into model_data.pkl")
    
    # 1. 처리할 데이터의 위치를 지정하는 인자들
    parser.add_argument("--model", type=str, help="Model name (e.g., gpt-4o, btom, nocost, truebelief, motionheuristic, human)")
    parser.add_argument("--condition", type=str, help="Experiment condition (e.g., reasoning, oneshot)")
    parser.add_argument("--mode", type=str, default="normal", choices=["normal", "everystep"], help="Analysis mode")
    
    # 레퍼런스(논문 원본) 데이터 변환 플래그
    parser.add_argument("--ref_only", action="store_true", help="Convert reference data (MAT/R) to PKL")
    parser.add_argument("--beta", type=float, default=2.5, help="Target beta score (default: 2.5)")

    args = parser.parse_args()

    print("\n" + "="*60)
    print("🚀 Data Processor Started")
    print("="*60)

    if args.ref_only:
        model_name = args.model.lower()
        if model_name == "human":
            print("💡 Task: Convert Human Data (MAT -> PKL)")
            convert_human_mat_to_pickle()
            pkl_out = HUMAN_PKL_PATH
        else:
            print(f"💡 Task: Convert Reference Model [{model_name}] (MAT -> PKL)")
            # 각 모델별 원본 MAT 파일 경로 (사전에 data/ 내에 존재해야 함)
            mat_path = os.path.join(REFERENCE_MAT_PATH, f"{model_name}_results_complete.mat")
            pkl_out = os.path.join(REFERENCE_PKL_DIR, model_name, f"{model_name}_data.pkl")
            
            convert_reference_model_to_pickle(
                mat_path=mat_path, 
                output_path=pkl_out, 
                model_type=model_name, 
                target_beta=args.beta
            )
            
        # 무결성 검증
        try:
            print("\n🔍 Running Data Integrity Check...")
            inspect_pickle_data(pkl_out)
        except ImportError:
            print("\n⚠️ [Notice] Could not import inspect_pickle_data from src.utils. Skipping check.")

    else:
        # [중요] LLM 처리를 하려고 하는데 모델/조건이 없으면 여기서 에러 발생시킴!
        if not args.model or not args.condition:
            parser.error("""LLM 데이터를 처리하려면 --model 및 --condition 인자가 반드시 필요합니다. 
                         (또는 --ref_only 플래그를 사용하세요)""")

        # 경로 조합 로직 (main_experiment.py와 동일)
        if args.mode == "everystep":
            target_dir = os.path.join(BASE_RESULTS_DIR, args.model, args.condition, "everystep")
        else:
            target_dir = os.path.join(BASE_RESULTS_DIR, args.model, args.condition)
            
        print(f"💡 Task: Process Model Data (CSV -> PKL)")
        print(f"📂 Target Directory: {target_dir}")
        print(f"⚙️ Mode: {args.mode}")
        
        # 1. CSV -> PKL 변환
        process_model_results(target_dir, mode=args.mode)
        
        # 2. 결과 검증 (utils.py 활용)
        try:
            pkl_path = os.path.join(target_dir, "model_data.pkl")
            print("\n🔍 Running Data Integrity Check...")
            inspect_pickle_data(pkl_path)
        except ImportError:
            print("\n⚠️ [Notice] Could not import inspect_pickle_data from src.utils. Skipping check.")
            
    print("\n✨ Data Processing Completed!")