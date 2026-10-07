import os
import sys
import glob
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, entropy
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from sklearn.feature_selection import mutual_info_classif
import warnings

# 무의미한 통계 경고 숨김
warnings.filterwarnings("ignore")

# 현재 스크립트의 상위 경로를 파이썬 탐색 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# 환경에 맞게 import 경로 조정
from src.config import BASE_RESULTS_DIR, get_group_indices
from src.prepare_everystep import load_reference_everystep

SCORE_COLS = ['desire_K', 'desire_L', 'desire_M', 'belief_L', 'belief_M', 'belief_Empty']

# 💡 평가 대상이 되는 인지 레퍼런스 모델 목록
REFERENCE_MODELS = ['btom', 'truebelief', 'nocost', 'hindsight', 'motionheuristic']

def get_model_raw_data(model_name, condition, allowed_scenarios, mode):
    """
    모델의 종류(LLM vs Reference)를 자동으로 판별하여 알맞은 방식으로 데이터를 로드
    정규화나 평균(mean) 계산 없이 원본 1~7점 척도 데이터를 그대로 반환
    mode('everystep' 또는 'endstep')에 따라 데이터를 다르게 로드합니다.
    - endstep (LLM): subject_*.csv 파일들을 순회하며 각 시나리오의 마지막(iloc[-1]) 스텝만 추출
    - endstep (Ref): load_reference_everystep 호출 후 시나리오별 마지막 행 추출
    """
    model_name_lower = model_name.lower()
    
    # [1] Reference Model인 경우 (.mat 또는 동적 생성)
    if model_name_lower in REFERENCE_MODELS:
        df = load_reference_everystep(model_name_lower)
        if df is None:
            print(f"⚠️ Warning: Failed to load reference data for {model_name}.")
            return None
        
        # 💡 endstep인 경우, 궤적에서 시나리오별 맨 마지막 타임스텝만 추출
        if mode == 'endstep':
            df = df.sort_values(['scenario_id', 'time_step']).groupby('scenario_id').last().reset_index()
            
    # [2] 일반 LLM 모델인 경우 (CSV 로드)
    else:
        if mode == 'everystep':
            # 기존 everystep 방식 (병합된 valid_only 파일 사용)
            data_path = os.path.join(BASE_RESULTS_DIR, model_name, condition, "everystep", "everystep_valid_only.csv")
            if not os.path.exists(data_path): return None
            df = pd.read_csv(data_path)
            
        elif mode == 'endstep':
            # subject_*.csv
            target_dir = os.path.join(BASE_RESULTS_DIR, model_name, condition)
            csv_files = glob.glob(os.path.join(target_dir, "subject_*.csv"))
            
            if not csv_files:
                print(f"⚠️ Warning: No subject files found in {target_dir}.")
                return None
                
            # 파일들을 읽고 한 번에 병합
            df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    # Irrational 시나리오 제외 필터링
    if allowed_scenarios is not None:
        df = df[df['scenario_id'].isin(allowed_scenarios)]
    
    # 필수 컬럼 존재 여부 확인
    if not all(col in df.columns for col in SCORE_COLS):
        print(f"⚠️ Warning: Missing required columns in {model_name}.")
        return None
        
    return df[SCORE_COLS]

def calculate_group_dependence_raw(models, condition, allowed_scenarios, mode):
    """
    주어진 모델 그룹의 데이터를 풀링(Pooling)하여 
    원본 1~7점 척도를 기반으로 MI, NMI, 카이제곱 p-value 및 유효 데이터 수(N) 계산
    - 1~7 이산형 데이터: 기존 교차표 기반 MI, NMI, 카이제곱 p-value 계산
    - 0~1 연속형 데이터: KSG 알고리즘을 통한 순수 MI 계산 및 자체 NMI 도출 (Binning X, p-value=NaN)
    """
    all_desire = []
    all_belief = []
    
    for model in models:
        df = get_model_raw_data(model, condition, allowed_scenarios, mode)
        if df is None: continue
        
        # 트럭 구분 해제: 모든 Desire와 Belief 값을 하나의 1차원 시리즈로 이어붙임
        desire_vals = pd.concat([df['desire_K'], df['desire_L'], df['desire_M']])
        belief_vals = pd.concat([df['belief_L'], df['belief_M'], df['belief_Empty']])
        
        all_desire.append(desire_vals)
        all_belief.append(belief_vals)
        
    if not all_desire:
        return None, None, None, 0
        
    # 그룹 데이터 풀링 
    combined_desire = pd.concat(all_desire)
    combined_belief = pd.concat(all_belief)
    
    # DataFrame으로 묶은 후 결측치(NaN)가 하나라도 있는 쌍(Pair)을 동시에 제거
    temp_df = pd.DataFrame({
        'desire': combined_desire.values, 
        'belief': combined_belief.values
    }).dropna()
    
    # Desire는 무조건 정수 처리
    temp_df['desire'] = temp_df['desire'].astype(int)
    # 💡 실제 분석에 사용된 유효 데이터 개수 (Valid N)
    valid_n = len(temp_df)
    
    # 연속형(BToM 등)인지 확인
    # 데이터를 1로 나눈 나머지(% 1)가 모두 0이면 딱 떨어지는 정수(이산형), 
    # 하나라도 나머지가 있으면 소수점이 있는 확률값(연속형)으로 판단합니다.
    is_continuous_belief = not (temp_df['belief'] % 1 == 0).all()

    if is_continuous_belief:
        # --- [1] 연속형 데이터 (Binning 없음) ---
        
        # 1) KSG 알고리즘으로 연속-이산 상호정보량(MI) 추정 (단위: nats)
        mi = mutual_info_classif(temp_df[['belief']], temp_df['desire'], discrete_features=False, random_state=42)[0]
        
        # 2) Desire(이산형)의 엔트로피 H(X) 계산 (단위: nats)
        # scikit-learn의 기본 MI 단위(nats)와 맞추기 위해 자연로그(base e) 사용
        value_counts = temp_df['desire'].value_counts(normalize=True)
        h_desire = entropy(value_counts, base=np.e)
        
        # 3) NMI 자체 계산 (MI / H(Desire))
        nmi = mi / h_desire if h_desire > 0 else 0.0
        
        # 연속형은 교차표를 만들 수 없으므로 카이제곱 검정 불가
        p = np.nan 

    else:
        # --- [2] 이산형 데이터 (기존 방식 유지) ---
        temp_df['belief'] = temp_df['belief'].astype(int)

        # 7x7 교차표 생성
        contingency = pd.crosstab(temp_df['desire'], temp_df['belief'])
        
        # 데이터가 너무 적어서 검정이 성립하지 않는 경우 방지
        if contingency.size < 4 or valid_n < 2:
            return np.nan, np.nan, np.nan, valid_n

        # 통계 지표 산출
        chi2, p, dof, ex = chi2_contingency(contingency)
        mi = mutual_info_score(temp_df['desire'], temp_df['belief'])
        
        # 💡 정규화된 상호정보량(NMI) 산출 (0.0 ~ 1.0)
        nmi = normalized_mutual_info_score(temp_df['desire'], temp_df['belief'])
    
    return mi, nmi, p, valid_n

def get_sig_star(p_val):
    """p-value에 따른 유의도 별표 반환"""
    if pd.isna(p_val): return ""
    if p_val < 0.001: return "***"
    if p_val < 0.01: return "**"
    if p_val < 0.05: return "*"
    return "ns"

def print_dependence_metrics(better_models, worse_models, condition, mode='everystep'):
    """
    계산된 종속성 지표를 터미널에 포맷팅하여 출력
    endstep 모드에서는 N수 부족으로 인해 [3] 단일 모델 분석을 건너뜁니다.
    """
    # Irrational을 제외한 시나리오 추출
    try:
        allowed_groups = get_group_indices(include_irrational=False)
        allowed_scenarios = [sc for group in allowed_groups for sc in group]
    except Exception as e:
        print("⚠️ Warning: Could not fetch allowed scenarios, using all available IDs.")
        allowed_scenarios = None # Fallback

    all_models = better_models + worse_models

    # 연산 수행
    print("\n" + "="*85)
    print("📊 Evaluating Raw Belief-Desire Dependence (1~7 Scale)")
    print("="*85)
    print("⏳ Calculating... Please wait.\n")

    all_mi, all_nmi, all_p, all_n = calculate_group_dependence_raw(all_models, condition, allowed_scenarios, mode)
    b_mi, b_nmi, b_p, b_n = calculate_group_dependence_raw(better_models, condition, allowed_scenarios, mode)
    w_mi, w_nmi, w_p, w_n = calculate_group_dependence_raw(worse_models, condition, allowed_scenarios, mode)

    # --- [1] 전체 모델 통합 결과 출력 ---
    print("[1] OVERALL DEPENDENCE (All Models Combined)")
    print("-" * 85)
    print(f" - Included Models: {', '.join(all_models)}")
    if all_mi is not None and not np.isnan(all_mi):
        print(f" - Valid Samples (N)    : {all_n:,}")
        print(f" - Mutual Info (MI)     : {all_mi:.4f} bits")
        print(f" - Normalized MI (NMI)  : {all_nmi:.4f} ({all_nmi*100:.2f}%)")
        print(f" - Chi-Square p-value   : {all_p:.2e} ({get_sig_star(all_p)})")
    else:
        print(" - ❌ Not enough valid data.")
    print()

    # --- [2] 그룹별 비교 결과 출력 ---
    print("[2] GROUP COMPARISON (Better vs Worse Models)")
    print("-" * 85)
    
    # Better Models
    print(f"🟢 Better Models: {', '.join(better_models)}")
    if b_mi is not None and not np.isnan(b_mi):
        print(f"   - Valid Samples (N)    : {b_n:,}")
        print(f"   - Mutual Info (MI)     : {b_mi:.4f} bits")
        print(f"   - Normalized MI (NMI)  : {b_nmi:.4f} ({b_nmi*100:.2f}%)")
        print(f"   - Chi-Square p-value   : {b_p:.2e} ({get_sig_star(b_p)})")
    else:
        print("   - ❌ Not enough valid data.")
    print()
    
    # Worse Models
    print(f"🔴 Worse Models:  {', '.join(worse_models)}")
    if w_mi is not None and not np.isnan(w_mi):
        print(f"   - Valid Samples (N)    : {w_n:,}")
        print(f"   - Mutual Info (MI)     : {w_mi:.4f} bits")
        print(f"   - Normalized MI (NMI)  : {w_nmi:.4f} ({w_nmi*100:.2f}%)")
        print(f"   - Chi-Square p-value   : {w_p:.2e} ({get_sig_star(w_p)})")
    else:
        print("   - ❌ Not enough valid data.")   
    print()

    # --- [3] 모델별 개별 분석 결과 출력 (everystep 모드에서만 출력) ---
    if mode == 'everystep':
        print("[3] PER-MODEL ANALYSIS")
        print("-" * 85)
        print(f"{'Group':<10} | {'Model Name':<20} | {'Valid N':<9} | {'MI (bits)':<9} | {'NMI (%)':<8} | {'p-value'}")
        print("-" * 85)

        model_list_with_labels = [('🟢 Better', m) for m in better_models] + [('🔴 Worse', m) for m in worse_models]

        for group_icon, model in model_list_with_labels:
            # 단일 모델 계산 시 리스트로 감싸서 전달
            mi, nmi, p, valid_n = calculate_group_dependence_raw([model], condition, allowed_scenarios, mode)
            
            if mi is not None and not np.isnan(mi):
                nmi_percent = f"{nmi*100:.2f}%"
                sig = get_sig_star(p)
                p_str = f"{p:.2e} ({sig})"
                print(f"{group_icon:<10} | {model:<20} | {valid_n:<9,} | {mi:<9.4f} | {nmi_percent:<8} | {p_str}")
            else:
                print(f"{group_icon:<10} | {model:<20} | {'-':<9} | {'-':<9} | {'-':<8} | ❌ No Data")
        print()
    else:
        print("[3] PER-MODEL ANALYSIS")
        print("-" * 85)
        print(" ⏭️  Skipped in 'endstep' mode due to insufficient N (N=234 per model) for reliable MI/Chi-Square.")
        print()

    # --- [4] BToM 레퍼런스 결과 출력 ---
    print("[4] REFERENCE MODEL ANALYSIS (Ground Truths)")
    print("-" * 85)
    
    # 💡 BToM 등 레퍼런스 모델 자동 순회
    for ref_model in ['btom', 'truebelief', 'nocost', 'motionheuristic', 'hindsight']:
        ref_mi, ref_nmi, ref_p, ref_n = calculate_group_dependence_raw([ref_model], condition, allowed_scenarios, mode)
        
        if ref_mi is not None and not np.isnan(ref_mi):
            print(f"🔵 Model: {ref_model.upper()}")
            print(f"   - Valid Samples (N)    : {ref_n:,}")
            print(f"   - Mutual Info (MI)     : {ref_mi:.4f} nats (KSG Estimator)")
            print(f"   - Normalized MI (NMI)  : {ref_nmi:.4f} ({ref_nmi*100:.2f}%)")
            print("   - Chi-Square p-value   : N/A (Continuous Belief Probability)\n")

if __name__ == "__main__":
    # 평가할 모델 리스트
    BETTER_MODELS = ['gemini-2.5-pro', 'deepseek-reasoner', 'deepseek-v4-pro', 'claude-opus-4-6'] 
    WORSE_MODELS = ['gpt-5.4', 'o4-mini', 'gemini-2.5-flash', 'claude-sonnet-4-6'] 
    CONDITION = 'vanilla'
    
    # 💡 모드를 지정하여 실행 (원하는 대로 주석 해제)
    # print_dependence_metrics(BETTER_MODELS, WORSE_MODELS, CONDITION, mode='everystep')
    print_dependence_metrics(BETTER_MODELS, WORSE_MODELS, CONDITION, mode='endstep')