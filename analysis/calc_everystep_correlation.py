import os
import sys
import argparse
import pandas as pd
import numpy as np
import pingouin as pg
from scipy.stats import pearsonr
import warnings

# 무의미한 통계 경고 숨김
warnings.filterwarnings("ignore")

# 현재 스크립트(analysis 폴더)의 상위 경로를 파이썬 탐색 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) # 상위 폴더 (프로젝트 루트)

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.config import BASE_RESULTS_DIR, get_group_indices, BEHAVIOR_GROUPS
from src.prepare_everystep import load_reference_everystep
from analysis.plot_everystep import get_phase_index

# 분석할 6개의 인지 점수 컬럼
SCORE_COLS = ['desire_K', 'desire_L', 'desire_M', 'belief_L', 'belief_M', 'belief_Empty']
# 표 출력을 위해 컬럼명을 짧게 축약 (예: desire_K -> d_K, belief_Empty -> b_Empty)
SHORT_COLS = [c.replace('desire_', 'd_').replace('belief_', 'b_') for c in SCORE_COLS]

# 터미널 가독성을 위한 Phase Index -> Name 매핑 딕셔너리
PHASE_NAMES = {
    0: 'Start', 1: 'Approach G1', 2: 'Pass G1/Select', 
    3: 'See G2', 4: 'Stop', 5: 'Return/Appr G2', 6: 'Selected', 9: 'Unknown'
}

def get_llm_data(model_name, condition, allowed_scenarios):
    """
    LLM 모델의 CSV 파일을 읽어와 Belief를 정규화한 데이터프레임 반환
    (plot_everystep.py의 로직을 그대로 가져옴)
    """
    # 1. 경로 설정
    output_dir = os.path.join(BASE_RESULTS_DIR, model_name, condition, "everystep")
    data_path = os.path.join(output_dir, "everystep_valid_only.csv")
    
    if not os.path.exists(data_path):
        print(f"❌ Error: Valid-only data not found at {data_path}")
        return None
        
    print(f"📥 Loading LLM data from {data_path}...")
    df = pd.read_csv(data_path)

    # Irrational 시나리오 제외 필터링
    df = df[df['scenario_id'].isin(allowed_scenarios)]
    
    # 2. 평균 구하기 (subject_id가 여러 개일 경우)
    df_mean = df.groupby(['scenario_id', 'group_id', 'time_step', 'phase'])[SCORE_COLS].mean().reset_index()
    
    # 3. Belief 정규화 (1~7점 -> 0~1 확률)
    belief_cols = ['belief_L', 'belief_M', 'belief_Empty']
    df_mean_shifted = df_mean[belief_cols] - 1
    df_mean_shifted = np.maximum(df_mean_shifted, 0)
    bel_sum = df_mean_shifted.sum(axis=1).replace(0, 1.0)
    df_mean[belief_cols] = df_mean_shifted.div(bel_sum, axis=0)
    
    return df_mean

def format_metrics(r, p, mae):
    """상관계수(r)와 유의성(p), 그리고 MAE를 15칸의 고정된 길이 문자열로 포맷팅"""
    if pd.isna(r) or pd.isna(p):
        r_str = "Const"
    else:
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        r_str = f"{r:5.2f}{stars}"

    # 예: " 0.85*** (0.12)" 또는 "Const    (0.05)" (총 15자리 고정)
    return f"{r_str:<8} ({mae:.2f})"

def calculate_correlations(llm_name, condition):
    # Irrational을 제외한 73개 시나리오 추출
    allowed_groups = get_group_indices(include_irrational=False)
    allowed_scenarios = [sc for group in allowed_groups for sc in group]
    # -------------------------------------------------------------
    # 1. 데이터 로드 및 병합
    # -------------------------------------------------------------
    df_llm = get_llm_data(llm_name, condition, allowed_scenarios)
    if df_llm is None: return

    # 순회할 레퍼런스 모델 목록
    ref_models = ['btom', 'truebelief', 'nocost', 'hindsight']
    
    # 모든 레퍼런스의 결과를 담을 거대한 리스트
    all_csv_results = []
    
    print("\n" + "="*80)
    print(f"🚀 Batch Analysis Started: [LLM: {llm_name}({condition})] vs ALL REFS")
    print("="*80)

    for ref_name in ref_models:
        df_ref = load_reference_everystep(ref_name)
        if df_ref is None:
            print(f"⚠️ Warning: Could not load {ref_name} data. Skipping...")
            continue
            
        df_ref = df_ref[df_ref['scenario_id'].isin(allowed_scenarios)]
        
        # 두 데이터프레임을 scenario_id, time_step, phase를 기준으로 병합 (Inner Join)
        # LLM 변수에는 '_llm', Ref 변수에는 '_ref' 접미사 부여
        df_merged = pd.merge(
            df_llm, 
            df_ref[['scenario_id', 'time_step', 'phase'] + SCORE_COLS], 
            on=['scenario_id', 'time_step', 'phase'], 
            suffixes=('_llm', '_ref'), 
            how='inner'
        )
        # Phase Index 계산 적용
        df_merged['phase_idx'] = df_merged.apply(lambda row: get_phase_index(row['group_id'], row['phase']), axis=1)

        print(f"\n[{ref_name.upper()}] Processing correlations and MAE...")

        # -------------------------------------------------------------
        # [1] Phase-based Analysis
        # -------------------------------------------------------------
        phase_indices = sorted(df_merged['phase_idx'].unique())
        for p_idx in phase_indices:
            if p_idx == 9: continue 
            df_phase = df_merged[df_merged['phase_idx'] == p_idx]

            # 💡 핵심 로직: 타임스텝 길이에 따른 편향을 막기 위해 시나리오 단위로 평균(Aggregation) 산출
            # LLM 점수와 REF 점수 모두 평균을 냄
            agg_cols = {f"{col}_llm": 'mean' for col in SCORE_COLS}
            agg_cols.update({f"{col}_ref": 'mean' for col in SCORE_COLS})
            df_phase_agg = df_phase.groupby('scenario_id').agg(agg_cols).reset_index()

            if len(df_phase) < 3: continue
            
            phase_name = PHASE_NAMES.get(p_idx, f"Phase {p_idx}")
            
            row_data = {
                'Reference_Model': ref_name.upper(),
                'Analysis_Type': 'Phase', 
                'Category': phase_name, 
                'N': len(df_phase_agg)
            }

            for col in SCORE_COLS:
                # MAE 계산 (절대 오차의 평균)
                mae = np.abs(df_phase_agg[f"{col}_llm"] - df_phase_agg[f"{col}_ref"]).mean()
                if df_phase_agg[f"{col}_llm"].std() == 0 or df_phase_agg[f"{col}_ref"].std() == 0:
                    r, p = np.nan, np.nan
                else:
                    r, p = pearsonr(df_phase_agg[f"{col}_llm"], df_phase_agg[f"{col}_ref"])
                
                row_data[f'{col}_r'] = r
                row_data[f'{col}_p'] = p
                row_data[f'{col}_mae'] = mae
                
            all_csv_results.append(row_data)

        # -------------------------------------------------------------
        # [2] Overall Trajectory Analysis
        # -------------------------------------------------------------
        row_data = {
            'Reference_Model': ref_name.upper(),
            'Analysis_Type': 'Overall', 
            'Category': 'Overall', 
            'N': df_merged['scenario_id'].nunique()
        }

        for col in SCORE_COLS:
            mae = np.abs(df_merged[f"{col}_llm"] - df_merged[f"{col}_ref"]).mean()
            try:
                rm_res = pg.rm_corr(data=df_merged, x=f"{col}_llm", y=f"{col}_ref", subject='scenario_id')
                r, p = rm_res['r'].iloc[0], rm_res['pval'].iloc[0]
            except (AssertionError, ValueError, np.linalg.LinAlgError):
                r, p = np.nan, np.nan
            
            row_data[f'{col}_r'] = r
            row_data[f'{col}_p'] = p
            row_data[f'{col}_mae'] = mae
            
        all_csv_results.append(row_data)

        # -------------------------------------------------------------
        # [3] Group-specific Analysis
        # -------------------------------------------------------------
        groups = sorted(df_merged['group_id'].unique())
        for g in groups:
            df_g = df_merged[df_merged['group_id'] == g]
            unique_scenarios = df_g['scenario_id'].nunique()
            group_name = BEHAVIOR_GROUPS.get(g, f"Group {g}")

            # 반복 측정 상관관계는 최소 3개의 독립적인 시나리오가 필요
            if unique_scenarios < 3: continue
                
            row_data = {
                'Reference_Model': ref_name.upper(),
                'Analysis_Type': 'Group', 
                'Category': group_name, 
                'N': unique_scenarios
            }

            for col in SCORE_COLS:
                mae = np.abs(df_g[f"{col}_llm"] - df_g[f"{col}_ref"]).mean()
                try:
                    rm_res = pg.rm_corr(data=df_g, x=f"{col}_llm", y=f"{col}_ref", subject='scenario_id')
                    r, p = rm_res['r'].iloc[0], rm_res['pval'].iloc[0]
                except (AssertionError, ValueError, np.linalg.LinAlgError):
                    r, p = np.nan, np.nan
                
                row_data[f'{col}_r'] = r
                row_data[f'{col}_p'] = p
                row_data[f'{col}_mae'] = mae
                
            all_csv_results.append(row_data)

    # print("="*125 + "\n")
    # print("* Format: r_value (MAE). MAE is Mean Absolute Error (lower is more similar).")
    # print("* Significance codes:  *** p<0.001,  ** p<0.01,  * p<0.05")
    # print("* Column names are abbreviated (e.g., d_K = desire_K, b_Empty = belief_Empty)")

    # -------------------------------------------------------------
    # 🌟 CSV 파일로 저장
    # -------------------------------------------------------------
    df_results = pd.DataFrame(all_csv_results)

    # 보기 좋게 컬럼 순서 정렬
    base_cols = ['Reference_Model', 'Analysis_Type', 'Category', 'N']
    score_cols_order = []
    for col in SCORE_COLS:
        score_cols_order.extend([f'{col}_r', f'{col}_p', f'{col}_mae'])
    
    df_results = df_results[base_cols + score_cols_order]

    save_dir = os.path.join(BASE_RESULTS_DIR, llm_name, condition, "everystep")
    os.makedirs(save_dir, exist_ok=True)
    
    save_filename = f"everystep_correlation_all_references.csv"
    save_path = os.path.join(save_dir, save_filename)
    
    df_results.to_csv(save_path, index=False)
    print("\n" + "="*80)
    print(f"✅ SUCCESS: All correlation data saved to -> {save_path}")
    print("="*80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate stepwise correlation between LLM and Reference Model.")
    parser.add_argument("--llm", type=str, required=True, help="LLM model name (e.g., gpt-4o)")
    parser.add_argument("--condition", type=str, required=True, help="Experiment condition (e.g., vanilla, oneshot)")
    
    args = parser.parse_args()
    
    calculate_correlations(args.llm, args.condition)