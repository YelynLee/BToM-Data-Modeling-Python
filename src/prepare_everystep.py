import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from collections import deque
import scipy.io

# 현재 스크립트(analysis 폴더)의 상위 경로를 파이썬 탐색 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) # 상위 폴더 (프로젝트 루트)

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.prepare_motionheur_everystep import generate_mh_scores
from src.dataset import df_btom
from src.config import BASE_RESULTS_DIR, REFERENCE_PKL_DIR, HUMAN_PKL_PATH, result_dir
from src.utils import get_valid_scenarios, build_master_dataframe

# =========================================================================
# 1. 벽을 우회하는 실제 최단 경로(BFS) 계산 헬퍼 함수
# =========================================================================
def get_true_distance(start_x, start_y, target_x, target_y, wx, wy, ww, wh):
    """
    15x5 그리드 내에서 벽을 통과하지 않고 목표까지 가는 실제 최단 이동 칸 수를 계산합니다.
    (자료형 언더플로우를 방지하기 위해 모두 int로 변환 후 연산)
    """
    # 안전한 연산을 위해 모두 int형으로 변환 (언더플로우 완벽 차단)
    sx, sy = int(start_x), int(start_y)
    tx, ty = int(target_x), int(target_y)
    
    # 벽 데이터가 없는 경우(NaN) 예외 처리
    if pd.isna(wx) or pd.isna(ww):
        wx, wy, ww, wh = -1, -1, 0, 0
    else:
        wx, wy, ww, wh = int(wx), int(wy), int(ww), int(wh)
    
    # BFS를 위한 큐와 방문 기록 세트
    queue = deque([(sx, sy, 0)])
    visited = set([(sx, sy)])
    
    while queue:
        cx, cy, dist = queue.popleft()
        
        if cx == tx and cy == ty:
            return dist
        
        # 상하좌우 이동 검사
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + dx, cy + dy
            
            # 1. 15x5 그리드 범위 내에 있는지 확인
            if 1 <= nx <= 15 and 1 <= ny <= 5:
                # 2. 벽의 영역(Bounding Box)에 부딪히는지 확인
                if wx <= nx < wx + ww and wy <= ny < wy + wh:
                    continue # 벽이면 통과 불가
                
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny, dist + 1))
                    
    # 도달할 수 없는 갇힌 상태라면 무한대 반환
    return float('inf')

# =========================================================================
# 2. Phase Labeling (수학적 판별 로직)
# =========================================================================
def apply_phase_labeling(df):
    """
    시간의 흐름(Timeline)에 따라 3개의 핵심 경계선(Anchors)을 순차적으로 찾아내어
    사건의 흐름(Phase Sequence)을 강제합니다.
    """
    df['phase'] = "Unknown"

    print("  -> Calculating True Distances (avoiding walls)...")
    
    # 기존 단순 맨해튼 거리 대신, 벽 정보를 포함한 실제 BFS 거리를 적용
    df['dist_G1'] = df.apply(lambda r: get_true_distance(
        r['agent_x'], r['agent_y'], 1, 1, 
        r['wall_start_x'], r['wall_start_y'], r['wall_width'], r['wall_height']), axis=1)
        
    df['dist_G2'] = df.apply(lambda r: get_true_distance(
        r['agent_x'], r['agent_y'], 15, 5, 
        r['wall_start_x'], r['wall_start_y'], r['wall_width'], r['wall_height']), axis=1)
    
    grouped = df.groupby(['subject_id', 'scenario_id'])
    
    for (subj_id, sc_id), group_data in grouped:
        group_id = group_data['group_id'].iloc[0]
        base_mask = (df['subject_id'] == subj_id) & (df['scenario_id'] == sc_id)
        
        # -------------------------------------------------------------
        # ⏱️ 타임라인 경계선(Anchors) 순차 추출
        # -------------------------------------------------------------
        
        # [0] G2 시야 확보 시점 (가장 확실한 타임라인 구분자)
        vis_data = group_data[group_data['visible_goal2'] == 1]
        ts_vis_G2 = vis_data['time_step'].min() if not vis_data.empty else float('inf')

        # [1] G1 기준 앵커 
        # 반드시 G2 시야 확보 이전 구간에서 탐색.
        limit_ts = ts_vis_G2 if ts_vis_G2 != float('inf') else group_data['time_step'].max()
        before_vis = group_data[group_data['time_step'] <= limit_ts]
        
        if not before_vis.empty:
            min_dist_G1 = before_vis['dist_G1'].min()
            ts_arrive_G1 = before_vis[before_vis['dist_G1'] == min_dist_G1]['time_step'].min()
            
            # 도착 이후, 거리가 '처음으로 다시 증가'하는 시점 찾기
            after_arrive = before_vis[before_vis['time_step'] >= ts_arrive_G1]
            depart_data = after_arrive[after_arrive['dist_G1'] > min_dist_G1]
            
            if not depart_data.empty:
                ts_leave_G1 = depart_data['time_step'].min() - 1
            else:
                ts_leave_G1 = after_arrive['time_step'].max()
        else:
            ts_leave_G1 = float('inf')
            min_dist_G1 = 'N/A'
            
        # [2] G2 기준 앵커 (반드시 G1을 떠난 '이후' 구간에서 탐색!)
        # 시작 위치(t=1)가 우연히 G2와 가깝더라도 무시.
        valid_leave_ts = ts_leave_G1 if ts_leave_G1 != float('inf') else 1
        after_leave_G1 = group_data[group_data['time_step'] >= valid_leave_ts]
        
        if not after_leave_G1.empty:
            min_dist_G2 = after_leave_G1['dist_G2'].min()
            ts_peak_G2_first = after_leave_G1[after_leave_G1['dist_G2'] == min_dist_G2]['time_step'].min()
            ts_peak_G2_last  = after_leave_G1[after_leave_G1['dist_G2'] == min_dist_G2]['time_step'].max()
        else:
            # 예외 상황 안전 장치
            ts_peak_G2_first = group_data['time_step'].max()
            ts_peak_G2_last  = group_data['time_step'].max()
            min_dist_G2 = 'N/A'
        
        # 안전한 비교를 위해 마스터 데이터프레임의 time_step 컬럼 지정
        ts_col = df['time_step']

        # -------------------------------------------------------------
        # [디버깅 코드 추가 1] Scenario 1의 주요 변수 값 출력
        # -------------------------------------------------------------
        if sc_id == 63 and subj_id == 1:
            print(f"\n[DEBUG] Subject: {subj_id} | Scenario: {sc_id} | Group: {group_id}")
            print(f"  👉 G1 관련: min_dist_G1={min_dist_G1}, ts_arrive_G1={ts_arrive_G1}, ts_leave_G1={ts_leave_G1}")
            print(f"  👉 G2 관련: min_dist_G2={min_dist_G2}, ts_peak_G2_first={ts_peak_G2_first}, ts_peak_G2_last={ts_peak_G2_last}")
            print(f"  👉 시야 관련: ts_vis_G2={ts_vis_G2}")
        
        # -------------------------------------------------------------
        # 🏷️ 시퀀스 룰 기반 Phase 할당
        # -------------------------------------------------------------
        
        # [A] No Check 패턴 (G3, G5: 직진 후 정착)
        if group_id in [3, 5]:
            ts_arrive_G1_only = group_data[group_data['dist_G1'] == group_data['dist_G1'].min()]['time_step'].min()
            df.loc[base_mask & (ts_col <= ts_arrive_G1_only), 'phase'] = "Approach G1"
            # df.loc[base_mask & (ts_col > ts_arrive_G1_only), 'phase'] = "Stay G1"
            
        # [B] Check-Stay 패턴 (G2: 탐색 후 G2 정착)
        elif group_id == 2:
            df.loc[base_mask & (ts_col < ts_leave_G1), 'phase'] = "Approach G1"
            df.loc[base_mask & (ts_col >= ts_leave_G1) & (ts_col < ts_vis_G2), 'phase'] = "Pass G1"

            if ts_vis_G2 != float('inf'):
                df.loc[base_mask & (ts_col == ts_vis_G2), 'phase'] = "See G2"
                df.loc[base_mask & (ts_col > ts_vis_G2) & (ts_col <= ts_peak_G2_first), 'phase'] = "Approach G2"
            else:
                df.loc[base_mask & (ts_col >= ts_leave_G1) & (ts_col <= ts_peak_G2_first), 'phase'] = "Approach G2"

            # df.loc[base_mask & (ts_col > ts_peak_G2_first), 'phase'] = "Stay G2"
            
        # [C] Check-GoBack 패턴 (G1, G4: 끝까지 가서 확인 후 회군)
        elif group_id in [1, 4]:
            df.loc[base_mask & (ts_col < ts_leave_G1), 'phase'] = "Approach G1"
            df.loc[base_mask & (ts_col >= ts_leave_G1) & (ts_col < ts_vis_G2), 'phase'] = "Pass G1"
            
            if ts_vis_G2 != float('inf'):
                df.loc[base_mask & (ts_col >= ts_vis_G2) & (ts_col <= ts_peak_G2_last), 'phase'] = "See G2"
            
            df.loc[base_mask & (ts_col > ts_peak_G2_last), 'phase'] = "Return G1"
            
        # [D] Check-Partial 패턴 (G6, G7: 부분 탐색 후 멈춤)
        elif group_id in [6, 7]:
            df.loc[base_mask & (ts_col < ts_leave_G1), 'phase'] = "Approach G1"
            df.loc[base_mask & (ts_col >= ts_leave_G1) & (ts_col < ts_vis_G2), 'phase'] = "Pass G1"
            
            if ts_vis_G2 != float('inf'):
                df.loc[base_mask & (ts_col >= ts_vis_G2) & (ts_col <= ts_peak_G2_first), 'phase'] = "See G2"
                
            # df.loc[base_mask & (ts_col > ts_peak_G2_first), 'phase'] = "Stop between G1 and G2"

    # =========================================================================
    # 🌟 [새로운 요구사항 반영] 후처리 (Post-processing) 라벨링
    # =========================================================================
    print("  -> Applying Post-processing Labels (Start, Stop, Selected)...")
    
    # 1. 'Stop' 판별: 연이은 time_step에서 위치가 동일한 경우
    # 각 피험자/시나리오 그룹 내에서 바로 이전 타임스텝의 x, y 좌표를 가져옵니다.
    df['prev_x'] = df.groupby(['subject_id', 'scenario_id'])['agent_x'].shift(1)
    df['prev_y'] = df.groupby(['subject_id', 'scenario_id'])['agent_y'].shift(1)
    
    # 현재 좌표와 이전 좌표가 같으면 'Stop' 할당 (t=1은 이전 좌표가 없으므로 제외됨)
    is_stopped = (df['agent_x'] == df['prev_x']) & (df['agent_y'] == df['prev_y'])
    df.loc[is_stopped, 'phase'] = 'Stop'
    
    # 임시로 만든 이전 좌표 컬럼은 삭제
    df.drop(['prev_x', 'prev_y'], axis=1, inplace=True)
    
    # 2. 'Start' 및 'Selected' 판별: 첫 번째와 마지막 time_step
    # 각 그룹별 최소(min) / 최대(max) time_step 값을 계산하여 행 크기에 맞게 가져옵니다.
    min_ts = df.groupby(['subject_id', 'scenario_id'])['time_step'].transform('min')
    max_ts = df.groupby(['subject_id', 'scenario_id'])['time_step'].transform('max')
    
    # [A] 첫 타임스텝(t=1)은 무조건 'Start'
    df.loc[df['time_step'] == min_ts, 'phase'] = 'Start'
    
    # 💡 [B] 마지막 타임스텝 및 BToM 연장 스텝 처리 (핵심 수정 부분)
    # 마지막 타임스텝의 좌표를 모든 행에 브로드캐스팅하여 임시 저장
    df['final_x'] = df.groupby(['subject_id', 'scenario_id'])['agent_x'].transform('last')
    df['final_y'] = df.groupby(['subject_id', 'scenario_id'])['agent_y'].transform('last')
    
    # (1) group_id가 6, 7이 '아닌' 경우 ➔ 일반 'Selected' 처리 (max_ts)
    mask_selected_max = (df['time_step'] == max_ts) & (~df['group_id'].isin([6, 7]))
    df.loc[mask_selected_max, 'phase'] = 'Selected'
    
    # (2) BToM 연장 대응: max_ts - 1 이면서 좌표가 최종 목적지와 완전히 동일한 경우 똑같이 'Selected'
    mask_selected_extended = (df['time_step'] == max_ts - 1) & (df['agent_x'] == df['final_x']) & (df['agent_y'] == df['final_y']) & (~df['group_id'].isin([6, 7]))
    df.loc[mask_selected_extended, 'phase'] = 'Selected'
    
    # (3) group_id가 6, 7인 경우 ➔ 'Stop between G1 and G2'
    # (앞선 일반 'Stop' 로직으로 인해 'Stop'으로 덮어씌워졌을 수 있으므로 다시 명확하게 잡아줌)
    mask_stop_max = (df['time_step'] == max_ts) & (df['group_id'].isin([6, 7]))
    df.loc[mask_stop_max, 'phase'] = 'Stop between G1 and G2'
    
    # (선택) group_id 6, 7도 btom inverse experiment에 포함할 예정이라면
    mask_stop_extended = (df['time_step'] == max_ts - 1) & (df['agent_x'] == df['final_x']) & (df['agent_y'] == df['final_y']) & (df['group_id'].isin([6, 7]))
    df.loc[mask_stop_extended, 'phase'] = 'Stop between G1 and G2'

    # 임시 컬럼 삭제
    df.drop(['final_x', 'final_y'], axis=1, inplace=True)

    return df

# =========================================================================
# 3. Reference Everystep 데이터 로드 함수
# =========================================================================
def load_reference_everystep(ref_model_name):
    """
    지정된 Reference Model의 매 스텝 데이터(.mat)를 불러와
    df_btom 구조와 매핑하고 Phase를 라벨링합니다. (Prior 데이터 t=0 포함)
    """
    df_merged = None # 최종 병합될 데이터프레임 초기화
    
    # ---------------------------------------------------------------------
    # 🌟 [분기 1] MotionHeuristic 모델인 경우 (동적 연산)
    # ---------------------------------------------------------------------
    if ref_model_name == 'motionheuristic':
        print(f"\n📥 Generating {ref_model_name.upper()} Everystep data dynamically...")
        
        # 1. human_data.pkl 로드 및 DataFrame(Target Y)으로 언롤링(Unrolling)
        if not os.path.exists(HUMAN_PKL_PATH):
            print(f"⚠️ Error: Human target data not found at {HUMAN_PKL_PATH}")
            return None
            
        with open(HUMAN_PKL_PATH, 'rb') as f:
            human_data = pickle.load(f)
            
        des_arr = human_data['des_inf_mean']       # shape: (3, 78)
        bel_arr = human_data['bel_inf_mean_norm']  # shape: (3, 78)
        
        records = []
        for sc_idx in range(78):
            sc_id = sc_idx + 1 # 1-based index (시나리오 1~78)
            
            # Desire 매핑: 행 0(K), 1(L), 2(M)
            records.append({'scenario_id': sc_id, 'model_type': 'Desire', 'truck': 'K', 'value': des_arr[0, sc_idx]})
            records.append({'scenario_id': sc_id, 'model_type': 'Desire', 'truck': 'L', 'value': des_arr[1, sc_idx]})
            records.append({'scenario_id': sc_id, 'model_type': 'Desire', 'truck': 'M', 'value': des_arr[2, sc_idx]})
            
            # Belief 매핑: 행 0(L), 1(M), 2(N: Empty)
            records.append({'scenario_id': sc_id, 'model_type': 'Belief', 'truck': 'L', 'value': bel_arr[0, sc_idx]})
            records.append({'scenario_id': sc_id, 'model_type': 'Belief', 'truck': 'M', 'value': bel_arr[1, sc_idx]})
            records.append({'scenario_id': sc_id, 'model_type': 'Belief', 'truck': 'N', 'value': bel_arr[2, sc_idx]})
            
        df_human_target = pd.DataFrame(records)
        
        # 2. world_mapping 동적 생성 (R 원본 논리 적용)
        # R 코드에 따르면 Group 4, 5, 7은 "G2 absent(우측 상단 빈 공간)" 시나리오입니다.
        # 따라서 이 그룹들은 world = 0 (Empty), 나머지는 world = 1 (L 트럭 존재)로 매핑합니다.
        world_mapping = {}
        for sc_id, group_data in df_btom.groupby('scenario_id'):
            grp = group_data['group_id'].iloc[0]
            # 4: Check-GoBack(Absent), 5: No Check(Absent), 7: Check-Partial(Absent)
            world_mapping[sc_id] = 0 if grp in [4, 5, 7] else 1
            
        # 3. 외부 모듈 호출 (가중치 피팅 및 스텝별 점수 산출)
        df_merged = generate_mh_scores(df_btom, df_human_target, world_mapping, exclude_irrational=True)
        
        # BToM 평가 스크립트와의 호환성을 위해 인지 모델은 피험자 0번으로 취급
        df_merged['subject_id'] = 0

    # ---------------------------------------------------------------------
    # 🌟 [분기 2] 기존 .mat 기반 모델인 경우 (BToM, TrueBelief 등)
    # ---------------------------------------------------------------------
    else:
        # 💡 최적 Beta 및 파일명 매핑 (필요시 이 부분만 수정하시면 됩니다)
        file_mapping = {
            'btom': 'btom_everystep_beta2.5.mat',
            'truebelief': 'truebelief_everystep_beta9.0.mat',
            'nocost': 'nocost_everystep_beta2.5.mat',
            'hindsight': 'hindsight_everystep_beta3.5.mat'
        }
        
        if ref_model_name not in file_mapping:
            print(f"\n⚠️ Error: Unknown reference model '{ref_model_name}'.")
            return None
            
        mat_filename = file_mapping[ref_model_name]
        mat_path = os.path.join(REFERENCE_PKL_DIR, ref_model_name, mat_filename)
        
        if not os.path.exists(mat_path):
            print(f"\n⚠️ Notice: {ref_model_name.upper()} everystep data not found at {mat_path}. Skipping overlay.")
            return None
            
        print(f"\n📥 Loading {ref_model_name.upper()} Everystep data from {mat_path}...")
        mat = scipy.io.loadmat(mat_path, squeeze_me=True)
        
        b_marg = mat['belief_marg'] 
        r_marg = mat['reward_marg']
        
        # -------------------------------------------------------------
        # 💡 [핵심 수정] 궤적 데이터 매핑 방식 변경 (Off-by-one 밀림 방지)
        # -------------------------------------------------------------
        # 1. 원본 궤적 데이터(df_btom)의 사본을 만듭니다.
        df_merged = df_btom.copy()

        # 💡 [핵심 수정] apply_phase_labeling에서 발생하는 KeyError 방지
        # 인지 모델은 피험자 0번(정답)으로 취급함을 명시적으로 할당
        df_merged['subject_id'] = 0
        
        # 병합할 빈 리스트들
        r_K_list, r_L_list, r_M_list = [], [], []
        b_L_list, b_M_list, b_Empty_list = [], [], []
        
        for idx, row in df_merged.iterrows():
            ns = int(row['scenario_id']) - 1 # 0-based index
            t  = int(row['time_step']) # MATLAB의 time_step (1부터 시작)
            
            b_arr = b_marg[ns]
            r_arr = r_marg[ns]

            # 만약 time_step이 1개라서 1D 배열(크기 3)로 추출되었다면 2D(3, 1)로 변경
            if b_arr.ndim == 1: b_arr = b_arr.reshape(3, -1)
            if r_arr.ndim == 1: r_arr = r_arr.reshape(3, -1)
            
            # 💡 t=1(출발점)에는 BToM 배열의 인덱스 0(Prior)을, t=2에는 인덱스 1(Post1)을 매핑!
            matlab_idx = min(t - 1, b_arr.shape[1] - 1)
            
            # 안전장치: 인덱스가 범위를 벗어나지 않도록
            matlab_idx = min(matlab_idx, b_arr.shape[1] - 1)
            
            r_K_list.append(r_arr[0, matlab_idx])
            r_L_list.append(r_arr[1, matlab_idx])
            r_M_list.append(r_arr[2, matlab_idx])
            b_L_list.append(b_arr[0, matlab_idx])
            b_M_list.append(b_arr[1, matlab_idx])
            b_Empty_list.append(b_arr[2, matlab_idx])
            
        df_merged['desire_K'] = r_K_list
        df_merged['desire_L'] = r_L_list
        df_merged['desire_M'] = r_M_list
        df_merged['belief_L'] = b_L_list
        df_merged['belief_M'] = b_M_list
        df_merged['belief_Empty'] = b_Empty_list
    
    # -------------------------------------------------------------
    # 💡 [핵심 연장] 배열의 마지막 값(최종 Posterior) 처리
    # -------------------------------------------------------------
    # 각 시나리오의 마지막 time_step 행을 복사하여 time_step을 +1 증가시킵니다.
    idx_last_steps = df_merged.groupby('scenario_id')['time_step'].idxmax()
    df_extensions = df_merged.loc[idx_last_steps].copy()
    df_extensions['time_step'] += 1
    
    # 연장된 행에 직전 행(각 시나리오의 원본 마지막 스텝)의 추론 점수들을 그대로 복사하여 유지시킵니다.
    score_columns = ['desire_K', 'desire_L', 'desire_M', 'belief_L', 'belief_M', 'belief_Empty']
    for col in score_columns:
        df_extensions[col] = df_merged.loc[idx_last_steps, col].values
    
    # 원래 데이터프레임과 연장된 행들을 합치고 재정렬
    df_merged = pd.concat([df_merged, df_extensions]).sort_values(['scenario_id', 'time_step']).reset_index(drop=True)

    print(f"  -> Applying Phase Labeling to {ref_model_name.upper()}...")
    df_final = apply_phase_labeling(df_merged)

    return df_final

# =========================================================================
# 추가 실험용 데이터 추출 스크립트 (Check-GoBack, Check-Stay)
# =========================================================================
def export_btom_experiment_data(ref_model_name):
    # 입력받은 ref_model_name으로 데이터 로드
    df_ref = load_reference_everystep(ref_model_name)
    
    # 타겟 그룹 필터링
    # 1: Check-GoBack(Present), 2: Check-Stay(Present), 4: Check-GoBack(Absent)
    target_groups = [1, 2, 4]
    df_target = df_ref[df_ref['group_id'].isin(target_groups)].copy()
    
    # 보기 좋게 확률을 소수점 3자리로 반올림
    prob_cols = ['desire_K', 'desire_L', 'desire_M', 'belief_L', 'belief_M', 'belief_Empty']
    df_target[prob_cols] = df_target[prob_cols].round(3)
    
    # CSV 저장
    output_path = os.path.join(BASE_RESULTS_DIR, ref_model_name, f"{ref_model_name}_reverse_inference_experiment.csv")
    df_target.to_csv(output_path, index=False)
    print(f"\n✅ 추가 실험용 BToM 데이터가 성공적으로 저장되었습니다: {output_path}")
    print(f"포함된 시나리오 수: {df_target['scenario_id'].nunique()}개")

# =========================================================================
# 메인 실행 래퍼 함수 (run_analysis.py에서 호출)
# =========================================================================
def run_prepare_everystep(model_name, condition, mode_dir="everystep"):
    """mode_dir: 'everystep', 'prefixstep', 'prefixstep_cur' 등. 출력 파일명은 호환을 위해 동일하게 유지."""
    print("\n" + "="*60)
    print(f"🚀 [{mode_dir}] Valid-only DataFrame Builder Started")
    print("="*60)
    
    target_dir = result_dir(model_name, condition, mode_dir)
    output_path = os.path.join(target_dir, "everystep_valid_only.csv")
    
    # 🌟 [NEW] 이미 파일이 존재하면 무거운 연산(BFS, 병합) 스킵
    if os.path.exists(output_path):
        print(f"⏩ [Skip] '{output_path}' 이미 존재합니다. 데이터 구축을 건너뜁니다.")
        
        # 시각화(plot_everystep.py)로 전달할 우등생 명단을 추출하기 위해 가볍게 로드
        df_master = pd.read_csv(output_path)
        selected_subjects = sorted(df_master['subject_id'].unique().tolist())
        
        return selected_subjects

    # ---------------------------------------------------------------------
    # 기존 데이터 구축 로직 (파일이 없을 때만 실행)
    # ---------------------------------------------------------------------
    valid_keys, selected_subjects = get_valid_scenarios(model_name, condition, mode_dir)
    
    if valid_keys:
        df_master = build_master_dataframe(model_name, condition, valid_keys, mode_dir)

    return selected_subjects # 이 명단을 run_analysis로 전달!

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=False, default="dummy", help="Model name (e.g., gpt-4o)")
    parser.add_argument("--condition", type=str, required=False, default="dummy", help="Condition (e.g., vanilla, reasoning, oneshot)")
    # 💡 [수정] --ref 인자 추가
    parser.add_argument("--mode_dir", type=str, default="everystep", help="everystep, prefixstep, prefixstep_cur 등 결과 하위 폴더")
    parser.add_argument("--ref", type=str, required=False, help="Reference model for generating experiment data (e.g., btom)")
    args = parser.parse_args()

    # 일반적인 valid_only 구축
    if args.model != "dummy" and args.condition != "dummy":
        run_prepare_everystep(args.model, args.condition, args.mode_dir)

    # 💡 --ref 인자가 들어왔을 때만 역방향 추론용 CSV 파일 추출 실행
    if args.ref:
        export_btom_experiment_data(args.ref)