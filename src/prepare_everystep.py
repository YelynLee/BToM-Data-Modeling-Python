import os
import argparse
import glob
import numpy as np
import pandas as pd
from collections import deque
import scipy.io
from src.dataset import df_btom
from src.config import BASE_RESULTS_DIR, get_group_indices, BTOM_EVERY_MAT_PATH

# =========================================================================
# 0. 벽을 우회하는 실제 최단 경로(BFS) 계산 헬퍼 함수
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
# 1. Phase Labeling (수학적 판별 로직)
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
                df.loc[base_mask & (ts_col == ts_vis_G2), 'phase'] = "See & Approach G2"
                df.loc[base_mask & (ts_col > ts_vis_G2) & (ts_col <= ts_peak_G2_first), 'phase'] = "Approach G2"
            else:
                df.loc[base_mask & (ts_col >= ts_leave_G1) & (ts_col <= ts_peak_G2_first), 'phase'] = "Approach G2"

            # df.loc[base_mask & (ts_col > ts_peak_G2_first), 'phase'] = "Stay G2"
            
        # [C] Check-GoBack 패턴 (G1, G4: 끝까지 가서 확인 후 회군)
        elif group_id in [1, 4]:
            df.loc[base_mask & (ts_col < ts_leave_G1), 'phase'] = "Approach G1"
            df.loc[base_mask & (ts_col >= ts_leave_G1) & (ts_col < ts_vis_G2), 'phase'] = "Pass G1"
            
            if ts_vis_G2 != float('inf'):
                df.loc[base_mask & (ts_col >= ts_vis_G2) & (ts_col <= ts_peak_G2_last), 'phase'] = "See & Reject G2"
            
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
    
    # [B] 마지막 타임스텝 처리
    # (1) group_id가 6, 7이 '아닌' 경우 ➔ 'Selected'
    mask_selected = (df['time_step'] == max_ts) & (~df['group_id'].isin([6, 7]))
    df.loc[mask_selected, 'phase'] = 'Selected'
    
    # (2) group_id가 6, 7인 경우 ➔ 'Stop between G1 and G2'
    # (앞선 일반 'Stop' 로직으로 인해 'Stop'으로 덮어씌워졌을 수 있으므로 다시 명확하게 잡아줌)
    mask_stop_between = (df['time_step'] == max_ts) & (df['group_id'].isin([6, 7]))
    df.loc[mask_stop_between, 'phase'] = 'Stop between G1 and G2'

    return df

# =========================================================================
# 1.5 데이터 정합성 검토 및 '유효한 시나리오' 추출 (Valid Subset Extraction)
# =========================================================================
def get_valid_scenarios(model_name, condition):
    """
    df_btom과 완벽하게 time_step 개수가 일치하는 (subject_id, scenario_id) 쌍만
    추출하여 set 형태로 반환합니다.
    -> 추후에 새롭게 응답을 받아야 할 것. 현재는 시간 문제로 스킵.
    """
    target_dir = os.path.join(BASE_RESULTS_DIR, model_name, condition, "everystep")
    csv_files = sorted(glob.glob(os.path.join(target_dir, "subject_*.csv")))
    
    if not csv_files:
        print(f"❌ Error: No CSV files found to validate in {target_dir}")
        return set()

    expected_counts = df_btom.groupby('scenario_id')['time_step'].count().to_dict()
    
    print("\n" + "="*60)
    print("🔍 Extracting Valid Scenarios Started")
    print("="*60)
    
    valid_keys = set() # 정상적인 (subject_id, scenario_id)를 담을 세트
    total_errors = 0
    total_valid = 0

    # 피험자별 유효 시나리오 목록을 담을 딕셔너리 (교집합 계산용)
    valid_by_subj = {i: set() for i in range(1, len(csv_files) + 1)}
    
    for subj_idx, file_path in enumerate(csv_files, start=1):
        try:
            df_model = pd.read_csv(file_path)
            actual_counts = df_model.groupby('scenario_id')['time_step'].count().to_dict()
            
            for sc_id, expected_len in expected_counts.items():
                actual_len = actual_counts.get(sc_id, 0)
                
                if expected_len == actual_len:
                    # ✅ 행 개수가 완벽히 일치하는 경우만 수집
                    valid_keys.add((subj_idx, sc_id))
                    valid_by_subj[subj_idx].add(sc_id)
                    total_valid += 1
                else:
                    # ❌ 누락된 경우 카운트 (터미널 도배를 막기 위해 에러 로그는 생략하거나 요약 가능)
                    total_errors += 1
                    
        except Exception as e:
            print(f"  ❌ Subject {subj_idx}: Failed to read. Error: {e}")

    # 🌟 [추가된 로직] 78개(전체 시나리오 수)를 모두 완벽하게 생성한 피험자 동적 추출
    max_scenarios = len(expected_counts)
    perfect_subjects = []

    print("\n  📊 [Valid Scenarios per Subject]")
    for subj_idx in sorted(valid_by_subj.keys()):
        valid_count = len(valid_by_subj[subj_idx])
        print(f"    - Subject {subj_idx:02d}: {valid_count:02d} valid scenarios")
        if valid_count == max_scenarios:
            perfect_subjects.append(subj_idx)

    # -------------------------------------------------------------------------
    # 🌟 [메인 로직] 분기점: Perfect Subjects vs Fallback Top 5
    # -------------------------------------------------------------------------
    final_valid_keys = set()
    selected_subjects = []
    
    if len(perfect_subjects) >= 5:
        # [플랜 A] 완벽한 피험자가 5명 이상 존재할 경우
        print(f"\n  🌟 [Plan A: Perfect Subjects Found]")
        print(f"    -> {len(perfect_subjects)} subjects completed all {max_scenarios} scenarios.")
        
        selected_subjects = perfect_subjects
        # 이미 찾아둔 raw 데이터 중에서 완벽한 피험자의 데이터만 쏙 빼서 씁니다.
        final_valid_keys = {(s, sc) for (s, sc) in valid_keys if s in selected_subjects}
    else:
        # [플랜 B] 완벽한 피험자가 5명보다 적을 경우 (Fallback)
        print(f"\n  ⚠️ [Plan B: No Perfect Subjects] -> Switching to Top 5 Fallback Logic")
        sorted_subjects = sorted(valid_by_subj.keys(), key=lambda x: len(valid_by_subj[x]), reverse=True)
        top_5_subjects = sorted_subjects[:5]
        
        print(f"  🏆 [Top 5 Subjects Selected]")
        for subj in top_5_subjects:
            print(f"    - Subject {subj:02d} (Passed: {len(valid_by_subj[subj])})")
        selected_subjects = top_5_subjects

        # 공통 시나리오 교집합 추출
        s_common_scenarios = set.intersection(*[valid_by_subj[s] for s in selected_subjects]) if selected_subjects else set()
        print(f"\n  🎯 [Common Valid Scenarios across Top 5 Subjects]")
        print(f"    -> {len(s_common_scenarios)} total common scenarios.")

        # 7개 그룹 커버리지 검토 (플랜 B 전용)
        print("\n  🔍 [Group Coverage Check]")
        groups_raw = get_group_indices(include_irrational=True)
        missing_groups = []
        
        for g_idx, group_scenarios in enumerate(groups_raw, start=1):
            intersection = s_common_scenarios.intersection(group_scenarios)
            if len(intersection) == 0:
                missing_groups.append(g_idx)
                print(f"    ⚠️ Group {g_idx}: 0 common scenarios! (Plotting might fail for this group)")
            else:
                print(f"    ✅ Group {g_idx}: {len(intersection)} common scenarios.")
                
        if missing_groups:
            print(f"    🚨 Warning: 그룹 {missing_groups}에 공통 시나리오가 없어 서브플롯이 비어 있을 수 있습니다.")
        else:
            print("    🎉 Excellent! 모든 7개 그룹에 최소 1개 이상의 공통 시나리오가 존재합니다.")

        # Master DataFrame 생성을 위해 Top 5 공통 시나리오만 남기기
        for subj in selected_subjects:
            for sc in s_common_scenarios:
                final_valid_keys.add((subj, sc))

    # -------------------------------------------------------------------------
    # 모든 피험자들의 공통 시나리오(Intersection) 계산
    # -------------------------------------------------------------------------
    if valid_by_subj:
        common_scenarios = set.intersection(*valid_by_subj.values())
    else:
        common_scenarios = set()
        
    print(f"\n  🎯 [Common Valid Scenarios across ALL subjects]")
    print(f"    -> {len(common_scenarios)} total common scenarios.")

    if common_scenarios:
        # 보기 좋게 오름차순 정렬해서 출력
        print(f"    -> Scenario IDs: {sorted(list(common_scenarios))}")
    else:
        print(f"    -> None 😢")
        
    print("\n  ================ Summary ================")
    print(f"  ✅ Total Found: {total_valid} valid scenario pairs.")
    print(f"  🚨 Total Dropped: {total_errors} scenario pairs due to missing time_steps.")
    print(f"  ✅ Prepared {len(final_valid_keys)} perfectly balanced pairs for the Master DataFrame.")
    print("-" * 60)
    
    return final_valid_keys, selected_subjects

# =========================================================================
# 2. Master DataFrame 생성 로직
# =========================================================================
def build_master_dataframe(model_name, condition, valid_keys):
    """
    df_btom과 모델의 Everystep 결과를 결합하여 마스터 데이터프레임을 생성합니다.
    """
    target_dir = os.path.join(BASE_RESULTS_DIR, model_name, condition, "everystep")
    csv_files = sorted(glob.glob(os.path.join(target_dir, "subject_*.csv")))
    
    if not valid_keys:
        print(f"❌ Error: No valid data to build master dataframe.")
        return None

    all_subjects_data = []

    # 1. 피험자별 데이터 병합
    print(f"🔗 Merging Valid subjects data...")
    for subj_idx, file_path in enumerate(csv_files, start=1):
        df_model = pd.read_csv(file_path)
        
        # 🌟 [핵심] 현재 피험자(subj_idx)의 유효한 scenario_id만 필터링
        valid_sc_ids = [sc_id for (s_id, sc_id) in valid_keys if s_id == subj_idx]
        
        if not valid_sc_ids:
            continue # 이 피험자는 정상적인 시나리오가 아예 없다면 건너뜀
            
        df_model_valid = df_model[df_model['scenario_id'].isin(valid_sc_ids)]

        # -------------------------------------------------------------
        # 🐞 [디버깅 추가] 시나리오 12번 병합(Merge) 과정 추적
        # -------------------------------------------------------------
        # (1) 일단 outer로 병합하고 indicator=True를 줘서 데이터의 출처('_merge')를 확인합니다.
        df_merged_debug = pd.merge(df_btom, df_model, 
                             on=['scenario_id', 'time_step'], 
                             how='outer', 
                             indicator=True)
        
        # (2) 시나리오 12번의 데이터가 어떻게 매칭되었는지 터미널에 출력 (피험자 1번일 때만)
        if subj_idx == 1:
            sc12_debug = df_merged_debug[df_merged_debug['scenario_id'] == 12]
            if not sc12_debug.empty:
                print(f"\n[DEBUG] Subject 1, Scenario 12 Merge Status:")
                # _merge 컬럼: 'both'(양쪽 다 있음), 'left_only'(df_btom에만 있음), 'right_only'(df_model에만 있음)
                print(sc12_debug[['time_step', '_merge']].head(15))
                print("-" * 50)
        # -------------------------------------------------------------
        
        # 필터링된 깨끗한 데이터만 inner merge (이제 inner를 써도 잘려나갈 걱정이 없음!)
        df_merged = pd.merge(df_btom, df_model_valid, 
                             on=['scenario_id', 'time_step'], 
                             how='inner')
        # 피험자 번호 명시
        df_merged.insert(0, 'subject_id', subj_idx)
        all_subjects_data.append(df_merged)

    # 2. 전체 마스터 데이터프레임 완성
    df_master = pd.concat(all_subjects_data, ignore_index=True)

    # 3. [핵심] Phase Labeling 로직 적용
    print("🏷️ Applying Phase Labeling...")
    df_master = apply_phase_labeling(df_master)

    # 4. 저장
    output_path = os.path.join(target_dir, "everystep_valid_only.csv")
    df_master.to_csv(output_path, index=False)
    print(f"✅ Master DataFrame saved: {output_path} (Shape: {df_master.shape})")
    
    return df_master

# 외부(run_analysis.py)에서 호출하기 위한 메인 래퍼 함수
def run_prepare_everystep(model_name, condition):
    print("\n" + "="*60)
    print("🚀 [Everystep] Valid-only DataFrame Builder Started")
    print("="*60)
    
    valid_keys, selected_subjects = get_valid_scenarios(model_name, condition)
    
    if valid_keys:
        build_master_dataframe(model_name, condition, valid_keys)
        
    return selected_subjects # 이 명단을 run_analysis로 전달!

# =========================================================================
# 3. BToM Everystep 데이터 로드 함수
# =========================================================================
def load_btom_everystep(mat_path=BTOM_EVERY_MAT_PATH):
    """
    MATLAB에서 추출한 BToM 모델의 매 스텝 데이터(.mat)를 불러와
    마스터 데이터프레임 구조와 똑같이 매핑하고 Phase를 라벨링합니다.
    """
    if not os.path.exists(mat_path):
        print(f"\n⚠️ Notice: BToM everystep data not found at {mat_path}. Skipping overlay.")
        return None
        
    print(f"\n📥 Loading BToM Everystep data from {mat_path}...")
    mat = scipy.io.loadmat(mat_path, squeeze_me=True)
    
    b_marg = mat['belief_marg'] 
    r_marg = mat['reward_marg']
    
    rows = []
    # 78개 시나리오 순회
    for ns in range(78):
        sc_id = ns + 1
        
        b_arr = b_marg[ns]
        r_arr = r_marg[ns]
        
        # 만약 time_step이 1개라서 1D 배열(크기 3)로 추출되었다면 2D(3, 1)로 변경
        if b_arr.ndim == 1: b_arr = b_arr.reshape(3, -1)
        if r_arr.ndim == 1: r_arr = r_arr.reshape(3, -1)
            
        path_len = b_arr.shape[1]
        
        for t in range(path_len):
            rows.append({
                'subject_id': 0, # BToM은 피험자 0번(정답)으로 취급
                'scenario_id': sc_id,
                'time_step': t + 1,
                'desire_K': r_arr[0, t],
                'desire_L': r_arr[1, t],
                'desire_M': r_arr[2, t],
                'belief_L': b_arr[0, t],
                'belief_M': b_arr[1, t],
                'belief_Empty': b_arr[2, t],
            })
            
    df_scores = pd.DataFrame(rows)
    
    # 궤적 정보(df_btom)와 합쳐서 agent_x, agent_y 등을 가져옴
    df_merged = pd.merge(df_btom, df_scores, on=['scenario_id', 'time_step'], how='inner')
    
    # LLM과 똑같은 기준으로 Phase 라벨링 수행
    print("  -> Applying Phase Labeling to BToM Ground Truth...")
    df_merged = apply_phase_labeling(df_merged)
    
    return df_merged

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., gpt-4o)")
    parser.add_argument("--condition", type=str, required=True, help="Condition (e.g., oneshot, reasoning)")
    args = parser.parse_args()

    run_prepare_everystep(args.model, args.condition)