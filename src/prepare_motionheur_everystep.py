import numpy as np
import pandas as pd
from collections import deque
from sklearn.linear_model import LinearRegression

# =========================================================================
# 1. BFS 룩업 테이블 생성기 (타겟 중심 확산)
# =========================================================================
def build_distance_lookup(target_x, target_y, wx, wy, ww, wh):
    """
    타겟(목표 지점)으로부터 15x5 맵 전체 셀로 확산하며 BFS 최단 거리를 룩업 테이블로 기록.
    반환: (16, 6) 크기의 2D Numpy 배열 (1-based index 사용을 위해 여유 공간 확보)
    """
    dist_map = np.full((16, 6), np.inf)
    
    if pd.isna(wx) or pd.isna(ww):
        wx, wy, ww, wh = -1, -1, 0, 0
    else:
        wx, wy, ww, wh = int(wx), int(wy), int(ww), int(wh)

    tx, ty = int(target_x), int(target_y)
    queue = deque([(tx, ty, 0)])
    dist_map[tx, ty] = 0
    
    while queue:
        cx, cy, dist = queue.popleft()
        
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + dx, cy + dy
            
            if 1 <= nx <= 15 and 1 <= ny <= 5:
                if wx <= nx < wx + ww and wy <= ny < wy + wh:
                    continue
                if np.isinf(dist_map[nx, ny]):
                    dist_map[nx, ny] = dist + 1
                    queue.append((nx, ny, dist + 1))
                    
    return dist_map

# =========================================================================
# 2. 경로별/스텝별 거리 변화량(ΔDist) 추출기
# =========================================================================
def get_path_diffs(path_coords, dist_map_lower, dist_map_upper):
    """
    주어진 궤적(x, y 리스트)에 대해 Lower, Upper까지의 스텝별 거리 변화량을 반환
    path_coords: [(x1, y1), (x2, y2), ...]
    """
    lower_dists = [dist_map_lower[int(x), int(y)] for x, y in path_coords]
    upper_dists = [dist_map_upper[int(x), int(y)] for x, y in path_coords]
    
    # 스텝별 변화량 (t+1 - t)
    # 값이 음수면 가까워짐(Towards), 양수면 멀어짐(Away)
    # 패딩(0.0)을 앞에 추가하여 t=1(시작점) 인덱스 밀림 방지
    lower_diffs = np.insert(np.diff(lower_dists), 0, 0.0)
    upper_diffs = np.insert(np.diff(upper_dists), 0, 0.0)

    return lower_diffs, upper_diffs

# =========================================================================
# 3. Desire & Belief 디자인 매트릭스(234행) 조립
# =========================================================================
def build_features_for_scenario(mean_lower, mean_upper, world, sc_id, model_type="Desire"):
    """
    특정 시나리오의 평균 변화량을 바탕으로 3개 트럭에 대한 [I, O, f3, f4] 리스트 반환
    world: 1 (Upper에 L 있음), 0 (Upper에 N 빈 공간)
    """
    
    # World가 1일 때만 Upper를 향하는 것을 유효한 대상 접근으로 봄
    upper_effective = mean_upper if world == 1 else 0.0
    
    features = []
    
    if model_type == "Desire":
        # R1 (트럭 K)
        features.append({"scenario_id": sc_id, "truck": "K", "I": mean_lower, "O": upper_effective, "f3": 1, "f4": 0})
        # R2 (트럭 L)
        features.append({"scenario_id": sc_id, "truck": "L", "I": upper_effective, "O": mean_lower, "f3": 0, "f4": 1})
        # R3 (트럭 M)
        features.append({"scenario_id": sc_id, "truck": "M", "I": 0.0, "O": mean_lower + upper_effective, "f3": 0, "f4": 0})
    
    elif model_type == "Belief":
        # R1 (월드 L)
        features.append({"scenario_id": sc_id, "truck": "L", "I": upper_effective, "O": mean_lower, "f3": 1, "f4": 0})
        # R2 (월드 M)
        features.append({"scenario_id": sc_id, "truck": "M", "I": 0.0, "O": mean_lower + upper_effective, "f3": 0, "f4": 1})
        # R3 (월드 N)
        features.append({"scenario_id": sc_id, "truck": "N", "I": mean_lower, "O": mean_upper, "f3": 0, "f4": 0})
        
    return features

# =========================================================================
# 4. 선형 점수 확률(Softmax) 정규화
# =========================================================================
def normalize_to_probability(scores):
    scores = np.array(scores)
    exp_scores = np.exp(scores - np.max(scores))
    return exp_scores / np.sum(exp_scores)

# =========================================================================
# 5. 가중치 추정 및 스텝별(Online) 점수 계산
# =========================================================================
def generate_mh_scores(df_base, df_human_target, world_mapping, exclude_irrational=True):
    """
    전체 데이터(df_train)로 회귀 피팅 후, 특정 경로의 스텝별 점수 리스트 산출
    df_base: BToM 궤적 데이터
    df_human_target: ['scenario_id', 'model_type', 'truck', 'value'] 형태의 사람 응답 정답지
    world_mapping: {scenario_id: 1(Upper=L) or 0(Upper=N)}
    """
    # ---------------------------------------------------------------------
    # Phase 1: 전체 데이터 기반 디자인 매트릭스 구축 및 회귀(Fit)
    # ---------------------------------------------------------------------
    desire_features = []
    belief_features = []
    scenario_cache = {} # Phase 2 연산 최적화를 위한 데이터 캐싱
    
    grouped = df_base.groupby('scenario_id')
    for sc_id, group_data in grouped:
        first_row = group_data.iloc[0]
        wx, wy, ww, wh = first_row['wall_start_x'], first_row['wall_start_y'], first_row['wall_width'], first_row['wall_height']
        
        dist_map_lower = build_distance_lookup(1, 1, wx, wy, ww, wh)
        dist_map_upper = build_distance_lookup(15, 5, wx, wy, ww, wh)
        
        path_coords = list(zip(group_data['agent_x'], group_data['agent_y']))
        lower_diffs, upper_diffs = get_path_diffs(path_coords, dist_map_lower, dist_map_upper)
        
        world = world_mapping.get(sc_id, 1)
        
        # 궤적 전체의 평균 거리 변화량 산출 (0으로 패딩된 t=1 제외)
        mean_lower = np.mean(lower_diffs[1:]) if len(lower_diffs) > 1 else 0.0
        mean_upper = np.mean(upper_diffs[1:]) if len(upper_diffs) > 1 else 0.0
        
        desire_features.extend(build_features_for_scenario(mean_lower, mean_upper, world, sc_id, "Desire"))
        belief_features.extend(build_features_for_scenario(mean_lower, mean_upper, world, sc_id, "Belief"))
        
        scenario_cache[sc_id] = (lower_diffs, upper_diffs, world)

    df_d_train = pd.DataFrame(desire_features)
    df_b_train = pd.DataFrame(belief_features)
    
    # 인간 응답(value) 병합 (Target 변수 Y 생성)
    df_d_train = pd.merge(df_d_train, df_human_target[df_human_target['model_type'] == 'Desire'], on=['scenario_id', 'truck'])
    df_b_train = pd.merge(df_b_train, df_human_target[df_human_target['model_type'] == 'Belief'], on=['scenario_id', 'truck'])
    
    # 비합리적 경로(Irrational trials) 학습 제외 처리 (R 코드 논리 반영)
    irrational_scenarios = [11, 12, 22, 71, 72]
    if exclude_irrational:
        df_d_train = df_d_train[~df_d_train['scenario_id'].isin(irrational_scenarios)]
        df_b_train = df_b_train[~df_b_train['scenario_id'].isin(irrational_scenarios)]

    # 선형 회귀 학습 (Learning Weights)
    reg_D = LinearRegression().fit(df_d_train[['I', 'O', 'f3', 'f4']], df_d_train['value'])
    reg_B = LinearRegression().fit(df_b_train[['I', 'O', 'f3', 'f4']], df_b_train['value'])
    
    w_d, int_d = reg_D.coef_, reg_D.intercept_
    w_b, int_b = reg_B.coef_, reg_B.intercept_

    # =====================================================================
    # 💡 [추가할 코드] 학습된 가중치를 콘솔에 출력하여 확인
    # =====================================================================
    print("\n" + "-"*50)
    print("🔍 [MotionHeuristic Learned Weights]")
    print(f" 🎯 DESIRE Model (Scale: 1~7):")
    print(f"    w_I (Into):   {w_d[0]:.4f}")
    print(f"    w_O (Other):  {w_d[1]:.4f}")
    print(f"    w_f3 (Truck): {w_d[2]:.4f}")
    print(f"    w_f4 (Truck): {w_d[3]:.4f}")
    print(f"    Intercept:    {int_d:.4f}")
    print(f"\n 🧠 BELIEF Model (Scale: Pre-Softmax):")
    print(f"    w_I (Into):   {w_b[0]:.4f}")
    print(f"    w_O (Other):  {w_b[1]:.4f}")
    print(f"    w_f3 (World): {w_b[2]:.4f}")
    print(f"    w_f4 (World): {w_b[3]:.4f}")
    print(f"    Intercept:    {int_b:.4f}")
    print("-" * 50 + "\n")
    # =====================================================================

    # ---------------------------------------------------------------------
    # Phase 2: 도출된 가중치 적용 및 스텝별(Online) 점수 계산
    # ---------------------------------------------------------------------
    df_scores = df_base.copy()
    desire_K_list, desire_L_list, desire_M_list = [], [], []
    belief_L_list, belief_M_list, belief_Empty_list = [], [], []
    
    for sc_id, group_data in grouped:
        lower_diffs, upper_diffs, world = scenario_cache[sc_id]
        
        # 누적 평균(Cumulative mean) 시뮬레이션
        cum_mean_lower = np.cumsum(lower_diffs) / np.arange(1, len(lower_diffs) + 1)
        cum_mean_upper = np.cumsum(upper_diffs) / np.arange(1, len(upper_diffs) + 1)
        
        for t in range(len(cum_mean_lower)):
            m_lower = cum_mean_lower[t]
            m_upper = cum_mean_upper[t]
            u_eff = m_upper if world == 1 else 0.0
            
            # Desire 예측 수식 계산
            s_d_K = (w_d[0]*m_lower) + (w_d[1]*u_eff) + (w_d[2]*1) + (w_d[3]*0) + int_d
            s_d_L = (w_d[0]*u_eff) + (w_d[1]*m_lower) + (w_d[2]*0) + (w_d[3]*1) + int_d
            s_d_M = (w_d[0]*0.0) + (w_d[1]*(m_lower + u_eff)) + int_d
            
            # normalize_to_probability 제거하고 바로 리스트에 추가 (1~7 스케일 유지)
            desire_K_list.append(s_d_K)
            desire_L_list.append(s_d_L)
            desire_M_list.append(s_d_M)
            
            # Belief 예측 수식 계산
            s_b_L = (w_b[0]*u_eff) + (w_b[1]*m_lower) + (w_b[2]*1) + (w_b[3]*0) + int_b
            s_b_M = (w_b[0]*0.0) + (w_b[1]*(m_lower + u_eff)) + (w_b[2]*0) + (w_b[3]*1) + int_b
            s_b_N = (w_b[0]*m_lower) + (w_b[1]*m_upper) + int_b
            
            # 💡 [핵심 수정] Softmax 제거! 
            # 선형 예측값이 이미 0~1을 타겟으로 학습되었으므로, 비율(Ratio)만 맞춰줍니다.
            raw_b = np.array([s_b_L, s_b_M, s_b_N])
            raw_b = np.maximum(raw_b, 0) # 음수 방지 (선형 모델의 한계 보완)
            
            sum_b = np.sum(raw_b)
            if sum_b > 0:
                norm_b = raw_b / sum_b
            else:
                norm_b = [1/3, 1/3, 1/3] # 모두 0 이하로 떨어질 경우 예외 처리
                
            belief_L_list.append(norm_b[0])
            belief_M_list.append(norm_b[1])
            belief_Empty_list.append(norm_b[2])

    df_scores['desire_K'] = desire_K_list
    df_scores['desire_L'] = desire_L_list
    df_scores['desire_M'] = desire_M_list
    df_scores['belief_L'] = belief_L_list
    df_scores['belief_M'] = belief_M_list
    df_scores['belief_Empty'] = belief_Empty_list
    
    return df_scores