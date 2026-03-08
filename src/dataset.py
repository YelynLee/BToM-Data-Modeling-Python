import scipy.io
import numpy as np
import pandas as pd

# 공통 설정 가져오기
from src.config import get_group_indices, BEHAVIOR_GROUPS, STIMULI_MAT_PATH
from src.utils import get_clean_value

def extract_btom_data(mat_file_path):
    try:
        mat_data = scipy.io.loadmat(mat_file_path)
    except FileNotFoundError:
        return "파일을 찾을 수 없습니다."
    
    # --- 그룹 매핑 사전 미리 생성 (모든 데이터 로딩을 위해 True 사용) ---
    all_groups = get_group_indices(include_irrational=True)
    SCENARIO_TO_GROUP = {}
    for g_idx, ids in enumerate(all_groups):
        for s_id in ids:
            SCENARIO_TO_GROUP[s_id] = g_idx + 1

    # =========================================================================
    # 만능 껍질 벗기기 함수 (Universal Unwrapper)
    # =========================================================================
    
    def get_flatten(val):
        # 2차원 이상의 배열(행렬)을 1차원 벡터로 평탄화
        # 예: [[15], [5]] (2x1) -> [15, 5] (1차원)
        # 예: [[1, 2, 3]] (1x3) -> [1, 2, 3] (1차원)
        if isinstance(val, np.ndarray) and val.ndim > 1:
            val = val.flatten()
            
        return val

    def get_target_world_struct(world_container, world_idx):
        """
        복잡하게 포장된 World 컨테이너에서 
        인덱스에 맞는 '진짜 구조체(Struct)' 하나만 쏙 꺼내는 전용 함수
        """
        # 1. 3개짜리 리스트가 나올 때까지 껍질 벗기기
        current = world_container
        while True:
            # 리스트나 배열이 아니면 중단
            if not isinstance(current, (np.ndarray, list)):
                break
        
            # 길이가 3인 배열/리스트를 찾았다! -> 여기서 인덱싱
            if len(current) == 3 or (isinstance(current, np.ndarray) and current.size == 3):
                # 배열이면 평탄화 후 선택, 리스트면 그냥 선택
                if isinstance(current, np.ndarray):
                    current = current.flatten()
            
                # 인덱스 안전장치
                idx = world_idx if 0 <= world_idx < 3 else 0
                selected = current[idx]
            
                # 선택된 녀석의 껍질을 다시 벗겨서 '구조체'로 만듦
                return get_flatten(get_clean_value(selected)) # 여기서 get_clean_value 재활용!
        
            # 길이가 1이면 계속 안으로 진입
            if len(current) == 1 or (isinstance(current, np.ndarray) and current.size == 1):
                if isinstance(current, list):
                    current = current[0]
                else:
                    current = current.flatten()[0] # flatten()[0]이 안전
            else:
                # 길이가 0이거나 이상하면 중단
                break
            
        return None # 실패 시

    def check_visibility(agent_pos, target_pos, wall_rect):
        """
        agent_pos: [x, y]
        target_pos: [x, y]
        wall_rect: [start_x, start_y, width, height]
        반환값: 1 (보임), 0 (가려짐)
        """
        ax, ay = agent_pos
        tx, ty = target_pos
        wx, wy, ww, wh = wall_rect
    
        dist = np.sqrt((tx - ax)**2 + (ty - ay)**2)
        if dist == 0: return 1 
    
        steps = int(np.ceil(dist))
        wall_x_min, wall_x_max = wx, wx + ww - 1
        wall_y_min, wall_y_max = wy, wy + wh - 1
    
        for i in range(1, steps): 
            t = i / steps
            sample_x = ax + (tx - ax) * t
            sample_y = ay + (ty - ay) * t
            grid_x = round(sample_x)
            grid_y = round(sample_y)
        
            if (wall_x_min <= grid_x <= wall_x_max) and (wall_y_min <= grid_y <= wall_y_max):
                return 0 
        return 1 

    def process_scenario_data(df):
        """
        DataFrame에 Goal 좌표를 추가하고, 
        '주차장 위치'를 기준으로 시야(Visibility)를 계산하여 컬럼을 추가함.
        """
    
        # 1. 고정된 주차장 좌표 설정 (Instruction에 따라 고정됨)
        # goal_space = [1, 15; 1, 5] -> Goal 1: (1,1), Goal 2: (15,5)
        # 모든 행에 동일하게 적용 (방송)
        df['goal1_x'] = 1
        df['goal1_y'] = 1
        df['goal2_x'] = 15
        df['goal2_y'] = 5
    
        vis_goal1 = []
        vis_goal2 = []
    
        for _, row in df.iterrows():
            agent_pos = [row['agent_x'], row['agent_y']]
            wall_rect = [row['wall_start_x'], row['wall_start_y'], 
                        row['wall_width'], row['wall_height']]
        
            # 2. Goal 1 (1,1)에 대한 시야 체크
            # 트럭이 있든 없든, '그 자리'가 보이는지 확인
            v1 = check_visibility(agent_pos, [1, 1], wall_rect)
            vis_goal1.append(v1)
        
            # 3. Goal 2 (15,5)에 대한 시야 체크
            v2 = check_visibility(agent_pos, [15, 5], wall_rect)
            vis_goal2.append(v2)
        
        df['visible_goal1'] = vis_goal1
        df['visible_goal2'] = vis_goal2
    
        return df
    

    # =========================================================================
    # 데이터 추출 로직
    # =========================================================================

    # 'scenario' 데이터 가져오기
    scenarios = mat_data['scenario'] # (1, 78) cell array
    extracted_data = []

    # Grid Size (15x5)
    width, height = 15, 5

    # 전체 78개 시나리오 반복
    for i in range(scenarios.shape[1]):
        # 각 시나리오는 (1, 1) 형태의 struct입니다.
        curr_scenario = scenarios[0, i]
        scenario_id = i + 1
        
        # 1. Irrational 제거 X -> 대신 컬럼으로 기록
        is_irrational = get_flatten(get_clean_value(curr_scenario['irrational']))
        # if is_irrational == 1:
        #     continue  # 합리적이지 않은 5개 시나리오는 패스
            
        # 2. Path (경로) 추출 및 좌표 변환
        # MATLAB은 1부터 시작하고, Column-major 순서로 인덱싱합니다.
        raw_path = get_flatten(get_clean_value(curr_scenario['path']))
        # print('path:', raw_path, type(raw_path))

        # 유효하지 않은 인덱스(76 등 종료 코드) 제거
        clean_path = raw_path[raw_path <= (width * height)]

        # 좌표 변환 (0-based, divmod 사용)
        # 몫=Y(Row), 나머지=X(Col) -> MATLAB은 Row-Major, Bottom-Left 기준
        path_indices = clean_path - 1
        y_indices, x_indices = divmod(path_indices, width)
        path_x = x_indices + 1
        path_y = y_indices + 1

        # 확인용 출력 (ID 18일 때)
        # if i == 17:
        #     print('확인용: ', [path_x, path_y])
        # 결과 예상: [(2, 2), (2, 1), (1, 1)]
        
        # 3. World 정보 추출
        # 1. 어떤 World를 쓸지 인덱스 계산
        condition_vec = get_flatten(get_clean_value(curr_scenario['condition']))
        try:
            true_world_idx = int(condition_vec[1]) - 1
        except:
            true_world_idx = 0
            
        # 2. 헬퍼 함수로 '구조체' 한 방에 추출
        target_world = get_target_world_struct(curr_scenario['world'], true_world_idx)
        # print('target world이란', target_world)
        
        if target_world is None or target_world.dtype.names is None:
            continue # 구조체 찾기 실패 시 건너뜀
        
        # --- 장애물(Wall) 정보 추출 ---
        # obst_pose: [x, y] (시작 좌표)
        # obst_sz: [w, h] (크기)
        obst_pose = get_flatten(get_clean_value(target_world['obst_pose']))
        # print('what is obst_pose:', obst_pose)
        obst_sz = get_flatten(get_clean_value(target_world['obst_sz']))
        # print('what is obst_size:', obst_sz)
        
        # 값이 없거나 이상하면 0으로 처리
        if obst_pose is None or np.isscalar(obst_pose): obst_pose = [0, 0]
        if obst_sz is None or np.isscalar(obst_sz): obst_sz = [0, 0]

        # --- Truck (Goal) 위치 추출 ---
        raw_goal_pose = target_world['goal_pose']
        
        # [단계 1] 무조건 길이 3인 1차원 리스트/배열로 만들기 (Flatten)
        # goal_pose는 3개의 트럭 정보를 담고 있어야 합니다.
        # [[K, L, M]] 형태이든, [K, L, M] 형태이든 무조건 1차원으로 폅니다.
        
        truck_list = raw_goal_pose
        
        # 껍질 벗기기: 배열인데 사이즈가 3이 아니면(포장되어 있으면) 계속 벗김
        while isinstance(truck_list, np.ndarray):
            # 우리가 원하는 건 size가 3인 1차원 배열
            if truck_list.size == 3:
                truck_list = truck_list.flatten() # (1, 3) -> (3,)
                break
            
            # size가 3이 아닌데(예: 1) 배열이면 껍질 벗김
            if truck_list.size == 1:
                truck_list = truck_list.flatten()[0]
            else:
                # size가 0이거나 이상한 경우 (데이터 없음)
                truck_list = []
                break
        
        # [단계 2] K, L, M 순서대로 좌표 뽑기
        truck_locs = {}
        truck_names = ['K', 'L', 'M'] # 순서는 사용자가 확인 필요 (K, M, L일수도 있음)

        # truck_list는 이제 길이가 3인 배열이어야 함
        for t_idx, t_name in enumerate(truck_names):
            if t_idx < len(truck_list):
                pos = get_flatten(get_clean_value(truck_list[t_idx]))
                # print(f'{i}th what is pos:', pos)
                
                if pos is not None and hasattr(pos, 'size') and pos.size >= 2:
                    truck_locs[f'{t_name}_x'] = pos[0]
                    truck_locs[f'{t_name}_y'] = pos[1]
                else:
                    truck_locs[f'{t_name}_x'] = 0
                    truck_locs[f'{t_name}_y'] = 0
            else:
                truck_locs[f'{t_name}_x'] = 0
                truck_locs[f'{t_name}_y'] = 0

        # 4. 행동 그룹 정보 가져오기
        group_num = SCENARIO_TO_GROUP.get(scenario_id, 0)
        group_desc = BEHAVIOR_GROUPS.get(group_num, "Unknown")

        # 5. 데이터 저장 (Time Step별로 Row 생성)
        # Tiny RNN 학습을 위해 매 시간(t)의 상태를 저장
        for t in range(len(path_x)):
            row = {
                'scenario_id': scenario_id,
                'time_step': t + 1,

                # 에이전트 상태
                'agent_x': path_x[t],
                'agent_y': path_y[t],

                # 행동 그룹 (Label)
                'group_id': group_num,
                'group_desc': group_desc,
                'is_irrational': is_irrational,

                # 장애물(Wall) 정보
                'wall_start_x': obst_pose[0],
                'wall_start_y': obst_pose[1],
                'wall_width': obst_sz[0],
                'wall_height': obst_sz[1],

                # 트럭(Goal) 위치
                'K_x': truck_locs['K_x'], 'K_y': truck_locs['K_y'],
                'L_x': truck_locs['L_x'], 'L_y': truck_locs['L_y'],
                'M_x': truck_locs['M_x'], 'M_y': truck_locs['M_y'],
            }
            extracted_data.append(row)

    # DataFrame 생성
    df = pd.DataFrame(extracted_data)
    df_final = process_scenario_data(df)

    return df_final

# 사용 예시
# 핵심! 이 스크립트가 '메인'으로 실행될 때만 동작하는 코드와
# 외부에서 import 될 때 실행될 코드를 구분합니다.

if __name__ == "__main__":
    # 이 파일을 직접 실행했을 때 (테스트용)
    print("데이터셋 추출을 시작합니다...")

    df_btom = extract_btom_data(STIMULI_MAT_PATH)

    print(f'Done! Total Rows: {len(df_btom)}')
    print('And columns: \n', df_btom.columns)
    print(f'Unique Scenarios: {df_btom["scenario_id"].nunique()}')
    # Irrational 포함 여부 확인
    irrational_count = df_btom[df_btom['is_irrational'] == 1]['scenario_id'].nunique()
    print(f'Irrational Scenarios included: {irrational_count} (Should be 5)')

    # Scenario 18로 verification
    # 모든 컬럼과 행이 잘리지 않고 보이도록 옵션 설정
    # pd.set_option('display.max_columns', None)  # 모든 열 보기
    # pd.set_option('display.max_rows', None)     # 모든 행 보기 (Time Step이 길어도 다 나옴)
    # pd.set_option('display.width', 1000)        # 가로 폭 늘리기 (줄바꿈 방지)

    # target_scenario = df_btom[df_btom['scenario_id'] == 18].copy()

    # Time Step 순서대로 정렬 (혹시 섞여 있을 경우를 대비)
    # target_scenario = target_scenario.sort_values(by='time_step')

    # 전체 데이터 출력
    # print("=== Scenario 18 Full Data ===")
    # print(target_scenario)

else:
    # 다른 파일에서 'import btom_dataset' 했을 때 실행됨
    # 여기서 df_btom을 생성해두면, import 하는 쪽에서 'btom_dataset.df_btom'으로 쓸 수 있음
    try:
        df_btom = extract_btom_data(STIMULI_MAT_PATH)
    except Exception as e:
        print(f"데이터 로드 실패: {e}")
        df_btom = None