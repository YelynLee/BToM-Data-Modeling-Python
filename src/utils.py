import os
import sys
import glob
import json
import pickle
import numpy as np
import pandas as pd

# 현재 스크립트(analysis 폴더)의 상위 경로를 파이썬 탐색 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) # 상위 폴더 (프로젝트 루트)

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

def get_clean_value(val):
    """
    MATLAB의 중첩된 Cell/Struct 배열에서 순수 데이터만 추출하는 강력한 함수.
    
    특징:
    1. (1, 1) 같은 불필요한 차원(껍질)은 벗깁니다.
    2. (3, 7) 같은 유의미한 행렬 데이터는 납작하게 펴지(Flatten) 않고 보존합니다.
    3. 데이터가 비어있거나(Empty), 0(Scalar)인 경우 None을 반환하여 체크하기 쉽게 합니다.
    """
    # 1. 입력이 배열이 아니면 그냥 반환
    if not isinstance(val, np.ndarray):
        return val
            
    # 2. 껍질 벗기기 반복문 (While Loop)
    # "배열인데 사이즈가 1개뿐이라면" 계속 안으로 파고듭니다.
    # 주의: [[1, 2]] 처럼 사이즈가 2 이상이면 멈춥니다.
    while isinstance(val, np.ndarray):
            
        # 데이터가 비어있으면 (MATLAB의 빈 cell) -> None 반환
        if val.size == 0:
            return None
            
        # 요소가 딱 1개인 경우에만 껍질을 벗김
        if val.size == 1:

            # 구조체(void type)이거나 필드명이 있으면 멈춤 (더 벗기면 깨짐)
            if val.dtype.names is not None:
                break
            
            # 0차원 스칼라가 아닐 때만 인덱싱
            if val.ndim > 0:
                val = val[0]
            else:
                # 0차원(스칼라)이면 값을 반환하고 종료
                val = val.item()
                break
        else:
            # 요소가 2개 이상(예: 3x7 행렬)이면 반복 종료 (Flatten 하지 않음!)
            break

    # 3. 예외 처리
    # 만약 꺼낸 값이 MATLAB의 빈 값을 의미하는 0(Scalar)이나 빈 배열이면 None 처리
    # (MATLAB loadmat은 빈 cell을 가끔 0.0으로 불러옵니다)
    if isinstance(val, (int, float)) and val == 0:
        return None
    if isinstance(val, np.ndarray) and val.size == 0:
        return None
            
    return val

def process_result_json(sc_id, meta, raw_response, model_name, condition, mode='normal'):
    """
    JSON 응답을 파싱하여 CSV용 Flat Dictionary(또는 List of Dictionaries)로 변환
    
    Args:
        raw_response (str or dict): api_client로부터 전달받은 raw string 또는 {"thinking":..., "text":...} 딕셔너리
        mode (str): 'normal' (Final decision) 또는 'everystep' (Step-by-step log)
    Returns:
        dict (if mode='normal') OR list (if mode='everystep')
    """
    # 전역 에러 핸들링 및 유연한 데이터 언패킹을 위해 초기화
    internal_thinking = ""
    raw_json_str = ""

    try:
        # 데이터 형태 방어 코드 (딕셔너리로 들어왔을 때와 문자열로 들어왔을 때 분기 처리)
        if isinstance(raw_response, dict):
            internal_thinking = raw_response.get("thinking", "")
            raw_json_str = raw_response.get("text", "")
        else:
            raw_json_str = raw_response

        # Markdown Code Block 제거
        clean_json = raw_json_str.replace("```json", "").replace("```", "").strip()
        
        # 가끔 모델이 [ ] 앞뒤로 텍스트를 붙이는 경우가 있어, 대괄호/중괄호 찾기
        if mode == 'everystep':

            # [Self-healing 로직]
            # 맨 앞뒤가 중괄호 {} 로 끝나는지 확인하고, 내부에 '},{' 패턴이 있다면 배열 괄호 누락으로 간주
            if clean_json.startswith("{") and clean_json.endswith("}"):
                if "},{" in clean_json.replace(" ", "").replace("\n", ""):
                    clean_json = f"[{clean_json}]"
                    # 디버깅용 메시지 (잘 작동하는지 확인하기 위해 당분간 남겨두는 것을 추천합니다)
                    print(f"💡 [Auto-Fix] Scenario {sc_id}: 누락된 배열 괄호 [ ]를 자동 복구했습니다.")

            # 기존 로직 (텍스트 앞뒤 잡동사니 제거)
            start = clean_json.find('[')
            end = clean_json.rfind(']') + 1
            if start != -1 and end != 0:
                clean_json = clean_json[start:end]
        else:
            # normal 모드일 때의 기존 코드 유지
            start = clean_json.find('{')
            end = clean_json.rfind('}') + 1
            if start != -1 and end != 0:
                clean_json = clean_json[start:end]

        data = json.loads(clean_json)

        # ---------------------------------------------------------
        # CASE A: Everystep Mode (List of Objects)
        # ---------------------------------------------------------
        if mode == 'everystep':
            
            # 껍질 벗기기 (Unwrapping)
            # 1. 모델이 {"steps": [...]} 처럼 딕셔너리 안에 리스트를 숨긴 경우
            if isinstance(data, dict):
                hidden_lists = [v for v in data.values() if isinstance(v, list)]
                if hidden_lists:
                    data = hidden_lists[0]  # 숨겨진 진짜 리스트를 꺼냄
                else:
                    data = [data]
            
            # 2. 우리가 강제로 []를 씌웠는데, 알고보니 [{"steps": [...]}] 였을 경우
            elif isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
                hidden_lists = [v for v in data[0].values() if isinstance(v, list)]
                if hidden_lists:
                    data = hidden_lists[0] # 숨겨진 진짜 리스트를 꺼냄
            
            parsed_list = []
            for item in data:

                # 엄격한 형식 검사 (Fail Fast)
                if 'time_step' not in item:
                    # time_step이 없다면 구조가 완전히 꼬인 것이므로, 강제로 파싱 에러를 발생시킴!
                    raise ValueError("JSON은 파싱되었으나 'time_step' 키가 없습니다. 모델의 응답 구조가 예상과 다릅니다.")

                # Everystep은 'reasoning' 필드가 하나로 통합되어 있거나 없을 수 있음(Vanilla)
                reasoning_text = item.get('reasoning', '')
                
                # Desire & Belief Scores 추출 (null 값이 들어와도 안전하게 빈 딕셔너리로 처리)
                desire = item.get('desire_scores') or {}
                belief = item.get('belief_scores') or {}
                
                # 행 데이터 생성
                row = {
                    'scenario_id': sc_id,
                    'time_step': item.get('time_step'), # Time Step 중요
                    'group_desc': meta['group_desc'],
                    'truck_presence': meta['truck_presence'],
                    'model': model_name,
                    'condition': condition,
                    'mode': mode,

                    # Everystep은 Desire/Belief 추론이 통합되어 있는 경우가 많음
                    # 분리되어 있다면 get으로 가져오고, 아니면 reasoning_text 사용
                    'desire_reasoning': item.get('desire_reasoning', reasoning_text),
                    'belief_reasoning': item.get('belief_reasoning', reasoning_text),

                    # Desire Columns
                    'desire_K': desire.get('K'),
                    'desire_L': desire.get('L'),
                    'desire_M': desire.get('M'),

                    # Belief Columns
                    'belief_K': belief.get('K'),
                    'belief_L': belief.get('L'),
                    'belief_M': belief.get('M'),
                    'belief_Empty': belief.get('Empty'),

                    # 🌟 내부 사고 추론 기록 컬럼 추가
                    'internal_thinking': internal_thinking
                }
                parsed_list.append(row)
            
            return parsed_list

        # ---------------------------------------------------------
        # CASE B: Normal Mode (Single Object - Final Decision)
        # ---------------------------------------------------------
        else:
            # Vanilla 조건은 reasoning 필드가 없을 수 있음 -> get으로 안전하게 처리
            d_reason = data.get('desire_reasoning', '')
            desire = data.get('desire_scores') or {}
            b_reason = data.get('belief_reasoning', '')
            belief = data.get('belief_scores') or {}
            
            return {
                'scenario_id': sc_id,
                'time_step': '',
                'group_desc': meta['group_desc'],
                'truck_presence': meta['truck_presence'],
                'model': model_name,
                'condition': condition,
                'mode': mode,

                # [추가] Reasoning Columns
                'desire_reasoning': d_reason,
                'belief_reasoning': b_reason,

                # Desire Columns (Flatten)
                'desire_K': desire.get('K'),
                'desire_L': desire.get('L'),
                'desire_M': desire.get('M'),
                
                # Belief Columns (Flatten)
                'belief_K': belief.get('K'),
                'belief_L': belief.get('L'),
                'belief_M': belief.get('M'),
                'belief_Empty': belief.get('Empty'),

                # 🌟 내부 사고 추론 기록 컬럼 추가
                'internal_thinking': internal_thinking
            }
        
    except Exception as e:
        return {
            'scenario_id': sc_id,
            'model': model_name,
            'condition': condition,
            'mode': mode,
            'error': str(e),
            'raw_response': raw_json_str,
            'internal_thinking': internal_thinking  # 에러가 발생해도 잡힌 추론 내역은 저장되도록 처리
        }
    
def inspect_pickle_data(file_path):
    """
    Pickle 파일을 로드하여 데이터 구조, 타입, Shape, 결측치(NaN) 등을 검사합니다.
    """
    print(f"\n{'='*60}")
    print(f"🔍 Inspecting Pickle: {file_path}")
    print(f"{'='*60}")

    if not os.path.exists(file_path):
        print(f"❌ Error: File not found at {file_path}")
        return

    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        print(f"✅ Load Success! Type: {type(data)}\n")
        
        if not isinstance(data, dict):
            print("⚠️ Warning: Data is not a dictionary.")
            return

        # 1. 키(Key) 및 Shape 요약
        print(f"{'Key Name':<25} | {'Type':<15} | {'Shape/Len':<15}")
        print("-" * 60)
        
        for key, value in data.items():
            v_type = type(value).__name__
            v_shape = "N/A"
            
            if isinstance(value, np.ndarray):
                v_shape = str(value.shape)
            elif isinstance(value, list):
                v_shape = f"len={len(value)}"
            
            print(f"{key:<25} | {v_type:<15} | {v_shape:<15}")

        # 2. 데이터 무결성 체크 (Data Integrity Check)
        print("-" * 60)
        print("📊 Data Integrity Check:")
        
        # (A) Desire Mean Check
        if 'des_inf_mean' in data:
            dm = data['des_inf_mean']
            nan_count = np.isnan(dm).sum()
            print(f"\n[1] Desire Mean (des_inf_mean) - First 5 Scenarios:")
            print(dm[:, :5]) 
            print(f"   -> Total NaNs: {nan_count}")
            if nan_count > 0:
                print("   ⚠️ Alert: NaNs found in mean! Some scenarios might have failed.")

        # (B) Belief Mean Check
        if 'bel_inf_mean_norm' in data:
            bm = data['bel_inf_mean_norm']
            col_sums = np.nansum(bm[:, :5], axis=0)
            print(f"\n[2] Belief Mean Normalized - First 5 Scenarios:")
            print(bm[:, :5])
            print(f"   -> Column Sums (Target ~1.0): {np.round(col_sums, 2)}")

        # (C) Raw Data Sample
        if 'des_inf' in data:
            raw_d = data['des_inf']
            # shape가 (Rating, Condition, Subject)라고 가정
            n_subj = raw_d.shape[2] if len(raw_d.shape) > 2 else 0
            print(f"\n[3] Raw Desire Data (des_inf) - Scenario 1, First {min(3, n_subj)} Subjects:")
            if n_subj > 0:
                print(raw_d[:, 0, :min(3, n_subj)]) 

    except Exception as e:
        print(f"❌ Error reading pickle: {e}")

def get_valid_scenarios(model_name, condition):
    """
    데이터 정합성 검토 및 '유효한 시나리오' 추출:
    df_btom과 완벽하게 time_step 개수가 일치하는 (subject_id, scenario_id) 쌍만
    추출하여 set 형태로 반환합니다.
    """
    from src.dataset import df_btom
    from src.config import BASE_RESULTS_DIR, get_group_indices

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

def build_master_dataframe(model_name, condition, valid_keys):
    """
    df_btom과 모델의 Everystep 결과를 결합하여 마스터 데이터프레임을 생성합니다.
    """
    from src.dataset import df_btom
    from src.config import BASE_RESULTS_DIR
    from src.prepare_everystep import apply_phase_labeling

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
        
        # 현재 피험자(subj_idx)의 유효한 scenario_id만 필터링
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
