import os
import json
import pickle
import numpy as np

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

def process_result_json(sc_id, meta, raw_json_str, model_name, condition, mode='normal'):
    """
    JSON 응답을 파싱하여 CSV용 Flat Dictionary(또는 List of Dictionaries)로 변환
    
    Args:
        mode (str): 'normal' (Final decision) 또는 'everystep' (Step-by-step log)
    Returns:
        dict (if mode='normal') OR list (if mode='everystep')
    """
    try:
        # Markdown Code Block 제거 (o1 대응)
        clean_json = raw_json_str.replace("```json", "").replace("```", "").strip()
        
        # 가끔 모델이 [ ] 앞뒤로 텍스트를 붙이는 경우가 있어, 대괄호/중괄호 찾기
        if mode == 'everystep':
            start = clean_json.find('[')
            end = clean_json.rfind(']') + 1
            if start != -1 and end != 0:
                clean_json = clean_json[start:end]
        else:
            start = clean_json.find('{')
            end = clean_json.rfind('}') + 1
            if start != -1 and end != 0:
                clean_json = clean_json[start:end]

        data = json.loads(clean_json)

        # ---------------------------------------------------------
        # CASE A: Everystep Mode (List of Objects)
        # ---------------------------------------------------------
        if mode == 'everystep':
            # 만약 모델이 리스트가 아니라 단일 객체로 줬다면 리스트로 감싸기
            if isinstance(data, dict):
                data = [data]
            
            parsed_list = []
            for item in data:
                # Everystep은 'reasoning' 필드가 하나로 통합되어 있거나 없을 수 있음(Vanilla)
                reasoning_text = item.get('reasoning', '')
                
                # Desire & Belief Scores 추출
                desire = item.get('desire_scores', {})
                belief = item.get('belief_scores', {})
                
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
                    'belief_Empty': belief.get('Empty')
                }
                parsed_list.append(row)
            
            return parsed_list

        # ---------------------------------------------------------
        # CASE B: Normal Mode (Single Object - Final Decision)
        # ---------------------------------------------------------
        else:
            # Vanilla 조건은 reasoning 필드가 없을 수 있음 -> get으로 안전하게 처리
            d_reason = data.get('desire_reasoning', '')
            desire = data.get('desire_scores', {})
            b_reason = data.get('belief_reasoning', '')
            belief = data.get('belief_scores', {})
            
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
                'belief_Empty': belief.get('Empty')
            }
        
    except Exception as e:
        return {
            'scenario_id': sc_id,
            'model': model_name,
            'condition': condition,
            'mode': mode,
            'error': str(e),
            'raw_response': raw_json_str
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
