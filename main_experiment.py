import os
import time
import argparse
import pandas as pd
from tqdm import tqdm
from src.dataset import df_btom
from src.prompts import generate_scenario_prompt
from src.api_client import call_model_api
from src.utils import process_result_json, inspect_pickle_data
from src.config import BASE_RESULTS_DIR
from src.data_processor import process_model_results

def run_experiment(model_name, condition, mode, num_subjects=16):
    print(f"🚀 실험 시작: Model=[{model_name}], Condition=[{condition}], Mode=[{mode}], Subjects=[{num_subjects}]")
    
    # 저장 경로 자동 생성: results/{model_name}/{condition}/
    # Everystep: results/gpt-4o/reasoning/everystep/
    if mode == "everystep":
        save_dir = os.path.join(BASE_RESULTS_DIR, model_name, condition, "everystep")
    else:
        save_dir = os.path.join(BASE_RESULTS_DIR, model_name, condition)

    os.makedirs(save_dir, exist_ok=True)
    print(f"📂 결과 저장 경로: {save_dir}")

    scenario_groups = list(df_btom.groupby('scenario_id'))

    for subject_idx in range(1, num_subjects + 1):

        # 🌟 [체크포인트] 현재 피험자의 최종 저장될 파일명 미리 정의
        filename = os.path.join(save_dir, f"subject_{subject_idx:02d}.csv")
        
        # 🌟 [여기에 이 두 줄을 추가해 보세요!]
        absolute_path = os.path.abspath(filename)
        print(f"🔍 [경로 확인] 파이썬이 찾고 있는 곳: {absolute_path}")

        # 🌟 [체크포인트] 파일이 이미 존재하면 실험을 건너뜀 (Skip)
        if os.path.exists(filename):
            print(f"\n⏩ Subject {subject_idx:02d}/{num_subjects} 이미 완료됨. 건너뜁니다! ({filename})")
            continue

        print(f"\n=== Subject {subject_idx}/{num_subjects} 진행 중 ===")
        results = []
        
        for sc_id, group_df in tqdm(scenario_groups, desc=f"Subj {subject_idx}"):
            # 1. 메타데이터 추출
            row0 = group_df.iloc[0]
            present_trucks = [t for t, k in [('K', 'K'), ('L', 'L'), ('M', 'M')] 
                              if row0[f'{k}_x'] != 0 or row0[f'{k}_y'] != 0]
            meta = {
                'group_desc': row0['group_desc'],
                'truck_presence': " and ".join(present_trucks) + " present" if present_trucks else "No trucks present"
            }
            
            # 2. 프롬프트 생성 (조건 반영)
            sys_prompt, user_prompt = generate_scenario_prompt(group_df, condition, mode)
            
            # 3. 모델 호출
            response_str = call_model_api(model_name, sys_prompt, user_prompt)
            
            # 4. 결과 처리
            if response_str:
                res = process_result_json(sc_id, meta, response_str, model_name, condition, mode)
                
                # [버그 수정 1] everystep이면 list가 오므로 extend를 사용, normal이면 dict이므로 append 사용
                if isinstance(res, list):
                    results.extend(res)
                else:
                    results.append(res)

                # [디버깅 코드] 만약 파싱 에러가 났다면 화면에 출력
                if isinstance(res, dict) and 'error' in res:
                    print(f"\n⚠️ [Parsing Error] Scenario {sc_id}: {res['error']}")
                    print(f"Raw Response: {res['raw_response']}") # 주석 해제 시 원본 텍스트 확인 가능

            else:
                results.append({'scenario_id': sc_id, 'error': 'API Fail', 'model': model_name})
            
            # Rate Limit 방지 (o1은 더 길게)
            time.sleep(2 if "o1" in model_name else 0.5)
        
        # 5. 파일 저장 (Subject 단위)
        if not results:
            print(f"⚠️ Subject {subject_idx}: 저장할 데이터가 없습니다.")
            continue

        df_res = pd.DataFrame(results)
        
        # 컬럼 순서 정렬 (보기 좋게)
        cols = ['scenario_id', 'time_step', 'group_desc', 'truck_presence', 'model', 'condition', 'mode',
                'desire_reasoning', 'desire_K', 'desire_L', 'desire_M', 
                'belief_reasoning', 'belief_K', 'belief_L', 'belief_M', 'belief_Empty']
        
        # 결과에 있는 컬럼만 필터링 (에러 시 일부 컬럼 없을 수 있음)
        final_cols = [c for c in cols if c in df_res.columns]
        df_res = df_res[final_cols]
        
        # 파일 저장 (체크포인트 검사용 파일 생성)
        df_res.to_csv(filename, index=False)
        print(f"✅ Subject {subject_idx} Saved.")


    print("\n✨ 모든 실험이 성공적으로 종료되었습니다!")

    # 실험 종료 후 자동으로 Pickle 변환 실행
    print("\n🔄 실험 데이터 후처리(Pickle 변환) 시작...")

    # 1. CSV -> Pickle 변환 (data_processor 담당)
    process_model_results(save_dir, mode=mode)

    # 2. 결과 검증 (utils 담당)
    pkl_path = os.path.join(save_dir, "model_data.pkl")
    inspect_pickle_data(pkl_path)

    print("\n✨ 모든 실험 및 데이터 검증 성공")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="gpt-4o, o1-preview, gemini-2.5-flash etc.")
    parser.add_argument("--condition", type=str, default="vanilla", choices=["vanilla", "reasoning", "oneshot"], help="Experiment condition")
    parser.add_argument("--mode", type=str, default="normal", choices=["normal", "everystep"], help="Experiment option")
    parser.add_argument("--subjects", type=int, default=16, help="Number of virtual subjects")
    
    args = parser.parse_args()
    
    run_experiment(args.model, args.condition, args.mode, args.subjects)