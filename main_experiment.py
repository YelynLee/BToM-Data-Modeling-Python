import os
import sys
import time
import argparse
import pandas as pd
from tqdm import tqdm

# 현재 파일(main_experiment.py)이 있는 경로를 기준으로 src 폴더의 절대 경로를 만듦
src_path = os.path.join(os.path.dirname(__file__), 'src')

# 파이썬이 모듈을 찾을 때 src 폴더도 뒤져보도록 경로 추가
if src_path not in sys.path:
    sys.path.append(src_path)

from src.dataset import df_btom
from src.prompts import generate_scenario_prompt
from src.api_client import call_model_api
from src.utils import process_result_json, inspect_pickle_data
from src.config import BASE_RESULTS_DIR
from src.data_processor import process_model_results

def run_experiment(model_name, condition, mode, num_subjects=16, effort=None, version=""):
    # effort 값이 있을 경우 로그에 표시
    effort_log = f", Effort=[{effort}]" if effort else ""

    # 💡 [추가 1] condition과 version을 결합한 새로운 디렉토리 이름 생성 (예: vanilla + 2 = vanilla2)
    condition_folder = f"{condition}{version}"

    print(f"🚀 실험 시작: Model=[{model_name}], Condition=[{condition_folder}], Mode=[{mode}], Subjects=[{num_subjects}]{effort_log}")
    
    # 저장 경로 자동 생성: results/{model_name}/{condition}/
    # Everystep: results/gpt-4o/reasoning/everystep/
    if mode in ["everystep", "reverse"]:
        base_dir = os.path.join(BASE_RESULTS_DIR, model_name, condition_folder, mode)
    else:
        base_dir = os.path.join(BASE_RESULTS_DIR, model_name, condition_folder)

    # effort 값이 지정되었다면 하위 폴더(예: effort_low)를 추가
    if effort:
        save_dir = os.path.join(base_dir, f"effort_{effort}")
    else:
        save_dir = base_dir

    os.makedirs(save_dir, exist_ok=True)
    print(f"📂 결과 저장 경로: {save_dir}")

    # =========================================================================
    # 🌟 [NEW] Mode에 따른 데이터소스 분기 (Reverse 모드 지원)
    # =========================================================================
    if mode == "reverse":
        reverse_csv_path = os.path.join(BASE_RESULTS_DIR, "btom", "btom_reverse_inference_experiment.csv")
        if not os.path.exists(reverse_csv_path):
            print(f"❌ Error: 역방향 추론 데이터가 없습니다. 먼저 '--ref btom' 인자로 데이터를 추출하세요.\n경로: {reverse_csv_path}")
            return
        df_target = pd.read_csv(reverse_csv_path)
        scenario_groups = list(df_target.groupby('scenario_id'))
    else:
        scenario_groups = list(df_btom.groupby('scenario_id'))

    # 💡 [NEW] 시나리오별 예상 Time step 개수를 딕셔너리로 저장
    expected_counts = {sc_id: len(gdf) for sc_id, gdf in scenario_groups}

    for subject_idx in range(1, num_subjects + 1):

        # [체크포인트] 현재 피험자의 최종 저장될 파일명 미리 정의
        filename = os.path.join(save_dir, f"subject_{subject_idx:02d}.csv")
        absolute_path = os.path.abspath(filename)

        # 💡 [수정된 부분] 파일이 존재할 경우 기존 데이터를 읽어와서 완벽한 시나리오만 추려냄
        existing_records = {} # 성공한 시나리오 데이터를 저장할 딕셔너리
        completed_sc_ids = set()

        # [체크포인트] 파일이 이미 존재하면 실험을 건너뜀 (Skip)
        if os.path.exists(filename):
            try:
                df_existing = pd.read_csv(filename)
                
                # 기존 데이터를 시나리오 단위로 묶어서 검사
                for sc_id, group in df_existing.groupby('scenario_id'):
                    is_valid = True
                    
                    # 조건 1: 에러가 기록되어 있으면 불완전
                    if 'error' in group.columns and group['error'].notna().any():
                        is_valid = False
                        
                    # 조건 2: (reverse 모드가 아닐 때) desire_K 등 핵심 값이 NaN이면 불완전한 것으로 간주
                    if mode != "reverse":
                        required_cols = ['desire_K', 'belief_L'] # 검사할 필수 컬럼 목록
                        for col in required_cols:
                            if col in group.columns and group[col].isna().any():
                                is_valid = False
                                break # 하나라도 발견되면 더 검사할 필요 없이 실패 처리

                    # 💡 [NEW] 조건 3: everystep 모드일 때, 저장된 행 개수가 원본 데이터의 행 개수(전체 timestep)와 다르면 불완전
                    if mode == "everystep" and len(group) != expected_counts.get(sc_id, 0):
                        is_valid = False
                    
                    # 유효한 시나리오라면 저장
                    if is_valid:
                        existing_records[sc_id] = group.to_dict('records')
                        completed_sc_ids.add(sc_id)
                
                # 모든 시나리오가 완벽하게 끝났다면 스킵
                if len(completed_sc_ids) == len(scenario_groups):
                    print(f"\n⏩ Subject {subject_idx:02d}/{num_subjects} 이미 완료됨. 건너뜁니다! ({filename})")
                    continue
                else:
                    missing_count = len(scenario_groups) - len(completed_sc_ids)
                    print(f"\n🔄 Subject {subject_idx:02d}/{num_subjects} 복구 시작: {missing_count}개 시나리오 누락 발견됨 ({len(completed_sc_ids)}/{len(scenario_groups)} 완료)")
                    
            except Exception as e:
                print(f"\n⚠️ 기존 파일 읽기 실패. 덮어쓰고 새로 시작합니다: {e}")

        print(f"\n=== Subject {subject_idx}/{num_subjects} 진행 중 ===")
        results = []
        
        for sc_id, group_df in tqdm(scenario_groups, desc=f"Subj {subject_idx}"):
            
            # 💡 이미 성공적으로 완료된 시나리오면 API 호출 생략하고 기존 데이터 바로 추가
            if sc_id in completed_sc_ids:
                results.extend(existing_records[sc_id])
                continue
            
            # 1. 메타데이터 추출
            row0 = group_df.iloc[0]
            # Reverse 모드에서는 좌표(K_x 등) 컬럼이 없을 수 있으므로 try-except나 get 사용 방어
            present_trucks = [t for t, k in [('K', 'K'), ('L', 'L'), ('M', 'M')] 
                              if group_df.get(f'{k}_x', pd.Series([0])).iloc[0] != 0 or 
                                 group_df.get(f'{k}_y', pd.Series([0])).iloc[0] != 0]
            meta = {
                'group_desc': group_df.get('group_desc', pd.Series(["Unknown"])).iloc[0],
                'truck_presence': " and ".join(present_trucks) + " present" if present_trucks else "No trucks present"
            }
            
            # 2. 프롬프트 생성 (조건 반영)
            sys_prompt, user_prompt = generate_scenario_prompt(group_df, condition, mode)
            
            # 3. 모델 호출
            response_str = call_model_api(model_name, sys_prompt, user_prompt, effort=effort)
            
            # 4. 결과 처리
            if response_str:
                # 역방향 추론은 기존 process_result_json이 아니라 단순 JSON 로드 형태로 저장해야 할 수 있음.
                # (일단은 원본 문자열을 통째로 저장하거나 별도의 파싱 로직을 타도록 분기)
                if mode == "reverse":
                    import json
                    try:
                        # Markdown 코드 블록 제거 등 간단한 전처리
                        clean_str = response_str.replace("```json", "").replace("```", "").strip()
                        parsed = json.loads(clean_str)
                        res = {'scenario_id': sc_id, 'model': model_name, 'mode': mode}
                        res.update(parsed)
                        results.append(res)
                    except Exception as e:
                        print(f"\n⚠️ [Parsing Error] Scenario {sc_id}: {e}")
                        results.append({'scenario_id': sc_id, 'error': str(e), 'raw_response': response_str})
                else:
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
        
        # # 컬럼 순서 정렬 (보기 좋게)
        # cols = ['scenario_id', 'time_step', 'group_desc', 'truck_presence', 'model', 'condition', 'mode',
        #         'desire_reasoning', 'desire_K', 'desire_L', 'desire_M', 
        #         'belief_reasoning', 'belief_K', 'belief_L', 'belief_M', 'belief_Empty',
        #         'error', 'raw_response']
        
        # # 결과에 있는 컬럼만 필터링 (에러 시 일부 컬럼 없을 수 있음)
        # final_cols = [c for c in cols if c in df_res.columns]
        # df_res = df_res[final_cols]
        
        # 파일 저장 (체크포인트 검사용 파일 생성)
        df_res.to_csv(filename, index=False)
        print(f"✅ Subject {subject_idx} Saved.")


    print("\n✨ 모든 실험이 성공적으로 종료되었습니다!")

    # Reverse 모드는 별도의 전용 파서/검증이 필요하므로 Pickle 변환은 스킵
    if mode != "reverse":
        # 실험 종료 후 자동으로 Pickle 변환 실행
        print("\n🔄 실험 데이터 후처리(Pickle 변환) 시작...")

        # 1. CSV -> Pickle 변환 (data_processor 담당)
        process_model_results(save_dir, mode=mode)

        # 2. 결과 검증 (utils 담당)
        pkl_path = os.path.join(save_dir, "model_data.pkl")
        inspect_pickle_data(pkl_path)

        print("\n✨ 모든 실험 및 데이터 검증 성공")
    else:
        print("\n✨ Reverse 실험의 CSV 파일 생성이 완료되었습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="gpt-4o, gemini-2.5-flash etc.")
    parser.add_argument("--condition", type=str, default="vanilla", choices=["vanilla", "reasoning", "oneshot"], help="Experiment condition")
    parser.add_argument("--mode", type=str, default="normal", choices=["normal", "everystep", "reverse", "control"], help="Experiment option")
    parser.add_argument("--subjects", type=int, default=16, help="Number of virtual subjects")
    parser.add_argument("--effort", type=str, default=None, choices=["none", "low", "medium", "high"], help="Reasoning effort level (only for supported models)")
    parser.add_argument("--version", type=str, default="", help="Condition 폴더명 뒤에 붙일 버전 (예: 2 -> vanilla2)")

    args = parser.parse_args()
    
    run_experiment(args.model, args.condition, args.mode, args.subjects, args.effort, args.version)