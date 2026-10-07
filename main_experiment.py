import os
import sys
import json
import time
import random
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
from src.config import (BASE_RESULTS_DIR, STEPWISE_MODES,
                        mode_folder, get_scenario_subset)
from src.data_processor import process_model_results

# 응답 JSON을 그대로 저장하는 mode (desire/belief 평정이 아닌 응답)
RAW_JSON_MODES = ("reverse", "control")


# =========================================================================
# 작업 단위(Work unit) 구성
# =========================================================================
def build_work_units(scenario_groups, mode):
    """
    API 호출 1회에 해당하는 작업 단위 목록을 만듦.
      - prefixstep: (scenario_id, t) 마다 1회. 로그는 1..t로 잘림.
      - 그 외      : scenario 마다 1회. 전체 로그 사용.
    Returns: list of (key, sc_id, t_cut, df_input)
        key = (sc_id, t_cut)  (t_cut은 prefixstep이 아니면 None)
    """
    units = []
    for sc_id, gdf in scenario_groups:
        if mode == "prefixstep":
            gdf = gdf.sort_values('time_step')
            for t_cut in gdf['time_step'].astype(int).tolist():
                units.append(((sc_id, t_cut), sc_id, t_cut, gdf[gdf['time_step'] <= t_cut]))
        else:
            units.append(((sc_id, None), sc_id, None, gdf))
    return units


def make_meta(group_df):
    """시나리오 메타데이터 (group_desc, truck_presence)."""
    # Reverse 모드에서는 좌표(K_x 등) 컬럼이 없을 수 있으므로 get 사용 방어
    present_trucks = [t for t in ['K', 'L', 'M']
                      if group_df.get(f'{t}_x', pd.Series([0])).iloc[0] != 0 or
                         group_df.get(f'{t}_y', pd.Series([0])).iloc[0] != 0]
    return {
        'group_desc': group_df.get('group_desc', pd.Series(["Unknown"])).iloc[0],
        'truck_presence': " and ".join(present_trucks) + " present" if present_trucks else "No trucks present"
    }


def resolve_belief_order(belief_order, subject_idx, sc_id, t_cut, seed=0):
    """
    두 belief 질문의 제시 순서를 정함.
    'random'이면 (seed, subject, scenario, t)로 시드를 고정한 난수로 호출마다 정하므로
    같은 명령을 다시 실행하거나 중간에 재개해도 같은 호출은 같은 순서를 받음.
    """
    if belief_order != "random":
        return belief_order
    rng = random.Random(f"{seed}-{subject_idx}-{sc_id}-{t_cut}")
    return rng.choice(["initial_first", "now_first"])


# =========================================================================
# 체크포인트: 기존 CSV에서 '완료된 작업 단위'만 복원
# =========================================================================
def required_columns(mode, current_belief):
    if mode in RAW_JSON_MODES:
        return []
    cols = ['desire_K', 'belief_L']
    if current_belief:
        cols.append('belief_now_L')
    return cols


def load_completed_units(filename, mode, current_belief, expected_counts):
    """
    Returns: dict {key: [row dicts]} — 유효한 작업 단위만.
    """
    completed = {}
    if not os.path.exists(filename):
        return completed

    df_existing = pd.read_csv(filename)
    req = required_columns(mode, current_belief)

    def rows_ok(rows):
        if 'error' in rows.columns and rows['error'].notna().any():
            return False
        for col in req:
            if col not in rows.columns or rows[col].isna().any():
                return False
        return True

    if mode == "prefixstep":
        # (scenario_id, time_step) 단위로 개별 복원 -> 시나리오 중간에서 끊겨도 이어서 진행
        for (sc_id, t), rows in df_existing.groupby(['scenario_id', 'time_step']):
            if len(rows) == 1 and rows_ok(rows):
                completed[(int(sc_id), int(t))] = rows.to_dict('records')
    else:
        for sc_id, rows in df_existing.groupby('scenario_id'):
            if not rows_ok(rows):
                continue
            # everystep: 저장된 행 개수가 원본 time step 개수와 같아야 완전함
            if mode == "everystep" and len(rows) != expected_counts.get(sc_id, 0):
                continue
            completed[(int(sc_id), None)] = rows.to_dict('records')
    return completed


def save_rows(results, filename):
    """작업 단위별 결과를 하나의 CSV로 저장 (scenario_id, time_step 순 정렬)."""
    rows = [r for key in sorted(results, key=lambda k: (k[0], k[1] or 0)) for r in results[key]]
    if rows:
        pd.DataFrame(rows).to_csv(filename, index=False)


# =========================================================================
# 응답 파싱
# =========================================================================
def parse_response(response_str, sc_id, t_cut, meta, model_name, condition, mode):
    """Returns: list of row dicts"""
    if mode in RAW_JSON_MODES:
        # reverse / control: 평정 컬럼이 아닌 자유 형식 JSON -> 그대로 펼쳐서 저장
        raw = response_str.get("text", "") if isinstance(response_str, dict) else response_str
        try:
            clean_str = raw.replace("```json", "").replace("```", "").strip()
            start, end = clean_str.find('{'), clean_str.rfind('}') + 1
            parsed = json.loads(clean_str[start:end] if start != -1 else clean_str)
            res = {'scenario_id': sc_id, 'model': model_name, 'condition': condition, 'mode': mode,
                   'response_json': json.dumps(parsed)}
            res.update({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in parsed.items()})
            return [res]
        except Exception as e:
            return [{'scenario_id': sc_id, 'model': model_name, 'mode': mode,
                     'error': str(e), 'raw_response': raw}]

    # prefixstep은 normal과 같은 단일 객체 형식으로 파싱
    res = process_result_json(sc_id, meta, response_str, model_name, condition, mode)
    rows = res if isinstance(res, list) else [res]
    if mode == "prefixstep":
        for r in rows:
            r['time_step'] = t_cut  # 로그를 자른 길이 = 이 응답이 대응하는 시점
    return rows


# =========================================================================
# 메인 실험 루프
# =========================================================================
def run_experiment(model_name, condition, mode, num_subjects=16, effort=None, version="",
                   scenarios="all", scenario_ids=None, current_belief=False,
                   belief_order="random", mask_hidden=False, preview=False,
                   legacy_wording=False, order_seed=0):
    # effort 값이 있을 경우 로그에 표시
    effort_log = f", Effort=[{effort}]" if effort else ""

    # 💡 condition과 version을 결합한 새로운 디렉토리 이름 생성 (예: vanilla + 2 = vanilla2)
    condition_folder = f"{condition}{version}"

    # belief_order는 호출마다 정해지므로 여기서는 빼고, 호출 직전에 넣음
    prompt_kwargs = dict(current_belief=current_belief, mask_hidden=mask_hidden, legacy_wording=legacy_wording)

    print(f"🚀 실험 시작: Model=[{model_name}], Condition=[{condition_folder}], Mode=[{mode}], "
          f"Subjects=[{num_subjects}]{effort_log}")
    if current_belief or mask_hidden:
        print(f"   -> current_belief={current_belief} (order={belief_order}), mask_hidden={mask_hidden}")
    if mode in STEPWISE_MODES:
        print(f"   -> belief wording: {'legacy (기존 문구)' if legacy_wording else 'only given the information up to step t'}")

    # 저장 경로: results/{model}/{condition}[/{mode_folder}][/effort_x]
    # 예) results/gpt-4o/vanilla/prefixstep_cur/
    sub = mode_folder(mode, current_belief, mask_hidden, legacy_wording)
    base_dir = os.path.join(BASE_RESULTS_DIR, model_name, condition_folder, sub) if sub \
        else os.path.join(BASE_RESULTS_DIR, model_name, condition_folder)
    save_dir = os.path.join(base_dir, f"effort_{effort}") if effort else base_dir

    # =========================================================================
    # Mode에 따른 데이터소스 분기 (Reverse 모드 지원)
    # =========================================================================
    if mode == "reverse":
        reverse_csv_path = os.path.join(BASE_RESULTS_DIR, "btom", "btom_reverse_inference_experiment.csv")
        if not os.path.exists(reverse_csv_path):
            print(f"❌ Error: 역방향 추론 데이터가 없습니다. 먼저 '--ref btom' 인자로 데이터를 추출하세요.\n경로: {reverse_csv_path}")
            return
        df_target = pd.read_csv(reverse_csv_path)
    else:
        df_target = df_btom

    # 시나리오 필터 (--scenarios check / --scenario_ids)
    keep_ids = scenario_ids if scenario_ids else get_scenario_subset(scenarios)
    if keep_ids is not None:
        df_target = df_target[df_target['scenario_id'].isin(keep_ids)]
    scenario_groups = list(df_target.groupby('scenario_id'))
    expected_counts = {sc_id: len(gdf) for sc_id, gdf in scenario_groups}

    units = build_work_units(scenario_groups, mode)
    print(f"🧮 시나리오 {len(scenario_groups)}개, 피험자당 API 호출 {len(units)}회 "
          f"(총 최대 {len(units) * num_subjects}회)")

    # -------------------------------------------------------------------------
    # 🔎 Preview: API 호출 없이 프롬프트만 확인
    # -------------------------------------------------------------------------
    if preview:
        if mode == "prefixstep":
            sc0 = units[0][1]
            sc_units = [u for u in units if u[1] == sc0]
            show = [sc_units[0]] + ([sc_units[-1]] if len(sc_units) > 1 else [])
        else:
            show = units[:1]
        for key, sc_id, t_cut, df_input in show:
            order = resolve_belief_order(belief_order, 1, sc_id, t_cut, order_seed)
            sys_prompt, user_prompt = generate_scenario_prompt(df_input, condition, mode,
                                                               belief_order=order, **prompt_kwargs)
            print("=" * 70)
            print(f"[Preview] scenario {sc_id}" + (f", t = {t_cut}" if t_cut else "")
                  + (f", belief order = {order}" if current_belief else "")
                  + f"  -> {save_dir}")
            print("-" * 70 + "\n[System]\n" + sys_prompt.strip())
            print("-" * 70 + "\n[User]\n" + user_prompt.strip())
        print("=" * 70 + "\n(preview 모드: API를 호출하지 않았습니다)")
        return

    os.makedirs(save_dir, exist_ok=True)
    print(f"📂 결과 저장 경로: {save_dir}")

    for subject_idx in range(1, num_subjects + 1):
        filename = os.path.join(save_dir, f"subject_{subject_idx:02d}.csv")

        # [체크포인트] 기존 파일에서 완료된 작업 단위 복원
        try:
            results = load_completed_units(filename, mode, current_belief, expected_counts)
        except Exception as e:
            print(f"\n⚠️ 기존 파일 읽기 실패. 덮어쓰고 새로 시작합니다: {e}")
            results = {}

        todo = [u for u in units if u[0] not in results]
        if not todo:
            print(f"\n⏩ Subject {subject_idx:02d}/{num_subjects} 이미 완료됨. 건너뜁니다! ({filename})")
            continue
        if results:
            print(f"\n🔄 Subject {subject_idx:02d}/{num_subjects} 복구 시작: "
                  f"{len(todo)}개 작업 단위 남음 ({len(results)}/{len(units)} 완료)")

        print(f"\n=== Subject {subject_idx}/{num_subjects} 진행 중 ===")

        for key, sc_id, t_cut, df_input in tqdm(todo, desc=f"Subj {subject_idx}"):
            meta = make_meta(df_input)

            # 프롬프트 생성 (prefixstep이면 df_input이 1..t로 잘려 있음)
            order = resolve_belief_order(belief_order, subject_idx, sc_id, t_cut, order_seed)
            sys_prompt, user_prompt = generate_scenario_prompt(df_input, condition, mode,
                                                               belief_order=order, **prompt_kwargs)

            # 모델 호출
            response_str = call_model_api(model_name, sys_prompt, user_prompt, effort=effort)

            if response_str:
                rows = parse_response(response_str, sc_id, t_cut, meta, model_name, condition, mode)
                if 'error' in rows[0]:
                    tag = f"Scenario {sc_id}" + (f", t={t_cut}" if t_cut else "")
                    print(f"\n⚠️ [Parsing Error] {tag}: {rows[0]['error']}")
            else:
                rows = [{'scenario_id': sc_id, 'time_step': t_cut, 'error': 'API Fail', 'model': model_name}]

            if current_belief:
                for r in rows:
                    r['belief_order'] = order  # 분석에서 순서 효과를 확인할 수 있도록 기록

            results[key] = rows
            save_rows(results, filename)  # 작업 단위마다 저장 -> 중간에 끊겨도 이어서 진행 가능

            # Rate Limit 방지 (o1은 더 길게)
            time.sleep(2 if "o1" in model_name else 0.5)

        print(f"✅ Subject {subject_idx} Saved.")

    print("\n✨ 모든 실험이 성공적으로 종료되었습니다!")

    if mode in RAW_JSON_MODES:
        print(f"\n✨ {mode} 실험의 CSV 파일 생성이 완료되었습니다. (Pickle 변환은 건너뜀)")
        return

    # 실험 종료 후 자동으로 Pickle 변환 실행
    print("\n🔄 실험 데이터 후처리(Pickle 변환) 시작...")
    process_model_results(save_dir, mode=mode)
    inspect_pickle_data(os.path.join(save_dir, "model_data.pkl"))
    print("\n✨ 모든 실험 및 데이터 검증 성공")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="gpt-4o, gemini-2.5-flash etc.")
    parser.add_argument("--condition", type=str, default="vanilla", choices=["vanilla", "reasoning", "oneshot"], help="Experiment condition")
    parser.add_argument("--mode", type=str, default="normal",
                        choices=["normal", "everystep", "prefixstep", "reverse", "control"],
                        help="normal(End-step) / everystep / prefixstep(t마다 로그를 잘라 End-step 질문) / reverse / control")
    parser.add_argument("--subjects", type=int, default=16, help="Number of virtual subjects")
    parser.add_argument("--effort", type=str, default=None, choices=["none", "low", "medium", "high"], help="Reasoning effort level (only for supported models)")
    parser.add_argument("--version", type=str, default="", help="Condition 폴더명 뒤에 붙일 버전 (예: 2 -> vanilla2)")

    # 비용 제어
    parser.add_argument("--scenarios", type=str, default="all", choices=["all", "check"],
                        help="check: Check-GoBack/Check-Stay/Check-Partial만 (irrational 경로 제외)")
    parser.add_argument("--scenario_ids", type=str, default=None,
                        help="쉼표로 구분한 scenario_id 목록 (예: 1,6,40). 지정 시 --scenarios보다 우선")

    # 실험 변형
    parser.add_argument("--current_belief", action="store_true",
                        help="initial belief(t=1)와 함께 에이전트의 현재 belief를 따로 질문 (everystep / prefixstep)")
    parser.add_argument("--belief_order", type=str, default="random", choices=["random", "initial_first", "now_first"],
                        help="--current_belief일 때 두 belief 질문의 제시 순서. random(기본): 호출마다 무작위(시드 고정), CSV의 belief_order 컬럼에 기록")
    parser.add_argument("--order_seed", type=int, default=0, help="belief_order=random의 시드")
    parser.add_argument("--legacy_wording", action="store_true",
                        help="everystep / prefixstep belief 문구를 수정 이전 것으로 사용 (기존 Every-step 결과 재현; 결과 폴더 'everystep')")
    parser.add_argument("--mask_hidden", action="store_true",
                        help="Map Configuration에서 Spot 2의 트럭 정체를 숨김 (Spot 2가 보일 때만 로그로 드러남)")

    parser.add_argument("--preview", action="store_true", help="API 호출 없이 생성될 프롬프트만 출력")

    args = parser.parse_args()

    if args.current_belief and args.mode not in STEPWISE_MODES:
        parser.error("--current_belief는 --mode everystep 또는 prefixstep에서만 사용할 수 있습니다.")
    if args.mask_hidden and args.mode not in STEPWISE_MODES + ("normal",):
        parser.error("--mask_hidden은 normal / everystep / prefixstep에서만 사용할 수 있습니다.")
    if args.mask_hidden and args.mode == "normal":
        # normal은 하위 폴더가 없어 End-step 결과를 덮어쓸 수 있으므로 막아둠
        parser.error("--mask_hidden + normal은 결과 폴더가 End-step과 겹칩니다. prefixstep의 마지막 스텝을 사용하세요.")

    ids = [int(x) for x in args.scenario_ids.split(",")] if args.scenario_ids else None

    run_experiment(args.model, args.condition, args.mode, args.subjects, args.effort, args.version,
                   scenarios=args.scenarios, scenario_ids=ids, current_belief=args.current_belief,
                   belief_order=args.belief_order, mask_hidden=args.mask_hidden, preview=args.preview,
                   legacy_wording=args.legacy_wording, order_seed=args.order_seed)
