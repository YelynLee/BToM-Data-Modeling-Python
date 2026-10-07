import os

# =========================================================================
# 저장 경로 설정
# =========================================================================
# Transformer 실험 결과가 저장될 최상위 폴더
BASE_RESULTS_DIR = "results"

# Human Data 원본 경로
HUMAN_MAT_PATH = "data/human_data.mat"
HUMAN_PKL_PATH = "data/human/human_data.pkl"

# Reference Models Data 원본 경로
REFERENCE_MAT_PATH = "C:/Users/user/Documents/MATLAB/BToM_paper/data"
REFERENCE_PKL_DIR = "data"

# Stimuli Data 원본 경로
STIMULI_MAT_PATH = "data/stimuli.mat"

# =========================================================================
# 실험 Mode / 결과 폴더 규칙
# =========================================================================
# 시점별(stepwise) 응답을 내는 mode. 행 단위가 (scenario_id, time_step)이며
# phase labeling, valid_only 구축 등 everystep 파이프라인을 그대로 공유함.
#   - everystep : 전체 궤적을 한 번에 주고 매 스텝 응답을 받음 (기존)
#   - prefixstep: t마다 독립 호출, 로그를 1..t로 잘라서 End-step 질문을 그대로 던짐
STEPWISE_MODES = ("everystep", "prefixstep")

# 결과를 하위 폴더에 따로 저장하는 mode (normal은 condition 폴더 바로 아래)
SUBFOLDER_MODES = ("everystep", "prefixstep", "reverse", "control")


def mode_folder(mode, current_belief=False, mask_hidden=False):
    """
    실험 변형까지 반영한 하위 폴더 이름을 반환함.
      everystep                      -> 'everystep'
      prefixstep + current_belief    -> 'prefixstep_cur'
      prefixstep + mask_hidden + cur -> 'prefixstep_mask_cur'
      normal                         -> ''  (condition 폴더 바로 아래)
    """
    if mode not in SUBFOLDER_MODES:
        return ""
    name = mode
    if mask_hidden:
        name += "_mask"
    if current_belief:
        name += "_cur"
    return name


def base_mode(mode_dir):
    """'prefixstep_mask_cur' 같은 폴더 이름에서 기본 mode('prefixstep')를 꺼냄."""
    return mode_dir.split("_")[0] if mode_dir else "normal"


def result_dir(model_name, condition, mode_dir=""):
    """results/{model}/{condition}[/{mode_dir}]"""
    if mode_dir:
        return os.path.join(BASE_RESULTS_DIR, model_name, condition, mode_dir)
    return os.path.join(BASE_RESULTS_DIR, model_name, condition)


# Check 계열(에이전트가 G2를 확인하러 가는) 시나리오 그룹:
#   G1 Check-GoBack(Present), G2 Check-Stay(Present), G4 Check-GoBack(Absent),
#   G6 CheckPartial(Present), G7 CheckPartial(Absent)
CHECK_GROUP_IDS = (1, 2, 4, 6, 7)


def get_scenario_subset(subset="all", include_irrational=False):
    """
    API를 돌릴 scenario_id 집합을 반환함.
      'all'  : 78개 전체 (기존 동작, irrational 포함)
      'check': Check-GoBack / Check-Stay / Check-Partial만. 분석에서 어차피 제외되는
               irrational 경로(11, 12, 22, 71, 72)는 기본적으로 빼서 비용을 줄임.
    """
    if subset == "all":
        return None  # 필터 없음
    if subset == "check":
        groups = get_group_indices(include_irrational=include_irrational)
        return sorted(sid for gid in CHECK_GROUP_IDS for sid in groups[gid - 1])
    raise ValueError(f"Unknown scenario subset: {subset}")


# =========================================================================
# 행동 그룹 정의 (Labeling용)
# =========================================================================
BEHAVIOR_GROUPS = {
    1: "Check-GoBack(Present)", 2: "Check-Stay(Present)", 3: "NoCheck(Present)",
    4: "Check-GoBack(Absent)",  5: "NoCheck(Absent)",     6: "CheckPartial(Present)",
    7: "CheckPartial(Absent)"
}

def get_group_indices(include_irrational=True):
    """
    BToM 실험의 7가지 행동 조건에 해당하는 Scenario ID 리스트를 반환함.
    
    Args:
        include_irrational (bool): 비합리적 시나리오(11, 12, 22, 71, 72)를 포함할지 여부.
                                   - Dataset 생성 시: True 권장 (데이터 보존)
                                   - 논문 재현 분석 시: False 권장 (이상치 제거)
    """
    if include_irrational:
        # 비합리적 시나리오 포함 (전체 78개)
        group_inds = [
            [1, 6, 11, 3, 8, 13, 40, 43, 46, 25, 28, 31],     # G1
            [2, 7, 12, 4, 9, 14, 41, 44, 47, 26, 29, 32],     # G2
            [5, 10, 15, 42, 45, 48, 27, 30, 33],              # G3
            [16, 19, 22, 17, 20, 23, 49, 51, 53, 34, 36, 38], # G4
            [18, 21, 24, 50, 52, 54, 35, 37, 39],             # G5
            [55, 63, 71, 59, 67, 75, 57, 65, 73, 61, 69, 77], # G6
            [56, 64, 72, 60, 68, 76, 58, 66, 74, 62, 70, 78]  # G7
        ]
    else:
        # 비합리적 시나리오 제외 (총 73개 - 논문 분석용)
        group_inds = [
            [1, 6, 3, 8, 13, 40, 43, 46, 25, 28, 31],
            [2, 7, 4, 9, 14, 41, 44, 47, 26, 29, 32],
            [5, 10, 15, 42, 45, 48, 27, 30, 33],
            [16, 19, 17, 20, 23, 49, 51, 53, 34, 36, 38],
            [18, 21, 24, 50, 52, 54, 35, 37, 39],
            [55, 63, 59, 67, 75, 57, 65, 73, 61, 69, 77],
            [56, 64, 60, 68, 76, 58, 66, 74, 62, 70, 78]
        ]
    return group_inds