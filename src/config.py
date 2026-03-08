# =========================================================================
# 저장 경로 설정
# =========================================================================
# Transformer 실험 결과가 저장될 최상위 폴더
BASE_RESULTS_DIR = "results"

# Human Data 원본 경로
HUMAN_MAT_PATH = "C:/Users/user/Desktop/BToM-master/BToM-master/BeliefDesireInference/data/human_data.mat"
HUMAN_PKL_PATH = "data/human/human_data.pkl"

# Reference Models Data 원본 경로
REFERENCE_MAT_PATH = "C:/Users/user/Desktop/BToM-master/BToM-master/BeliefDesireInference/data"
REFERENCE_PKL_DIR = "data"

# Stimuli Data 원본 경로
STIMULI_MAT_PATH = "C:/Users/user/Desktop/BToM-master/BToM-master/BeliefDesireInference/data/stimuli.mat"

# BToM Everystep Data 원본 경로
BTOM_EVERY_MAT_PATH = "C:/Users/user/Desktop/project/BToM 코드/data/btom/btom_everystep_beta2.5.mat"

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