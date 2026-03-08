import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from src.config import HUMAN_PKL_PATH

# 설정 (상수)
# 각 행동 그룹(G1~G7)을 대표하는 시나리오들.
CONDITIONS = [48, 73, 46, 47, 54, 74, 53]

# X축 라벨 설정 (User Confirmation)
# N (Empty)
DESIRE_LABELS = ['K', 'L', 'M']
BELIEF_LABELS = ['L', 'M', 'N']

# 행동 그룹 이름 (논문 참조, 그래프 제목용)
GROUP_NAMES = [
    "No Check (Present)",       # 48 (G3)
    "Check-Partial (Present)",  # 73 (G6)
    "Check-GoBack (Present)",   # 46 (G1)
    "Check-Stay (Present)",     # 47 (G2)
    "No Check (Absent)",        # 54 (G5)
    "Check-Partial (Absent)",   # 74 (G7)
    "Check-GoBack (Absent)"     # 53 (G4)
]

def plot_comparison_bars(model_pkl_path, output_img_path, model_name="Model"):
    """
    Args:
        model_pkl_path: 모델 데이터 Pickle 경로 (입력)
        output_img_path: 저장할 이미지 경로 (출력)
        model_name: 그래프에 표시할 모델 이름 (예: GPT-4o)
    """
    # 1. 데이터 로드
    if not os.path.exists(HUMAN_PKL_PATH) or not os.path.exists(model_pkl_path):
        print("❌ Data files not found.")
        return

    with open(HUMAN_PKL_PATH, 'rb') as f:
        human = pickle.load(f)
    with open(model_pkl_path, 'rb') as f:
        model = pickle.load(f)

    # 2. Plot 초기화 (2행 7열)
    # Row 1: Desire, Row 2: Belief
    fig, axs = plt.subplots(2, 7, figsize=(20, 7))
    
    # 바 차트 설정
    bar_width = 0.3
    x = np.arange(3) # 0, 1, 2 (항목 3개)
    
    # 3. 루프: 7개 조건(Condition)에 대해 그리기
    for i, cond_id in enumerate(CONDITIONS):
        # Python 0-based index 변환
        idx = cond_id - 1
        
        # ---------------------------------------------------------
        # (1) Desire Plot (Row 0)
        # ---------------------------------------------------------
        ax_d = axs[0, i]
        
        # 데이터 추출 (MATLAB: data - 1 하여 0~6 스케일로 변환)
        # shape: (3, 78) -> (3,)
        h_des_mean = human['des_inf_mean'][:, idx] - 1
        h_des_se   = human['des_inf_se'][:, idx]
        m_des_mean = model['des_inf_mean'][:, idx] - 1
        
        # Bar Plotting
        # Model (Left, Blue)
        ax_d.bar(x - bar_width/2, m_des_mean, bar_width, label='Model', color='royalblue', alpha=0.8)
        
        # Human (Right, Gray with ErrorBar)
        ax_d.bar(x + bar_width/2, h_des_mean, bar_width, label='Human', color='lightgray', edgecolor='gray')
        ax_d.errorbar(x + bar_width/2, h_des_mean, yerr=h_des_se, fmt='none', ecolor='black', capsize=3)
        
        # Axis Settings
        ax_d.set_ylim(0, 6)
        ax_d.set_xticks(x)
        ax_d.set_xticklabels(DESIRE_LABELS, fontsize='x-large')
        
        # Y축 라벨: 0,1,2,3,4,5,6 -> 실제 의미 1,2,3,4,5,6,7
        ax_d.set_yticks(range(7))
        ax_d.set_yticklabels([str(k) for k in range(1, 8)])
        
        # 첫 번째 열에만 Y축 제목 표시
        if i == 0:
            ax_d.set_ylabel('Desire Rating (1-7)', fontsize='large', fontweight='bold')
            # 범례는 첫 번째 그래프에만 표시
            ax_d.legend(loc='upper right', fontsize='small')
        
        # 제목 (시나리오 번호)
        # [핵심] 현재 순서에 맞는 그룹 이름 가져오기
        group_title = GROUP_NAMES[i]
        ax_d.set_title(f"{group_title}", fontsize=14, fontweight='bold')

        # ---------------------------------------------------------
        # (2) Belief Plot (Row 1)
        # ---------------------------------------------------------
        ax_b = axs[1, i]
        
        # 데이터 추출 (Probability 0~1)
        h_bel_mean = human['bel_inf_mean_norm'][:, idx]
        h_bel_se   = human['bel_inf_se'][:, idx]
        m_bel_mean = model['bel_inf_mean_norm'][:, idx]
        
        # Bar Plotting
        ax_b.bar(x - bar_width/2, m_bel_mean, bar_width, label='Model', color='royalblue', alpha=0.8)
        
        # Human with ErrorBar
        ax_b.bar(x + bar_width/2, h_bel_mean, bar_width, label='Human', color='lightgray', edgecolor='gray')
        ax_b.errorbar(x + bar_width/2, h_bel_mean, yerr=h_bel_se, fmt='none', ecolor='black', capsize=3)
        
        # Axis Settings
        ax_b.set_ylim(0, 1.05) # 확률이므로 0~1
        ax_b.set_xticks(x)
        ax_b.set_xticklabels(BELIEF_LABELS, fontsize='x-large') # L, M, N
        
        if i == 0:
            ax_b.set_ylabel('Belief Probability', fontsize='large', fontweight='bold')
            # 범례는 첫 번째 그래프에만 표시
            ax_b.legend(loc='upper right', fontsize='medium')
            
        # ax_b.set_title(f"{group_title}", fontsize=10, fontweight='bold')

    # 4. 저장 및 출력
    # 폴더 생성
    os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
    
    # tight_layout을 쓰되, 위쪽 5% 공간(Top=0.95)은 제목을 위해 비워둠
    plt.suptitle(f"{model_name} vs Human: Example Scenario Analysis", fontsize=28, y=0.98, fontweight='bold')
    # rect=[left, bottom, right, top]
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(output_img_path, bbox_inches='tight', pad_inches=0.5, dpi=150)

    plt.show()
    
    print(f"✅ Comparison Bar Plot saved to: {output_img_path}")