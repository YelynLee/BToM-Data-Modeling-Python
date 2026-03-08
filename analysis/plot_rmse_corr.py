import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import os
from src.config import HUMAN_PKL_PATH, get_group_indices

# 분석할 그룹 순서 (Bar Plot과 동일)
GROUP_NAMES_ORDERED = [
    "No Check(P)", 
    "Check-Partial(P)", 
    "Check-GoBack(P)", 
    "Check-Stay(P)", 
    "No Check(A)", 
    "Check-Partial(A)", 
    "Check-GoBack(A)"
]
ORDER_INDICES = [2, 5, 0, 1, 4, 6, 3] # 위 이름에 해당하는 get_group_indices 인덱스

def calc_metrics(h_data, m_data):
    """RMSE와 Correlation 계산"""
    # 1D로 펼치기 (그룹 내 모든 데이터 포인트 비교)
    h_flat = h_data.flatten()
    m_flat = m_data.flatten()
    
    # NaN 제거
    mask = ~np.isnan(h_flat) & ~np.isnan(m_flat)
    if np.sum(mask) < 2:
        return 0.0, 0.0 # 데이터 부족
        
    h_clean = h_flat[mask]
    m_clean = m_flat[mask]
    
    # RMSE (낮을수록 좋음)
    rmse = np.sqrt(mean_squared_error(h_clean, m_clean))
    
    # Correlation (높을수록 좋음)
    # 데이터가 상수(분산=0)일 경우 에러 방지
    if np.std(h_clean) == 0 or np.std(m_clean) == 0:
        r = 0.0 
    else:
        r, _ = pearsonr(h_clean, m_clean)
        
    return rmse, r

def plot_combined_metrics(model_pkl_path, output_img_path, model_name="Model"):
    """
    Desire(위)와 Belief(아래)의 성능을 하나의 Figure에 통합하여 시각화
    """
    with open(HUMAN_PKL_PATH, 'rb') as f: human = pickle.load(f)
    with open(model_pkl_path, 'rb') as f: model = pickle.load(f)

    # 분석 대상 정의 (Title, Key, Threshold)
    targets = [
        ("Desire Inference (1-7 Scale)", 'des_inf_mean', 1.5),
        ("Belief Inference (0-1 Probability)", 'bel_inf_mean_norm', 0.25)
    ]
    
    groups_raw = get_group_indices(include_irrational=False)
    
    x = np.arange(len(GROUP_NAMES_ORDERED))
    width = 0.6

    # ---------------------------------------------------------
    # Canvas 설정: 2행(Desire/Belief) x 1열, 사이즈 키움
    # ---------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    plt.subplots_adjust(hspace=0.4) # 위아래 간격

    for row_idx, (title, key, threshold) in enumerate(targets):
        ax1 = axes[row_idx] # 현재 그릴 메인 축 (RMSE용)
        
        # 데이터 추출
        h_full = human[key]
        m_full = model[key]
        
        rmses, corrs, labels = [], [], []
        
        for i, group_idx in enumerate(ORDER_INDICES):
            g_scenarios = groups_raw[group_idx]
            g_idx_0 = [idx - 1 for idx in g_scenarios]
            h_sub = h_full[:, g_idx_0]
            m_sub = m_full[:, g_idx_0]
            rmse, r = calc_metrics(h_sub, m_sub)
            rmses.append(rmse)
            corrs.append(r)
            labels.append(GROUP_NAMES_ORDERED[i])

        # --- Dual Axis Plotting ---
        # 1. RMSE Bar (Left Axis)
        ax1.bar(x - width/2, rmses, width/2, label='RMSE (Error)', color='salmon', alpha=0.9, edgecolor='brown')
        ax1.axhline(threshold, color='red', linestyle=':', linewidth=2, label=f'Threshold ({threshold})')
        
        ax1.set_ylabel('RMSE (Lower is Better)', fontweight='bold', color='brown')
        ax1.tick_params(axis='y', labelcolor='brown')
        ax1.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax1.grid(axis='y', linestyle='--', alpha=0.3)
        
        # RMSE 값 표시
        for i, v in enumerate(rmses):
            ax1.text(i - width/2, v + 0.01, f"{v:.2f}", ha='center', va='bottom', fontsize=8, color='brown', fontweight='bold')

        # 2. Correlation Bar (Right Axis)
        ax2 = ax1.twinx()
        ax2.bar(x + width/2, corrs, width/2, label='Correlation', color='royalblue', alpha=0.9, edgecolor='navy')
        ax2.axhline(0, color='black', linewidth=1)
        
        ax2.set_ylabel('Correlation (Higher is Better)', fontweight='bold', color='navy')
        ax2.tick_params(axis='y', labelcolor='navy')
        ax2.set_ylim(-1.1, 1.1)

        # Correlation 값 표시
        for i, v in enumerate(corrs):
            va = 'bottom' if v >= 0 else 'top'
            offset = 0.05 if v >= 0 else -0.05
            ax2.text(i + width/2, v + offset, f"{v:.2f}", ha='center', va=va, fontsize=8, color='navy', fontweight='bold')

        # X축 설정
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=15, ha='right', fontsize=10, fontweight='bold')
        
        # 범례 통합 (첫 번째 로우에만 표시하거나 각각 표시)
        lines1, lab1 = ax1.get_legend_handles_labels()
        lines2, lab2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, lab1 + lab2, loc='upper left', fontsize=9)

    plt.suptitle(f"[{model_name}] Comprehensive Performance Analysis", fontsize=18, y=0.96, fontweight='bold')
    
    os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_img_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Combined Performance Plot saved to: {output_img_path}")