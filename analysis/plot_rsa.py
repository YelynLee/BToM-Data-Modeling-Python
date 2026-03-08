import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import os
from src.config import get_group_indices

# 그룹 이름 (히트맵 축 라벨용) -> bar plot과 동일하게 정렬
GROUP_NAMES = [
    "No Check(P)", 
    "Check-Partial(P)", 
    "Check-GoBack(P)", 
    "Check-Stay(P)", 
    "No Check(A)", 
    "Check-Partial(A)", 
    "Check-GoBack(A)"
]

# get_group_indices() 결과 리스트에서 가져올 순서 (인덱스)
# 원래 순서: 0=G1, 1=G2, 2=G3, 3=G4, 4=G5, 5=G6, 6=G7
ORDER_INDICES = [2, 5, 0, 1, 4, 6, 3]

def compute_rdm(data_matrix, metric='correlation'):
    """
    Representational Dissimilarity Matrix (RDM) 계산
    입력: (Features, Samples) -> 여기서는 (3개 트럭, 78개 시나리오)
    출력: (78, 78) 대칭 행렬 (시나리오 간의 비유사도)
    """
    # pdist 계산 전 NaN 체크 (혹시 몰라서)
    # 여기서는 "NaN을 0으로 채우고 + 분산이 0인 경우 노이즈 추가" 전략
    if np.isnan(data_matrix).any():
        print("⚠️ Warning: Data contains NaNs. Filling with 0.")
        data_matrix = np.nan_to_num(data_matrix)
    
    # 상수 벡터 방지
    # 아주 미세한 노이즈(1e-9)를 더해서 0으로 나누는 에러 방지
    data_matrix += np.random.normal(0, 1e-9, data_matrix.shape)

    # pdist는 행(Row)을 기준으로 거리를 계산하므로 Transpose 필요
    # 입력: (78, 3) -> 78개 시나리오가 각각 3차원(K,L,M) 벡터를 가짐
    dist_vec = pdist(data_matrix.T, metric=metric)
    rdm = squareform(dist_vec)
    return rdm

def plot_rsa_analysis(x_pkl_path, y_pkl_path, output_img_path, 
                      x_name="Model", y_name="Reference"):
    """
    Desire와 Belief의 RSA 결과를 2x2 Grid로 통합 시각화
    Row 1: Desire (Human vs Model)
    Row 2: Belief (Human vs Model)
    추가:
        어떤 두 개의 Pickle 데이터가 들어오든 상관없이 X축과 Y축에 배치
    """
    # 데이터 로드
    with open(x_pkl_path, 'rb') as f:
        data_x = pickle.load(f) # Target, 기본은 model
    with open(y_pkl_path, 'rb') as f:
        data_y = pickle.load(f) # Baseline, 기본은 human

    targets = [
        ("Desire", 'des_inf_mean'),
        ("Belief", 'bel_inf_mean_norm')
    ]
    
    # 2행 2열 캔버스 생성
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    plt.subplots_adjust(hspace=0.3, wspace=0.1)
    
    # [추가] Irrational 시나리오(11, 12, 22, 71, 72)는 분석에서 제외하는 것이 좋습니다.
    # 이들은 애초에 정답이 없거나 이상한 데이터일 확률이 높기 때문입니다.
    groups = get_group_indices(include_irrational=False)
    
    sorted_indices = []
    group_boundaries = [0] # 경계선 위치 저장
    
    for group_idx in ORDER_INDICES:
        g_list = groups[group_idx]

        # 1-based index -> 0-based index 변환
        g_idx_0 = [idx - 1 for idx in g_list]
        sorted_indices.extend(g_idx_0)
        group_boundaries.append(len(sorted_indices)) # 누적 개수 저장

    # X/Y축 라벨을 그룹의 정중앙에 달기 위한 센터 인덱스 계산
    group_centers = []
    for i in range(len(group_boundaries)-1):
        center = (group_boundaries[i] + group_boundaries[i+1]) / 2.0
        group_centers.append(center)

    # 그리기 루프
    for row_idx, (name, key) in enumerate(targets):
        y_data = data_y[key][:, sorted_indices]
        x_data = data_x[key][:, sorted_indices]
        
        # RDM 계산 (78x78 Matrix)
        # metric='euclidean' 또는 'correlation' 사용
        rdm_y = compute_rdm(y_data, 'correlation')
        rdm_x = compute_rdm(x_data, 'correlation')
        
        # Second-order Correlation (RDM 간의 유사도)
        # 상삼각행렬(Upper Triangle)만 추출하여 상관관계 계산
        idx = np.triu_indices(len(rdm_y), k=1)
        score, p_val = spearmanr(rdm_y[idx], rdm_x[idx])
        
        # --- Baseline Plot (Left) ---
        ax_y = axes[row_idx, 0]
        sns.heatmap(rdm_y, ax=ax_y, cmap='viridis', square=True, cbar=False, vmin=0, vmax=2)

        ax_y.set_ylabel(f"{name} Representation", fontsize=16, fontweight='bold')
        if row_idx == 0: ax_y.set_title(f"[{y_name}] Ground Truth", fontsize=14)
        
        # --- Target Plot (Right) ---
        ax_x = axes[row_idx, 1]
        sns.heatmap(rdm_x, ax=ax_x, cmap='viridis', square=True, 
                    cbar_kws={'label': 'Dissimilarity (1-r)'}, vmin=0, vmax=2)
        if row_idx == 0: ax_x.set_title(f"[{x_name}] Prediction", fontsize=14)
        
        # 텍스트 추가 (RSA Score)
        ax_x.text(0.5, -0.1, f"RSA Score (r) = {score:.3f}", transform=ax_x.transAxes, 
                  ha='center', fontsize=16, fontweight='bold', color='blue')

        # 공통 장식 (경계선, 라벨)
        for ax in [ax_y, ax_x]:
            for b in group_boundaries[1:-1]:
                ax.axhline(b, color='white', lw=1.5, ls='--')
                ax.axvline(b, color='white', lw=1.5, ls='--')

            # X, Y축 라벨(Group Names)은 왼쪽 Human 그래프(ax_h)에만 달기!
            if ax == ax_y:
                # X축 라벨
                ax.set_xticks(group_centers)
                ax.set_xticklabels(GROUP_NAMES, rotation=45, ha='right', fontsize=11)
                # Y축 라벨
                ax.set_yticks(group_centers)
                ax.set_yticklabels(GROUP_NAMES, rotation=0, fontsize=11)
            else:
                # Model 쪽은 X, Y축 눈금을 모두 지워서 깔끔하게 여백 유지
                ax.set_xticks([])
                ax.set_yticks([])
    
    plt.suptitle(f"[{x_name} vs {y_name}] RSA Analysis: Structure Comparison", fontsize=20, y=0.94, fontweight='bold')
    
    os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
    plt.savefig(output_img_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ RSA Heatmap saved to: {output_img_path}")