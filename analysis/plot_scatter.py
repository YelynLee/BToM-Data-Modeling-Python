import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

# 제외할 비합리적 시나리오 (1-based index)
EXCL_SCENARIOS = [11, 12, 22, 71, 72]

# 축 범위 설정 (MATLAB 코드 참조)
# Desire: 1~7점 척도 (여유 있게 1~7.5)
# Belief: 0~1 확률 (여유 있게 0~1.05)
AXIS_DESIRE = [1, 7.5]
AXIS_BELIEF = [0, 1.05]

def get_valid_indices(total_scenarios=78):
    """제외할 시나리오를 뺀 유효 인덱스(0-based) 반환"""
    all_indices = np.arange(total_scenarios)
    excl_indices = np.array(EXCL_SCENARIOS) - 1
    valid_mask = ~np.isin(all_indices, excl_indices)
    return valid_mask

def calc_stats(x, y):
    """상관계수(r)와 RMSE 계산 (NaN 제외)"""
    # 1D로 펼치기
    x_flat = x.flatten()
    y_flat = y.flatten()
    
    # NaN 제거
    mask = ~np.isnan(x_flat) & ~np.isnan(y_flat)
    x_clean = x_flat[mask]
    y_clean = y_flat[mask]
    
    if len(x_clean) < 2:
        return 0.0, 0.0, 0.0 # 데이터 부족
        
    r, p_val = pearsonr(x_clean, y_clean)
    rmse = np.sqrt(mean_squared_error(x_clean, y_clean))
    
    return r, rmse, len(x_clean)

def draw_scatter_subplot(ax, x_data, y_data, title, axis_range, 
                         x_label, y_label, y_err=None):
    """서브플롯 그리기 헬퍼 함수"""
    # 통계 계산
    r, rmse, n = calc_stats(x_data, y_data)
    
    # 산점도 그리기
    # x: Model, y: Human
    ax.scatter(x_data.flatten(), y_data.flatten(), color='black', s=20, alpha=0.6, label='Data Points')
    
    # Error Bar가 있다면 추가 (Group Analysis용)
    if y_err is not None:
        ax.errorbar(x_data.flatten(), y_data.flatten(), 
                    yerr=y_err.flatten(), fmt='none', ecolor='black', elinewidth=1, capsize=3)

    # 45도 대각선 (Reference Line)0
    lims = [axis_range[0], axis_range[1]]
    ax.plot(lims, lims, 'k--', alpha=0.3, label='Perfect Fit')
    
    # 스타일링
    ax.set_xlim(axis_range)
    ax.set_ylim(axis_range)
    ax.set_xlabel(x_label, fontweight='bold', fontsize='large')
    ax.set_ylabel(y_label, fontweight='bold', fontsize='large')
    ax.set_title(f"{title}\n(r = {r:.2f}, RMSE = {rmse:.2f}, N = {n})", fontsize=15)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    return r, rmse

def plot_scatter_analysis(x_pkl_path, y_pkl_path, output_img_path, 
                          x_name="Model", y_name="Reference"):
    """
    메인 plotting 함수
    Args:
        model_pkl_path: 모델 Pickle 경로
        output_img_path: 저장할 이미지 경로
        model_name: 모델 이름 (제목용)
    추가:
        어떤 두 개의 Pickle 데이터가 들어오든 상관없이 X축과 Y축에 배치
    """
    # 1. 데이터 로드
    with open(x_pkl_path, 'rb') as f:
        data_x = pickle.load(f) # Target, 기본은 model
    with open(y_pkl_path, 'rb') as f:
        data_y = pickle.load(f) # Baseline, 기본은 human

    # 유효 데이터 인덱스 (Irrational 제외)
    valid_mask = get_valid_indices()
    
    # 캔버스 생성 (2x2 Grid)
    fig, axs = plt.subplots(2, 2, figsize=(11, 9))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # ---------------------------------------------------------
    # 1. Individual Trial Analyses (Left Column)
    # ---------------------------------------------------------
    
    # (1-1) Desire (Individual)
    x_des = data_x['des_inf_mean'][:, valid_mask]
    y_des = data_y['des_inf_mean'][:, valid_mask]
    
    draw_scatter_subplot(axs[0, 0], x_des, y_des, 
                 "Individual Trials: Desire", AXIS_DESIRE, x_name, y_name)

    # (1-2) Belief (Individual)
    # Normalized Probability (0~1)
    x_bel = data_x['bel_inf_mean_norm'][:, valid_mask]
    y_bel = data_y['bel_inf_mean_norm'][:, valid_mask] # 이미 Alignment 완료된 데이터 가정
    
    draw_scatter_subplot(axs[1, 0], x_bel, y_bel, 
                 "Individual Trials: Belief", AXIS_BELIEF, x_name, y_name)

    # ---------------------------------------------------------
    # 2. Grouped Trial Analyses (Right Column)
    # ---------------------------------------------------------
    
    # (2-1) Desire (Grouped)
    # MATLAB: des_inf_group_mean 사용
    x_des_grp = data_x['des_inf_group_mean']
    y_des_grp = data_y['des_inf_group_mean']
    
    # Error Bar용 SD (MATLAB: des_inf_group_sd)
    # human_data.pkl 만들 때 _se 혹은 _sd를 매핑했으므로 확인
    # 만약 키가 없다면 None 처리
    y_des_err = data_y.get('des_inf_group_sd') 
    
    draw_scatter_subplot(axs[0, 1], x_des_grp, y_des_grp, 
                 "Grouped Trials: Desire", AXIS_DESIRE, x_name, y_name, y_err=y_des_err)

    # (2-2) Belief (Grouped)
    x_bel_grp = data_x['bel_inf_group_mean']
    y_bel_grp = data_y['bel_inf_group_mean']
    y_bel_err = data_y.get('bel_inf_group_sd')
    
    draw_scatter_subplot(axs[1, 1], x_bel_grp, y_bel_grp, 
                 "Grouped Trials: Belief", AXIS_BELIEF, x_name, y_name, y_err=y_bel_err)

    # 3. 저장 및 출력
    os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
    
    # 제목 여백 확보
    plt.suptitle(f"{x_name} vs {y_name}: Correlation Analysis", fontsize=28, fontweight='bold', y=0.98)
    
    # 상단 공간 비우기
    plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=5.0, w_pad=3.0)
    
    plt.savefig(output_img_path, dpi=150, bbox_inches='tight', pad_inches=0.5)
    
    plt.show()
    
    print(f"✅ Scatter Plots saved to: {output_img_path}")