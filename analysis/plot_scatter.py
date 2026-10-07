import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

from src.config import BEHAVIOR_GROUPS, REFERENCE_PKL_DIR, HUMAN_PKL_PATH, get_group_indices

# 제외할 비합리적 시나리오 (1-based index)
EXCL_SCENARIOS = [11, 12, 22, 71, 72]

# 축 범위 설정 (MATLAB 코드 참조)
# Desire: 1~7점 척도 (여유 있게 1~7.5)
# Belief: 0~1 확률 (여유 있게 0~1.05)
AXIS_DESIRE = [0.8, 7.2]
AXIS_BELIEF = [-0.05, 1.05]

# 7개 그룹을 뚜렷하게 구분할 컬러 팔레트
GROUP_COLORS = ["#EA6A75", '#F4A261', '#E9C46A', '#2A9D8F', '#264653', '#8AB17D', '#9D4EDD']

# 2. 레퍼런스 모델 팔레트 (마커 점 색상)
REF_MODELS = ['human', 'btom', 'truebelief', 'nocost', 'motionheuristic', 'hindsight']
DISPLAY_MODELS = ['Human', 'BToM', 'TrueBelief', 'NoCost', 'MotionHeur', 'Hindsight']
# REF_COLORS = ['#3498DB', '#2ECC71', '#E74C3C', '#F1C40F', '#9B59B6', '#E67E22']
REF_COLORS = ["#91BBEA", "#9BE8D7", '#F5A623', '#F8E71C', "#B8969A", "#E4A1F2"]

# 🌟 궤적 선의 색상 (Desire: K, L, M / Belief: L, M, Empty)
ENTITY_COLORS = ["#E45252", "#558FB9", "#A0625B"] # 회색, 파란색, 빨간색
ENTITY_LABELS_DES = ['Truck K', 'Truck L', 'Truck M']
ENTITY_LABELS_BEL = ['Truck L', 'Truck M', 'Empty / N']

# 🌟 인지적 난이도 위계에 따른 그룹 인덱스 순서 (0-based)
# 순서: nocheck(p) -> nocheck(a) -> check-stay(p) -> checkpartial(p) -> checkpartial(a) -> check-goback(p) -> check-goback(a)
# (주의: 아래 숫자는 예시입니다. data_x['des_inf_group_mean']의 실제 열 인덱스에 맞게 꼭 수정해주세요!)
HIERARCHY_INDICES = [2, 4, 1, 6, 5, 3, 0]

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

    # 🌟 [추가] X축과 Y축의 시각적 비율을 1:1 완벽한 정사각형으로 고정
    ax.set_aspect('equal', adjustable='box')
    
    ax.set_xlabel(x_label, fontweight='bold', fontsize='large')
    ax.set_ylabel(y_label, fontweight='bold', fontsize='large')
    ax.set_title(f"{title}\n(r = {r:.2f}, RMSE = {rmse:.2f}, N = {n})", fontsize=15)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    return r, rmse

def draw_colored_variance_subplot(ax, x_mean, y_mean, x_err, y_err, title, axis_range, x_label, y_label, show_legend=False):
    """🌟 [NEW] 그룹별 컬러와 양방향 X, Y 분산(Error Bar)을 그려주는 헬퍼 함수"""
    r, rmse, n = calc_stats(x_mean, y_mean)
    
    # 7개의 그룹 순회 (g: 0 ~ 6)
    num_groups = x_mean.shape[1]
    
    for g in range(num_groups):
        color = GROUP_COLORS[g % len(GROUP_COLORS)]
        group_name = BEHAVIOR_GROUPS.get(g + 1, f"Group {g+1}")
        
        # 해당 그룹의 3개 점(Truck K, L, M) 추출
        x_pts = x_mean[:, g]
        y_pts = y_mean[:, g]
        
        # Error 값 안전하게 가져오기
        x_err_pts = x_err[:, g] if x_err is not None else None
        y_err_pts = y_err[:, g] if y_err is not None else None
        
        # 양방향 Error Bar와 함께 점 찍기 (xerr 추가!)
        ax.errorbar(x_pts, y_pts, xerr=x_err_pts, yerr=y_err_pts, 
                    fmt='o', color=color, markersize=6, alpha=0.8, 
                    ecolor=color, elinewidth=1.5, capsize=4, 
                    label=group_name)

    lims = [axis_range[0], axis_range[1]]
    ax.plot(lims, lims, 'k--', alpha=0.3)
    
    ax.set_xlim(axis_range)
    ax.set_ylim(axis_range)
    ax.set_xlabel(x_label, fontweight='bold', fontsize='large')
    ax.set_ylabel(y_label, fontweight='bold', fontsize='large')
    ax.set_title(f"{title}\n(r = {r:.2f}, RMSE = {rmse:.2f})", fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # 범례는 복잡해질 수 있으므로, 요청될 때만 그래프 우측 바깥에 표시
    if show_legend:
        ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=10, 
                  title="Behavior Groups", title_fontsize='11', frameon=True)

# =========================================================================
# 🌟 [NEW] 개별 Trial 데이터를 그룹별 색상으로 찍어주는 함수
# =========================================================================
def draw_colored_individual_subplot(ax, x_full, y_full, x_grp_mean, y_grp_mean, group_indices, title, axis_range, x_label, y_label, show_legend=False):
    """모든 개별 Trial 점들을 찍고, 미리 계산된 그룹 평균 좌표 3개를 속이 빈 네모(Hollow Square)로 오버레이합니다."""
    
    # 전체 통계 계산 (비합리적 시나리오를 제외한 유효 인덱스 마스크 사용)
    valid_mask = get_valid_indices(x_full.shape[1])
    r, rmse, n = calc_stats(x_full[:, valid_mask], y_full[:, valid_mask])
    
    excl_idx = np.array(EXCL_SCENARIOS) - 1
    
    # 범례 커스텀을 위한 핸들 리스트
    legend_handles = []
    
    # 각 그룹별로 시나리오 인덱스를 찾아 점을 찍음
    for g in range(len(group_indices)):
        color = GROUP_COLORS[g % len(GROUP_COLORS)]
        group_name = BEHAVIOR_GROUPS.get(g + 1, f"Group {g+1}")
        
        # 0-based 인덱스로 변환 및 비합리적 시나리오 배제
        g_idx_0based = np.array(group_indices[g]) - 1
        valid_g_idx = [i for i in g_idx_0based if i not in excl_idx]
        
        if not valid_g_idx:
            continue
            
        # 1. 개별 데이터 추출 및 Flatten
        x_pts = x_full[:, valid_g_idx].flatten()
        y_pts = y_full[:, valid_g_idx].flatten()
        
        # 2. [미시적 뷰] 개별 Scatter Plot (작고 투명하게)
        ax.scatter(x_pts, y_pts, color=color, s=40, alpha=0.6, 
                   edgecolors='white', linewidths=0.5)
                   
        # 3. 🌟 미리 계산된 3개의 대표 평균 좌표(Truck K, L, M) 추출 및 빈 네모 그리기
        grp_mean_x = x_grp_mean[:, g]
        grp_mean_y = y_grp_mean[:, g]

        # ax.scatter(grp_mean_x, grp_mean_y, facecolors='none', edgecolors=color, 
        #            marker='D', s=100, linewidths=2.2, zorder=5) # 크기(s), 굵기(linewidths), 최상단 렌더링(zorder)
        
        # 범례에 추가할 개별 그룹 핸들 (꽉 찬 작은 색상 점)
        legend_handles.append(mlines.Line2D([], [], color='white', marker='o', 
                                            markerfacecolor=color, markersize=8, label=group_name))

    lims = [axis_range[0], axis_range[1]]
    ax.plot(lims, lims, 'k--', alpha=0.3)
    
    ax.set_xlim(axis_range)
    ax.set_ylim(axis_range)
    ax.set_xlabel(x_label, fontweight='bold', fontsize='large')
    ax.set_ylabel(y_label, fontweight='bold', fontsize='large')
    ax.set_title(f"{title}\n(r = {r:.2f}, RMSE = {rmse:.2f}, N = {n})", fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    if show_legend:
        # 🌟 범례 맨 아래에 "Group Mean"이 무엇을 의미하는지 안내하는 커스텀 마커 추가
        # mean_handle = mlines.Line2D([], [], color='white', marker='D', markeredgecolor='black', 
        #                             markerfacecolor='none', markersize=8, markeredgewidth=1.5, 
        #                             label='Group Mean')
        # legend_handles.append(mean_handle)
        
        ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1.05, 0.5), 
                  fontsize=10, title="Behavior Groups", title_fontsize='11', frameon=True, labelspacing=0.8)

def load_all_references():
    """모든 레퍼런스 모델의 Pickle 데이터를 한 번에 로드"""
    ref_data_dict = {}
    for ref in REF_MODELS:
        path = HUMAN_PKL_PATH if ref == 'human' else os.path.join(REFERENCE_PKL_DIR, ref, f"{ref}_data.pkl")
        if os.path.exists(path):
            with open(path, 'rb') as f:
                ref_data_dict[ref] = pickle.load(f)
        else:
            ref_data_dict[ref] = None
    return ref_data_dict

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
    
    # 디렉토리 생성
    os.makedirs(os.path.dirname(output_img_path), exist_ok=True)

    # ---------------------------------------------------------
    # 🎨 캔버스 1: 기본 2x2 Scatter Plots (흑백 평균 분석)
    # ---------------------------------------------------------
    # 캔버스 생성 (2x2 Grid)
    fig1, axs1 = plt.subplots(1, 2, figsize=(14, 6.5))
    plt.subplots_adjust(wspace=0.25)
    
    # ---------------------------------------------------------
    # 1. Individual Trial Analyses (Left Column)
    # ---------------------------------------------------------
    
    # (1-1) Desire (Individual)
    x_des = data_x['des_inf_mean'][:, valid_mask]
    y_des = data_y['des_inf_mean'][:, valid_mask]
    
    draw_scatter_subplot(axs1[0], x_des, y_des, 
                 "Individual Trials: Desire", AXIS_DESIRE, x_name, y_name)

    # (1-2) Belief (Individual)
    # Normalized Probability (0~1)
    x_bel = data_x['bel_inf_mean_norm'][:, valid_mask]
    y_bel = data_y['bel_inf_mean_norm'][:, valid_mask] # 이미 Alignment 완료된 데이터 가정
    
    draw_scatter_subplot(axs1[1], x_bel, y_bel, 
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
    y_des_err_y = data_y.get('des_inf_group_sd') 
    
    # draw_scatter_subplot(axs1[0, 1], x_des_grp, y_des_grp, 
    #              "Grouped Trials: Desire", AXIS_DESIRE, x_name, y_name, y_err=y_des_err_y)

    # (2-2) Belief (Grouped)
    x_bel_grp = data_x['bel_inf_group_mean']
    y_bel_grp = data_y['bel_inf_group_mean']
    y_bel_err_y = data_y.get('bel_inf_group_sd')
    
    # draw_scatter_subplot(axs1[1, 1], x_bel_grp, y_bel_grp, 
    #              "Grouped Trials: Belief", AXIS_BELIEF, x_name, y_name, y_err=y_bel_err_y)
    
    # 제목 여백 확보
    fig1.suptitle(f"{x_name} vs {y_name}: Correlation Analysis", fontsize=28, fontweight='bold', y=0.98)
    
    # 상단 공간 비우기
    fig1.tight_layout(rect=[0, 0, 1, 0.95], h_pad=5.0, w_pad=3.0)
    fig1.savefig(output_img_path, dpi=150, bbox_inches='tight', pad_inches=0.5)
    plt.close(fig1)

    # ---------------------------------------------------------
    # 🎨 캔버스 2: 1x2 Color Scatter Plots (그룹 분산 정밀 분석)
    # ---------------------------------------------------------
    # 저장될 새로운 파일명 생성 (기존 파일명에 _variance 추가)
    base_path, ext = os.path.splitext(output_img_path)
    variance_img_path = f"{base_path}_variance{ext}"
    
    fig2, axs2 = plt.subplots(1, 2, figsize=(14, 6.5))
    plt.subplots_adjust(wspace=0.25)
    
    # X축(Target) 데이터의 SD (양방향 에러바 용도)
    x_des_err_x = data_x.get('des_inf_group_sd')
    x_bel_err_x = data_x.get('bel_inf_group_sd')

    # (2-1) Desire 컬러 분산
    draw_colored_variance_subplot(axs2[0], x_des_grp, y_des_grp, x_des_err_x, y_des_err_y, 
                                  "Desire Variance Analysis", AXIS_DESIRE, x_name, y_name, show_legend=False)

    # (2-2) Belief 컬러 분산 (오른쪽 그래프에 범례 표시)
    draw_colored_variance_subplot(axs2[1], x_bel_grp, y_bel_grp, x_bel_err_x, y_bel_err_y, 
                                  "Belief Variance Analysis", AXIS_BELIEF, x_name, y_name, show_legend=True)

    fig2.suptitle(f"[{x_name} vs {y_name}] Group Variance Analysis", fontsize=22, fontweight='bold', y=1.05)
    fig2.savefig(variance_img_path, dpi=200, bbox_inches='tight')
    plt.close(fig2)

    # ---------------------------------------------------------
    # 🎨 캔버스 3: [NEW] 1x2 Color Scatter Plots (개별 데이터 그룹별 컬러 분석)
    # ---------------------------------------------------------
    individual_colored_img_path = f"{base_path}_individual_colored{ext}"
    
    fig3, axs3 = plt.subplots(1, 2, figsize=(14, 6.5))
    plt.subplots_adjust(wspace=0.25)
    
    # 그룹별 시나리오 인덱스를 불러옴 (irrational 포함 7개 전체 맵 로드 후 함수 내부에서 걸러냄)
    group_inds = get_group_indices(include_irrational=True)

    draw_colored_individual_subplot(axs3[0], 
                                    data_x['des_inf_mean'], data_y['des_inf_mean'],
                                    x_des_grp, y_des_grp,
                                    group_inds, "Desire: Individual Trials", 
                                    AXIS_DESIRE, x_name, y_name, show_legend=False)
                                    
    draw_colored_individual_subplot(axs3[1], 
                                    data_x['bel_inf_mean_norm'], data_y['bel_inf_mean_norm'], 
                                    x_bel_grp, y_bel_grp,
                                    group_inds, "Belief: Individual Trials", 
                                    AXIS_BELIEF, x_name, y_name, show_legend=True)

    fig3.suptitle(f"[{x_name} vs {y_name}] Individual Trial Dissection", fontsize=22, fontweight='bold', y=1.05)
    fig3.savefig(individual_colored_img_path, dpi=200, bbox_inches='tight')
    plt.close(fig3)
    
    print(f"✅ Basic Scatter Plot saved to: {output_img_path}")
    print(f"✅ Variance Scatter Plot saved to: {variance_img_path}")
    print(f"✅ Individual Colored Scatter Plot saved to: {individual_colored_img_path}")