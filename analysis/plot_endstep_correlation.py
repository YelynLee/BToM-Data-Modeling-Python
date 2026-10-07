import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
# 텍스트 가독성을 위한 패스 이펙트 (배경색과 무관하게 글씨를 잘 보이게 함)
import matplotlib.patheffects as path_effects
from scipy.stats import pearsonr
import matplotlib.lines as mlines

# 1. 경로 설정 (기존 프로젝트 구조 반영)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.config import REFERENCE_PKL_DIR, BASE_RESULTS_DIR, HUMAN_PKL_PATH, get_group_indices

# 2. 모델 및 시나리오 설정
REF_MODELS = ['human', 'btom', 'truebelief', 'nocost', 'motionheuristic', 'hindsight']
# 그래프 x축에 실제로 표시할 이름들
DISPLAY_MODELS = ['human', 'btom', 'truebelief', 'nocost', 'motionheur', 'hindsight']

# 💡 [NEW] 모델 그룹 정의 및 앙상블 행 추가
BETTER_MODELS = ['gemini-2.5-pro', 'deepseek-reasoner', 'deepseek-v4-pro', 'claude-opus-4-6']
WORSE_MODELS = ['gpt-4o', 'gpt-5.4', 'o4-mini', 'gemini-2.5-flash', 'deepseek-chat', 'claude-sonnet-4-6']
LLM_MODELS = BETTER_MODELS + WORSE_MODELS
# LLM_MODELS = WORSE_MODELS

PLOT_ROWS = ['Aggregate (Better Models)', 'Aggregate (Worse Models)'] + LLM_MODELS
# PLOT_ROWS = ['Aggregate (Worse Models)'] + LLM_MODELS

# 🌟 [NEW] Human vs BToM 기준값 (Benchmark)
BENCHMARKS = {
    'desire': {'ind': 0.91, 'grp': 0.97},
    'belief': {'ind': 0.78, 'grp': 0.90}
}

def get_valid_indices(total_scenarios=78, include_partial=False):
    """
    비합리적 시나리오를 제외하고,
    include_partial 값에 따라 Check-Partial 그룹의 포함 여부를 결정하여 유효 인덱스 반환
    """
    # 비합리적 시나리오가 이미 제외된 그룹별 시나리오 ID(1-based) 리스트 가져오기
    group_inds = get_group_indices(include_irrational=False)
    
    # 🌟 [수정 포인트 1] 모드에 따른 그룹 인덱스 범위 설정
    # include_partial이 True면 7개 그룹(0~6), False면 5개 그룹(0~4) 사용
    num_groups = 7 if include_partial else 5

    # Group 1~5 (인덱스 0~4)에 해당하는 시나리오 ID만 수집 (Check-Partial 제외)
    valid_scenarios_1based = []
    for i in range(num_groups): 
        valid_scenarios_1based.extend(group_inds[i])
        
    # 1-based ID를 numpy 배열로 바꾸고 0-based index로 변환
    valid_indices_0based = np.array(valid_scenarios_1based) - 1
    
    # 전체 시나리오 개수만큼 False로 채워진 boolean 배열 생성
    valid_mask = np.zeros(total_scenarios, dtype=bool)
    
    # 유효한(1~5그룹) 인덱스 위치만 True로 활성화
    valid_mask[valid_indices_0based] = True
    
    return valid_mask

def calc_r(x, y):
    """상관계수(r) 계산 (NaN 제외)"""
    x_flat, y_flat = x.flatten(), y.flatten()
    mask = ~np.isnan(x_flat) & ~np.isnan(y_flat)
    x_clean, y_clean = x_flat[mask], y_flat[mask]
    
    if len(x_clean) < 2 or np.std(x_clean) == 0 or np.std(y_clean) == 0:
        return np.nan
        
    r, _ = pearsonr(x_clean, y_clean)
    return r

def load_pickle_safe(path):
    """안전한 Pickle 로드 (파일이 없으면 None 반환)"""
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

def plot_correlation_bars(condition="vanilla", include_partial=False):
    mode_text = "INCL. PARTIAL" if include_partial else "EXCL. PARTIAL"
    print(f"📊 [{condition.upper()} | {mode_text}] 조건의 Grouped Bar Graph 생성을 시작합니다...")
    
    valid_mask = get_valid_indices(include_partial=include_partial)
    
    # ---------------------------------------------------------
    # 1. Reference 데이터 미리 로드 (Human + Cognitive Models)
    # ---------------------------------------------------------
    ref_data_dict = {}
    for ref in REF_MODELS:
        if ref == 'human':
            ref_data_dict[ref] = load_pickle_safe(HUMAN_PKL_PATH)
        else:
            path = os.path.join(REFERENCE_PKL_DIR, ref, f"{ref}_data.pkl")
            ref_data_dict[ref] = load_pickle_safe(path)

    # ---------------------------------------------------------
    # 2. 💡 [NEW] LLM 데이터 로드 및 앙상블(Aggregate) 연산
    # ---------------------------------------------------------
    llm_matrices = {}
    
    for llm in LLM_MODELS:
        llm_path = os.path.join(BASE_RESULTS_DIR, llm, condition, "model_data.pkl")
        data = load_pickle_safe(llm_path)
        if data:
            llm_matrices[llm] = {
                'des_ind': data['des_inf_mean'],
                'bel_ind': data['bel_inf_mean_norm'],
                'des_grp': data['des_inf_group_mean'],
                'bel_grp': data['bel_inf_group_mean']
            }

    def create_aggregate(models):
        mats = [llm_matrices[m] for m in models if m in llm_matrices]
        if not mats: return None
        return {
            'des_ind': np.nanmean(np.stack([m['des_ind'] for m in mats]), axis=0),
            'bel_ind': np.nanmean(np.stack([m['bel_ind'] for m in mats]), axis=0),
            'des_grp': np.nanmean(np.stack([m['des_grp'] for m in mats]), axis=0),
            'bel_grp': np.nanmean(np.stack([m['bel_grp'] for m in mats]), axis=0)
        }

    # Aggregate 데이터 딕셔너리에 추가
    llm_matrices['Aggregate (Better Models)'] = create_aggregate(BETTER_MODELS)
    llm_matrices['Aggregate (Worse Models)'] = create_aggregate(WORSE_MODELS)

    # ---------------------------------------------------------
    # 3. 캔버스 준비 (Rows: LLMs, Cols: Desire, Belief)
    # ---------------------------------------------------------
    n_rows = len(PLOT_ROWS)
    fig, axes = plt.subplots(nrows=n_rows, ncols=2, figsize=(12, 3 * n_rows))
    plt.subplots_adjust(hspace=2, wspace=2)
     
    # 6개 Reference 모델 고유 색상
    colors = ["#91BBEA", "#E54B58FF", '#F5A623', "#DC91EB", "#B8969A", "#747ED2"] # 예전 btom 색: "#9BE8D7"

    # --- 개별 서브플롯 그리기 함수 ---
    def draw_bars_with_group_ext(ax, r_ind_values, r_grp_values, col_type):
        x = np.arange(len(REF_MODELS))
        
        # Human-BToM 기준선 (빨간색 라인) 추가
        ref_ind = BENCHMARKS[col_type]['ind']
        ref_grp = BENCHMARKS[col_type]['grp']

        # Individual 기준선 (실선)
        ax.axhline(ref_ind, color='red', linestyle='--', linewidth=1.5, alpha=0.7, zorder=0)
        # Grouped 기준선 (점선)
        # ax.axhline(ref_grp, color='red', linestyle='--', linewidth=1.2, alpha=0.5, zorder=0)

        # 기준선 텍스트 라벨 (첫 번째 행에만 표시하여 가독성 확보)
        if ax.get_subplotspec().rowspan.start == 0:
            ax.text(len(REF_MODELS)-0.5, ref_ind + 0.06, f'Human-BToM ({ref_ind})', 
                    color='red', fontsize=8, ha='right', va='bottom', fontweight='bold', alpha=0.8)
            # ax.text(len(REF_MODELS)-0.5, ref_grp + 0.02, f'Group ({ref_grp})', 
            #         color='red', fontsize=8, ha='right', va='bottom', alpha=0.6)

        # 가장 높은 상관계수 인덱스 찾기 (NaN 무시)
        valid_inds = [v for v in r_ind_values if not np.isnan(v)]
        max_val = max(valid_inds) if valid_inds else np.nan
        max_idx = r_ind_values.index(max_val) if not np.isnan(max_val) else -1
        
        for i, ref in enumerate(REF_MODELS):
            r_ind = r_ind_values[i]
            r_grp = r_grp_values[i]
            
            is_max = (i == max_idx)
            
            # 최고값 막대 하이라이트 (빨간색 굵은 테두리)
            edge_color = 'black' if is_max else 'black'
            line_width = 2.2 if is_max else 1.0
            
            if np.isnan(r_ind):
                # 데이터가 없는 경우 (Placeholder)
                ax.text(i, 0.05, 'N/A', ha='center', va='bottom', color='gray', fontsize=10, fontweight='bold')
                continue
                
            # 기본 막대 (Individual Correlation) 그리기
            bar = ax.bar(i, r_ind, color=colors[i], edgecolor=edge_color, linewidth=line_width, alpha=0.85, zorder=2)
            
            # Grouped Correlation을 오차 막대(Extension) 느낌으로 추가
            # if not np.isnan(r_grp):
            #     # Individual -> Grouped 로 이어지는 점선
            #     ax.vlines(i, r_ind, r_grp, color='black', linewidth=1.5, linestyle='--', zorder=3)
            #     # Grouped 수치 상단 캡(-) 모양
            #     ax.hlines(r_grp, i - 0.2, i + 0.2, color='black', linewidth=1.5, zorder=3)
                
            # 텍스트 막대 안으로 (혹은 값이 너무 작으면 밖으로)
            font_weight = 'bold' if is_max else 'normal'
            font_size = 11 if is_max else 10
            
            # 값이 너무 작으면 축과 겹치므로 위/아래로 살짝 빼줌, 아니면 막대의 중간(r_ind/2)에 위치
            if abs(r_ind) < 0.15:
                y_pos = r_ind + 0.05 if r_ind >= 0 else r_ind - 0.05
                va = 'bottom' if r_ind >= 0 else 'top'
            else:
                y_pos = r_ind / 2
                va = 'center'
                
            txt = ax.text(i, y_pos, f'{r_ind:.2f}', ha='center', va=va, 
                          color='black', fontweight=font_weight, fontsize=font_size)
            # 글씨가 어떤 배경색에서도 잘 보이도록 흰색 외곽선 이펙트 추가
            txt.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])

        ax.set_ylim(-1.0, 1.15)
        ax.axhline(0, color='black', linewidth=1.2, zorder=1)
        # x축의 좌우 여백(Padding)을 명시적으로 넉넉하게 줘서 hindsight가 안 잘리게 함
        ax.set_xlim(-0.6, len(REF_MODELS) - 0.4)
        ax.set_xticks(x)
        
        ax.set_xticklabels(DISPLAY_MODELS, ha='center', fontsize=11)
            
        ax.grid(axis='y', linestyle=':', alpha=0.6, zorder=1)

    # ---------------------------------------------------------
    # 4. 행(Row) 순회하며 서브플롯 그리기
    # ---------------------------------------------------------
    for row_idx, llm in enumerate(PLOT_ROWS):
        ax_des = axes[row_idx, 0]
        ax_bel = axes[row_idx, 1]

        # 소제목은 첫 번째 행에만, 모델명은 각 행의 Y축(왼쪽) 라벨로 설정
        if row_idx == 0:
            ax_des.set_title("Desire Correlation", fontsize=16, fontweight='bold', pad=15)
            ax_bel.set_title("Belief Correlation", fontsize=16, fontweight='bold', pad=15)

        # 💡 [NEW] 앙상블(Aggregate) 행은 색상과 크기를 다르게 하여 시각적으로 분리
        if llm == 'Aggregate (Better Models)':
            label_color = '#E63946'  # Red
            font_size = 13
        elif llm == 'Aggregate (Worse Models)':
            label_color = '#457B9D'  # Blue
            font_size = 13
        else:
            label_color = 'black'
            font_size = 13

        # Y축 라벨로 LLM 모델명 크게 표시 (마치 행렬의 Row Header처럼)
        ax_des.set_ylabel(llm, fontsize=font_size, fontweight='bold', labelpad=15, color=label_color)
        
        llm_data = llm_matrices.get(llm)
        
        r_des_ind, r_bel_ind = [], []
        r_des_grp, r_bel_grp = [], []

        # 모드에 따른 Grouped Data 추출 열 인덱스 분기
        valid_group_idx = [0, 1, 2, 3, 4, 5, 6] if include_partial else [0, 1, 2, 3, 4]
        
        # 각 Reference 모델과의 Correlation 계산
        for ref in REF_MODELS:
            ref_data = ref_data_dict.get(ref)
            
            if llm_data is None or ref_data is None:
                r_des_ind.append(np.nan); r_bel_ind.append(np.nan)
                r_des_grp.append(np.nan); r_bel_grp.append(np.nan)
                continue
                
            # Individual Data 추출 (valid_mask 적용)
            # x_des_i = llm_data['des_inf_mean'][:, valid_mask]
            # y_des_i = ref_data['des_inf_mean'][:, valid_mask]
            # x_bel_i = llm_data['bel_inf_mean_norm'][:, valid_mask]
            # y_bel_i = ref_data['bel_inf_mean_norm'][:, valid_mask]
            x_des_i = llm_data['des_ind'][:, valid_mask]
            y_des_i = ref_data['des_inf_mean'][:, valid_mask]
            x_bel_i = llm_data['bel_ind'][:, valid_mask]
            y_bel_i = ref_data['bel_inf_mean_norm'][:, valid_mask]
            
            r_des_ind.append(calc_r(x_des_i, y_des_i))
            r_bel_ind.append(calc_r(x_bel_i, y_bel_i))

            # Grouped Data 추출 (이미 그룹핑 과정에서 mask가 적용되어 있으므로 바로 사용)
            # x_des_g = llm_data['des_inf_group_mean'][:, valid_group_idx]
            # y_des_g = ref_data['des_inf_group_mean'][:, valid_group_idx]
            # x_bel_g = llm_data['bel_inf_group_mean'][:, valid_group_idx]
            # y_bel_g = ref_data['bel_inf_group_mean'][:, valid_group_idx]
            x_des_g = llm_data['des_grp'][:, valid_group_idx]
            y_des_g = ref_data['des_inf_group_mean'][:, valid_group_idx]
            x_bel_g = llm_data['bel_grp'][:, valid_group_idx]
            y_bel_g = ref_data['bel_inf_group_mean'][:, valid_group_idx]
            
            r_des_grp.append(calc_r(x_des_g, y_des_g))
            r_bel_grp.append(calc_r(x_bel_g, y_bel_g))

        draw_bars_with_group_ext(ax_des, r_des_ind, r_des_grp, 'desire')
        draw_bars_with_group_ext(ax_bel, r_bel_ind, r_bel_grp, 'belief')

    # Grouped Extension에 대한 범례(Legend) 수동 추가 (우측 상단 서브플롯)
    # custom_line = mlines.Line2D([], [], color='black', linestyle='--', 
    #                             marker='_', markersize=10, markeredgewidth=1.5, 
    #                             label='Grouped Correlation')
    # ref_line = mlines.Line2D([], [], color='red', linestyle='--', linewidth=1.5, 
    #                          label='Human-BToM Correlation')
    # axes[0, 0].legend(handles=[ref_line], loc='lower right', fontsize=10)

    # ---------------------------------------------------------
    # 5. 마무리 및 저장
    # ---------------------------------------------------------
    results_dir = os.path.join(parent_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # 🌟 [수정 포인트 3] 모드에 따른 저장명 및 타이틀 분기
    suffix = "with_partial" if include_partial else "no_partial"

    save_path = os.path.join(results_dir, f"correlation_endstep_supplement_bars_{condition}_{suffix}.png")
    
    plt.suptitle(f"LLM vs Reference Models Correlation (Condition: {condition})", 
                 fontsize=24, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0.3)
    print(f"✅ Bar graph saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    # 터미널에서 조건(vanilla, oneshot 등)을 받을 수 있게 확장 가능
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, default="vanilla")
    # 🌟 [추가 포인트] 터미널에서 --include_partial 플래그를 통해 모드 제어 가능
    parser.add_argument("--include_partial", action="store_true", help="Include Check-Partial (G6, G7) groups")
    args = parser.parse_args()
    
    plot_correlation_bars(condition=args.condition, include_partial=args.include_partial)