import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 현재 스크립트(analysis 폴더)의 상위 경로를 파이썬 탐색 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) # 상위 폴더 (프로젝트 루트)

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.config import BASE_RESULTS_DIR, get_group_indices

GROUP_NAMES = [
    "No Check(P)", 
    "Check-Partial(P)", 
    "Check-GoBack(P)", 
    "Check-Stay(P)", 
    "No Check(A)", 
    "Check-Partial(A)", 
    "Check-GoBack(A)"
]
ORDER_INDICES = [2, 5, 0, 1, 4, 6, 3]

def compute_error_scores(df):
    """모든 지표(1~6번)의 단일 오차 점수(Error Score) 산출"""
    
    # 0. Baseline
    df['err_kl'] = df['base_kl_mean']
    df['err_rmse'] = df['base_desire_rmse']
    
    # 1. TrueBelief (LLM이 BToM보다 보이지 않는 대상을 확신하는 정도)
    df['err_tb'] = df['tb_llm_belief_t1'] - df['tb_btom_prob_t1']
    
    # 2. NoCost (실제 도달한 G2 타겟의 최종 Desire 오차: LLM - BToM)
    def get_nocost_error(row):
        target = row['actual_g2']
        if target in ['K', 'L', 'M'] and f'nc_btom_des_{target}_end' in row:
            return abs(row[f'nc_llm_des_{target}_end'] - row[f'nc_btom_des_{target}_end'])
        return np.nan
    df['err_nc'] = df.apply(get_nocost_error, axis=1)

    # 3. MotionHeuristic (초기 K 가치 오차)
    df['err_mh'] = df['mh_llm_des_K_t1'] - df['mh_btom_des_K_t1']

    # 4. Hindsight Bias (관측 전후의 요동폭 차이)
    llm_hb_delta = df['hb_llm_post'] - df['hb_llm_pre']
    btom_hb_delta = df['hb_btom_post'] - df['hb_btom_pre']
    df['err_hb'] = llm_hb_delta - btom_hb_delta
    
    # 5. Rational Consistency (기대효용 수식 위반 비율)
    df['err_rc'] = df['rc_llm_violation_ratio'] # BToM은 위반율 0%로 간주
    
    # 6. ZeroSum Bias (가치 하락 페널티 L + M 합산)
    penalty_L = df['zs_btom_delta_L'] - df['zs_llm_delta_L']
    penalty_M = df['zs_btom_delta_M'] - df['zs_llm_delta_M']
    df['err_zs'] = penalty_L + penalty_M
    
    return df

def plot_scatter_biases(sc_df, target_dir, model_name):
    """2번(NoCost)과 3번(MotionHeuristic) 편향을 위한 산점도 및 회귀선 시각화"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # [1] MotionHeuristic (초기 거리차 vs K 초기 가치)
    sns.regplot(data=sc_df, x='mh_dist_diff', y='mh_llm_des_K_t1', 
                ax=axes[0], color='red', label='LLM', scatter_kws={'alpha':0.6})
    sns.regplot(data=sc_df, x='mh_dist_diff', y='mh_btom_des_K_t1', 
                ax=axes[0], color='blue', label='BToM', scatter_kws={'alpha':0.6})
    axes[0].set_title(f"[{model_name}] Motion Heuristic: Distance vs Target Value", fontsize=14)
    axes[0].set_xlabel("Distance Diff (G1 - G2)", fontsize=12)
    axes[0].set_ylabel("Initial Desire for K", fontsize=12)
    axes[0].legend()

    # [2] NoCost (행동 경로 길이 vs 최종 가치)
    # LLM이 도달한 타겟의 최종 점수 모으기
    sc_df['final_desire_llm'] = sc_df.apply(lambda r: r[f"nc_llm_des_{r['actual_g2']}_end"] if r['actual_g2'] in ['K','L','M'] else np.nan, axis=1)
    sc_df['final_desire_btom'] = sc_df.apply(lambda r: r[f"nc_btom_des_{r['actual_g2']}_end"] if r['actual_g2'] in ['K','L','M'] and f"nc_btom_des_{r['actual_g2']}_end" in r else np.nan, axis=1)

    sns.regplot(data=sc_df, x='path_length', y='final_desire_llm', 
                ax=axes[1], color='red', label='LLM', scatter_kws={'alpha':0.6})
    if sc_df['final_desire_btom'].notna().any():
        sns.regplot(data=sc_df, x='path_length', y='final_desire_btom', 
                    ax=axes[1], color='blue', label='BToM', scatter_kws={'alpha':0.6})
    axes[1].set_title(f"[{model_name}] No Cost: Effort(Path Length) vs Final Value", fontsize=14)
    axes[1].set_xlabel("Path Length (Timesteps)", fontsize=12)
    axes[1].set_ylabel("Final Desire for Chosen Target", fontsize=12)
    axes[1].legend()

    plt.tight_layout()
    plot_path = os.path.join(target_dir, "bias_scatter_plots.png")
    plt.savefig(plot_path, dpi=300)
    print(f"✅ 산점도(Scatter Plots) 저장 완료: {plot_path}")
    plt.show()

def analyze_and_plot_biases(model_name, condition, n_extremes=3, filter_start_x=None, filter_wall_x=None, filter_wall_width=None):
    print(f"\n📊 [Plot Bias] 팩토리얼 시각화 및 Worst 시나리오 추출: {model_name} - {condition}")
    
    target_dir = os.path.join(BASE_RESULTS_DIR, model_name, condition, "everystep")
    data_path = os.path.join(target_dir, "bias_indicators.csv")
    
    if not os.path.exists(data_path):
        print(f"❌ 데이터가 없습니다: {data_path}")
        return
        
    df_raw = pd.read_csv(data_path)
    df_err = compute_error_scores(df_raw)
    
    # 🌟 시나리오 단위로 평균 집계 (Subject 변동성을 시나리오 기준으로 응축)
    group_cols = ['scenario_id', 'group_desc', 'is_irrational', 'actual_g2', 'path_length', 'start_x', 'start_y', 'wall_x', 'wall_y', 'wall_width']
    error_cols = ['err_kl', 'err_rmse', 'err_tb', 'err_nc', 'err_mh', 'err_hb', 'err_rc', 'err_zs']
    
    # 에러 및 플롯 산출에 필요한 모든 컬럼을 평균/첫값 등으로 가져옴
    agg_dict = {col: 'mean' for col in error_cols}
    # Scatter용 컬럼들 추가
    scatter_cols = ['mh_dist_diff', 'mh_llm_des_K_t1', 'mh_btom_des_K_t1']
    for t in ['K', 'L', 'M']:
        scatter_cols.extend([f'nc_llm_des_{t}_end'])
        if f'nc_btom_des_{t}_end' in df_err.columns:
            scatter_cols.append(f'nc_btom_des_{t}_end')
            
    for col in scatter_cols:
        if col in df_err.columns: agg_dict[col] = 'mean'

    sc_df = df_err.groupby(group_cols).agg(agg_dict).reset_index()
    
    # 🌟 Z-score 정규화 (지표별 스케일 통일)
    z_cols = []
    for col in error_cols:
        z_col = f"z_{col}"
        # 값이 다 똑같아서 std가 0일 경우 NaN 방지
        std_val = sc_df[col].std(ddof=1)
        if pd.isna(std_val) or std_val == 0:
            sc_df[z_col] = 0.0
        else:
            sc_df[z_col] = (sc_df[col] - sc_df[col].mean()) / std_val
        z_cols.append(z_col)
        
    # 통합 편향 점수(Global Bias Score) 산출 (NaN 무시하고 계산)
    sc_df['Global_Bias_Score'] = sc_df[z_cols].mean(axis=1, skipna=True)

    # =========================================================
    # 🌟 그룹별 정렬 로직 (Irrational 포함, 그룹 내 Global 내림차순)
    # =========================================================
    groups = get_group_indices(include_irrational=True)
    
    ordered_scenarios = []
    group_boundaries = [0]
    
    for idx in ORDER_INDICES:
        g_list = groups[idx]
        # 해당 그룹에 속한 시나리오들만 추출하여 Global_Bias_Score 기준 내림차순 정렬
        sub_df = sc_df[sc_df['scenario_id'].isin(g_list)].sort_values('Global_Bias_Score', ascending=False)
        ordered_scenarios.extend(sub_df['scenario_id'].tolist())
        group_boundaries.append(len(ordered_scenarios))
        
    group_centers = [(group_boundaries[i] + group_boundaries[i+1]) / 2.0 for i in range(len(group_boundaries)-1)]

    # 지정된 순서대로 데이터프레임 강제 정렬
    sc_df_sorted = sc_df.set_index('scenario_id').loc[ordered_scenarios].reset_index()

    # =========================================================
    # 1. 다차원 편향 히트맵 시각화
    # =========================================================
    plt.figure(figsize=(14, 14))

    heatmap_data = sc_df_sorted[z_cols]
    
    # 가독성을 위해 컬럼명 변경 (2, 3번 포함)
    heatmap_data.columns = ['KL Div', 'RMSE', 'TrueBelief', 'NoCost', 'MotionHeur', 'Hindsight', 'Rational', 'ZeroSum']
    
    # 히트맵 그리기
    ax = sns.heatmap(heatmap_data, cmap='Reds', center=0, annot=False, 
                     cbar_kws={'label': 'Z-Score (Higher = More Biased)'})
    
    # 그룹 간 경계선(흰색 점선) 추가
    for b in group_boundaries[1:-1]:
        ax.axhline(b, color='white', lw=1.5, ls='--')
        
    # Y축 눈금(Ticks)을 그룹 중앙에 맞추고 라벨링
    ax.set_yticks(group_centers)
    ax.set_yticklabels(GROUP_NAMES, rotation=0, fontsize=12, fontweight='bold')

    plt.title(f"Multidimensional Bias Heatmap ({model_name.upper()})", fontsize=20, fontweight='bold', pad=20)
    plt.ylabel("Behavior Groups", fontsize=15, labelpad=10)
    plt.xlabel("Cognitive Bias Indicators", fontsize=15, labelpad=10)
    plt.tight_layout()
    
    plot_path = os.path.join(target_dir, "bias_heatmap.png")
    plt.savefig(plot_path, dpi=300)
    print(f"✅ 히트맵 저장 완료: {plot_path}")
    plt.show()
    
    # =========================================================
    # 2. 산점도(Scatter Plot) 시각화 (2번, 3번 지표)
    # =========================================================
    plot_scatter_biases(sc_df_sorted, target_dir, model_name)
    plt.show()

    # =========================================================
    # 3. 통합 Best N & Worst N DataFrame 추출 (plot_everystep.py 연계용)
    # =========================================================

    # 🌟 [NEW] 필터링 전, 전체 시나리오 대상의 실제 랭크 사전 계산
    # 모든 평가 지표 리스트 (2번 NoCost, 3번 MotionHeuristic 포함 8개 + Global)
    all_display_names = ['KL_Div', 'RMSE', 'TrueBelief', 'NoCost', 'MotionHeur', 'Hindsight', 'RationalCons', 'ZeroSum']
    all_err_cols = ['err_kl', 'err_rmse', 'err_tb', 'err_nc', 'err_mh', 'err_hb', 'err_rc', 'err_zs']
    
    indicators = [('Global_Bias', 'Global_Bias_Score')] + list(zip(all_display_names, all_err_cols))

    # 각 지표별로 전체 중 오름차순(Best), 내림차순(Worst) 랭크를 sc_df에 새 컬럼으로 추가
    # 동점일 경우 최소 순위(min)를 부여하며, NaN 값은 순위에서 제외됨
    for _, err_col in indicators:
        sc_df[f'{err_col}_rank_best'] = sc_df[err_col].rank(method='min', ascending=True).astype('Int64')
        sc_df[f'{err_col}_rank_worst'] = sc_df[err_col].rank(method='min', ascending=False).astype('Int64')

    # 🌟 [NEW] 터미널 입력 조건에 따른 동적 필터링 적용
    filtered_df = sc_df.copy()
    applied_filters = []
    
    if filter_start_x is not None:
        filtered_df = filtered_df[filtered_df['start_x'] == filter_start_x]
        applied_filters.append(f"start_x={filter_start_x}")
        
    if filter_wall_x is not None:
        filtered_df = filtered_df[filtered_df['wall_x'] == filter_wall_x]
        applied_filters.append(f"wall_x={filter_wall_x}")
        
    if filter_wall_width is not None:
        filtered_df = filtered_df[filtered_df['wall_width'] == filter_wall_width]
        applied_filters.append(f"wall_width={filter_wall_width}")
        
    filter_msg = ", ".join(applied_filters) if applied_filters else "None (All Scenarios)"

    print("\n" + "="*90)
    print(f" 🚨 [Extremes Report] 각 평가 지표별 Best {n_extremes} & Worst {n_extremes} 시나리오")
    print(f" 🔍 [Applied Filters] {filter_msg}")
    print("="*90)
    
    extremes_dfs = []
    
    # 🌟 터미널 출력용 컬럼 (읽기 편하게 물리적 특성 모두 포함)
    print_cols = ['rank_type', 'rank', 'scenario_id', 'group_desc', 'is_irrational',
                  'path_length', 'start_x', 'wall_x', 'wall_width', 'error_score']
                  
    # 🌟 DataFrame 저장/전달용 컬럼
    out_cols = ['indicator_name', 'rank_type', 'rank', 'scenario_id', 
                'group_desc', 'is_irrational', 'error_score']
    
    for display_name, err_col in indicators:
        print(f"\n🏆 [{display_name}] 지표")
        
        # 해당 지표가 예외처리(NaN)된 시나리오는 제외
        valid_df = filtered_df.dropna(subset=[err_col]).copy()

        # 필터링 결과 해당 지표에 유효한 시나리오가 아예 없을 경우 스킵
        if valid_df.empty:
            print("   ⚠️ 조건에 맞는 유효한 시나리오가 없습니다.")
            continue
        
        # 🟢 Best N (필터링 목록 내에서 오름차순 정렬 후 상위 N개 추출)
        best_n = valid_df.sort_values(err_col, ascending=True).head(n_extremes).copy()
        best_n['indicator_name'] = display_name
        best_n['rank_type'] = f'Top {n_extremes} (Best)'
        best_n['rank'] = best_n[f'{err_col}_rank_best'] # 전체 기준 실제 랭크 할당
        best_n['error_score'] = best_n[err_col]
        
        # 🔴 Worst N (필터링 목록 내에서 내림차순 정렬 후 상위 N개 추출)
        worst_n = valid_df.sort_values(err_col, ascending=False).head(n_extremes).copy()
        worst_n['indicator_name'] = display_name
        worst_n['rank_type'] = f'Bottom {n_extremes} (Worst)'
        worst_n['rank'] = worst_n[f'{err_col}_rank_worst'] # 전체 기준 실제 랭크 할당
        worst_n['error_score'] = worst_n[err_col]
        
        # 터미널 출력 (물리적 특성 포함)
        disp_df = pd.concat([best_n, worst_n])

        # 출력하려는 컬럼이 데이터프레임에 모두 존재하는지 안전망 추가
        actual_print_cols = [col for col in print_cols if col in disp_df.columns]
        print(disp_df[actual_print_cols].to_string(index=False, float_format="%.3f"))
        
        # 전달용 DataFrame 누적 (핵심 컬럼만)
        actual_out_cols = [col for col in out_cols if col in disp_df.columns]
        extremes_dfs.extend([best_n[actual_out_cols], worst_n[actual_out_cols]])

    # 조건에 맞는 데이터가 전혀 없어서 extremes_dfs가 비어있는 경우 방어
    if not extremes_dfs:
        print("\n❌ 지정한 필터 조건에 해당하는 데이터가 없어 CSV를 생성하지 않습니다.")
        return None

    # 최종 병합
    df_extremes_all = pd.concat(extremes_dfs, ignore_index=True)
    
    out_csv_path = os.path.join(target_dir, f"extremes_top{n_extremes}_{filter_msg}_summary.csv")
    df_extremes_all.to_csv(out_csv_path, index=False)
    
    print("\n" + "="*90)
    print(f"✅ Best/Worst 시나리오 통합 DataFrame 저장 완료: {out_csv_path}")

    return df_extremes_all

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gemini-2.5-flash")
    parser.add_argument("--condition", type=str, default="vanilla")
    parser.add_argument("--n_extremes", type=int, default=3, help="Best/Worst 추출 개수")
    parser.add_argument("--start_x", type=float, default=None, help="Agent 시작 X 좌표 필터")
    parser.add_argument("--wall_x", type=float, default=None, help="벽의 X 좌표 필터")
    parser.add_argument("--wall_width", type=float, default=None, help="벽의 너비 필터")
    args = parser.parse_args()

    analyze_and_plot_biases(
        model_name=args.model,
        condition=args.condition,
        n_extremes=args.n_extremes,
        filter_start_x=args.start_x,
        filter_wall_x=args.wall_x,
        filter_wall_width=args.wall_width
    )