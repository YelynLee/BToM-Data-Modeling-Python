import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from src.prepare_everystep import load_btom_everystep

# 1. 현재 스크립트(analysis 폴더)의 상위 경로를 파이썬 탐색 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) # 상위 폴더 (프로젝트 루트)

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.config import BASE_RESULTS_DIR, BEHAVIOR_GROUPS, get_group_indices

# =========================================================================
# 1. Phase-Normalization (위상 정규화) 로직
# =========================================================================
def get_phase_index(group_id, phase_name):
    """
    각 그룹별로 Phase가 발생해야 하는 논리적 순서(Integer Bin)를 매핑합니다.
    서로 다른 시나리오라도 같은 Phase면 같은 X축 구간(예: 1.0 ~ 2.0)에 놓이게 됩니다.
    """
    if phase_name == 'Start': return 0
    if phase_name == 'Approach G1': return 1
    
    # 그룹별 고유 Phase 매핑
    if group_id in [1, 4]: # Check-GoBack
        if phase_name == 'Pass G1': return 2
        if phase_name == 'See & Reject G2': return 3
        if phase_name == 'Stop': return 4
        if phase_name == 'Return G1': return 5
        if phase_name == 'Selected': return 6
    elif group_id == 2: # Check-Stay
        if phase_name == 'Pass G1': return 2
        if phase_name == 'See & Approach G2': return 3
        if phase_name == 'Stop': return 4
        if phase_name == 'Approach G2': return 5
        if phase_name == 'Selected': return 6
    elif group_id in [3, 5]: # No Check
        if phase_name == 'Selected': return 2
    elif group_id in [6, 7]: # Check-Partial
        if phase_name == 'Pass G1': return 2
        if phase_name == 'See G2': return 3
        if phase_name == 'Stop between G1 and G2': return 4
    
    return 8 # Unknown

def get_group_phase_labels(group_id):
    """X축 하단에 표시될 라벨 텍스트 생성"""
    labels = {0: 'Start', 1: 'Appr G1'}

    if group_id in [1, 4]:
        labels.update({2: 'Pass G1', 3: 'See&Rej G2', 4: 'Stop', 5: 'Return G1', 6: 'Selected'})
    elif group_id == 2:
        labels.update({2: 'Pass G1', 3: 'See&Appr G2', 4: 'Stop', 5: 'Appr G2', 6: 'Selected'})
    elif group_id in [3, 5]:
        labels.update({2: 'Selected'})
    elif group_id in [6, 7]:
        labels.update({2: 'Pass G1', 3: 'See G2', 4: 'Stop Btw'})

    return labels

def normalize_scenario_x(df_sc):
    """단일 시나리오 내에서 time_step을 Phase 구간(Bin)으로 정규화합니다."""
    df_sc = df_sc.copy()
    df_sc['x_norm'] = 0.0
    group_id = df_sc['group_id'].iloc[0]
    
    for phase in df_sc['phase'].unique():
        idx = get_phase_index(group_id, phase)
        mask = df_sc['phase'] == phase
        t_vals = df_sc.loc[mask, 'time_step']
        
        if len(t_vals) == 1:
            # 해당 Phase가 1개 타임스텝뿐이면 구간의 한가운데(0.5)에 배치
            df_sc.loc[mask, 'x_norm'] = idx + 0.5
        else:
            # 여러 타임스텝이면 구간[idx, idx+1] 내에 균등 분배
            t_min, t_max = t_vals.min(), t_vals.max()
            df_sc.loc[mask, 'x_norm'] = idx + (t_vals - t_min) / (t_max - t_min)
            
    return df_sc.sort_values('time_step')

# =========================================================================
# 2. 플롯 시각화 헬퍼 함수
# =========================================================================
def plot_score_figure(df_data, df_btom_data, score_type, title_prefix, cols, colors, labels, output_dir, perfect_count):
    """
    score_type: 'Desire' 또는 'Belief'
    cols: 그릴 컬럼 리스트 (예: ['desire_K', 'desire_L', ...])
    colors: 선 색상 리스트
    labels: 범례 라벨 리스트
    추가:
        LLM과 BToM 간의 오차(MAE)를 계산하여, Phase 구간별로 배경에 셰이딩(Shading)을 추가합니다.
        - 초록색 배경: BToM(정답)과 매우 유사하게 판단한 구간 (오차 낮음)
        - 빨간색 배경: BToM(정답)과 크게 다르게 판단한 구간 (오차 높음)
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharey=True)
    axes = axes.flatten()
    
    # 🌟 [NEW] 추세(Trend) 일치도 계산 로직
    trend_match_rates = None
    if df_btom_data is not None:
        # 데이터 병합 (LLM과 BToM)
        btom_cols = ['scenario_id', 'time_step', 'group_id', 'phase'] + cols
        merged = pd.merge(df_data, df_btom_data[btom_cols], on=['scenario_id', 'time_step', 'group_id', 'phase'], suffixes=('', '_btom'))
        
        # 시나리오별로 정렬 후 변화량(diff) 계산
        merged = merged.sort_values(['scenario_id', 'time_step'])
        
        # 0.05 미만의 미세한 변화는 무시하고 '유지(0)'로 간주하는 헬퍼 함수
        def get_trend_sign(series, tol=0.05):
            diff = series.diff()
            return np.where(diff > tol, 1.0, np.where(diff < -tol, -1.0, 0.0))

        match_scores = []
        for col in cols:
            llm_trend = get_trend_sign(merged[col])
            btom_trend = get_trend_sign(merged[col + '_btom'])
            
            # 첫 스텝(t=1)은 변화량이 없으므로 제외하기 위해 nan 처리
            llm_trend[merged['time_step'] == 1] = np.nan
            
            # 방향이 완벽히 일치하면 1, 아니면 0
            match_scores.append(llm_trend == btom_trend)
            
        # 모든 타겟(차량 3개)에 대한 평균 일치 여부
        merged['trend_match'] = np.mean(match_scores, axis=0)
        
        # Group 및 Phase 별 '추세 일치율(Match Rate)' 집계 (0 ~ 1.0)
        trend_match_rates = merged.groupby(['group_id', 'phase'])['trend_match'].mean().reset_index()

    # truck_presence에 따른 선 모양 매핑 (데이터 내 실제 텍스트에 맞춰 키값을 수정하세요!)
    presence_styles = {
        "K and L present": '-',   # 실선
        "K and M present": ':',  # 짧은 점선
    }

    for i in range(1, 8): # Group 1 ~ 7
        ax = axes[i-1]
        group_df = df_data[df_data['group_id'] == i]
        
        if group_df.empty:
            continue
            
        scenarios = group_df['scenario_id'].unique()
        
        # 🌟 [배경 셰이딩 그리기] LLM 데이터의 선을 그리기 전에 배경을 먼저 칠해줍니다.
        if trend_match_rates is not None:
            err_df = trend_match_rates[trend_match_rates['group_id'] == i]
            for _, row in err_df.iterrows():
                p_name = row['phase']
                match_rate = row['trend_match']
                idx = get_phase_index(i, p_name)
                
                if idx == 8 or p_name == 'Start': continue # 시작점(변화량 없음)과 Unknown 제외
                
                # [판단 기준] 
                # 80% 이상 일치: 초록색 (추세를 아주 잘 따라감)
                # 40% 이하 일치: 빨간색 (변화의 방향을 완전히 놓침)
                if match_rate >= 0.80:
                    ax.axvspan(idx, idx+1, color='#2CA02C', alpha=0.15, lw=0)
                elif match_rate <= 0.40:
                    ax.axvspan(idx, idx+1, color='#D62728', alpha=0.10, lw=0)

        # 각 시나리오별로 선(Trajectory)을 연하게(alpha=0.3) 겹쳐 그림
        for sc_id in scenarios:
            sc_data = group_df[group_df['scenario_id'] == sc_id]

            presence_val = str(sc_data['truck_presence'].iloc[0]) 
            l_style = presence_styles.get(presence_val, '-') # 매핑 실패시 기본 실선

            # 기존 LLM 데이터 그리기
            for col, color, label in zip(cols, colors, labels):
                ax.plot(sc_data['x_norm'], sc_data[col], 
                        color=color, linestyle=l_style, alpha=0.5, linewidth=2, 
                        label=label if sc_id == scenarios[0] else "") # 범례는 한 번만
        
        # X축 꾸미기 (점선 및 라벨)
        phase_labels = get_group_phase_labels(i)
        ax.set_xticks(list(phase_labels.keys()))
        ax.set_xticklabels(list(phase_labels.values()), rotation=45, ha='right', fontsize=9)
        
        # Phase 경계선(회색 점선) 추가
        for x_val in phase_labels.keys():
            ax.axvline(x=x_val, color='gray', linestyle='--', alpha=0.3)
            
        # 제목 설정
        group_name = BEHAVIOR_GROUPS.get(i, f"Group {i}")
        ax.set_title(f"{group_name}\n(n={len(scenarios)} scenarios)", fontsize=11, fontweight='bold')
        ax.grid(True, axis='y', linestyle=':', alpha=0.3)

        # Y축 7칸 규격 통일 (Desire 1~7 vs Belief 0~1)
        if score_type == "Desire":
            ax.set_ylim(1, 7)
            ax.set_yticks(range(1, 8))
            if i == 1 or i == 5:
                ax.set_ylabel("Desire Rating (1-7)", fontweight='bold')
        elif score_type == "Belief":
            ax.set_ylim(0, 1.05)
            # 0부터 1까지 7개의 눈금(Tick)을 생성하여 Desire와 칸을 동일하게 맞춤
            ax.set_yticks(np.linspace(0, 1, 7))
            ax.set_yticklabels([f"{val:.2f}" for val in np.linspace(0, 1, 7)])
            if i == 1 or i == 5:
                ax.set_ylabel("Belief Probability", fontweight='bold')
        
    # 8번째 빈 Subplot 삭제
    fig.delaxes(axes[7])
    
    # 전체 제목 및 레이아웃 설정
    # 사용자 요청: 제목이 잘리지 않도록 rect 속성 활용
    fig.suptitle(f"{title_prefix} - Averaged across {perfect_count} 'Perfect' Subjects", 
                 fontsize=18, fontweight='bold')
    
    # 커스텀 범례 생성 (트럭 색상 + Presence 조건)
    legend_elements = []
    for color, label in zip(colors, labels):
        legend_elements.append(mlines.Line2D([0], [0], color=color, lw=3, label=label))
        
    legend_elements.append(mlines.Line2D([], [], color='none', label=' ')) # 공백 추가
    
    legend_elements.append(mlines.Line2D([0], [0], color='black', linestyle='-', lw=2, label='Present: K, L'))
    legend_elements.append(mlines.Line2D([0], [0], color='black', linestyle=':', lw=2, label='Present: K, M'))
    
    # 🌟 [범례] 배경색 의미 추가
    if df_btom_data is not None:
        legend_elements.append(mlines.Line2D([], [], color='none', label=' '))
        legend_elements.append(mpatches.Patch(color='#2CA02C', alpha=0.3, label='Trend Match (≥80%)'))
        legend_elements.append(mpatches.Patch(color='#D62728', alpha=0.2, label='MisaligTrend Mismatch (≤40%)'))
    
    fig.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(0.95, 0.1), fontsize=11, frameon=True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 저장
    # [추가] BToM만 그렸을 때 파일명 충돌 방지 (Optional)
    prefix = "btom_baseline_" if df_btom_data is None and "Baseline" in title_prefix else ""
    save_path = os.path.join(output_dir, f"{prefix}plot_{score_type.lower()}_phase.png")
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved {score_type} plot to {save_path}")
    plt.show() # 메모리 관리를 위해 창 닫기 (자동화 시 필수)

# =========================================================================
# 3. 메인 분석 함수 (run_analysis.py에서 호출할 엔트리포인트)
# =========================================================================
def run_plot_everystep(model_name, condition, target_subjects):
    """
    Args:
        model_name: 모델 이름 (예: gpt-4o)
        condition: 실험 조건 (예: vanilla, oneshot)
        target_subjects: 완벽하게 78개 시나리오를 통과한 피험자 리스트 (예: [3, 5, 6, 13, 16])
    """
    # 1. 경로 설정 (동적 할당)
    target_dir = os.path.join(BASE_RESULTS_DIR, model_name, condition, "everystep")
    data_path = os.path.join(target_dir, "everystep_valid_only.csv")
    
    if not os.path.exists(data_path):
        print(f"❌ Error: Valid-only data not found at {data_path}")
        return
        
    print(f"📥 Loading Valid Everystep data from {target_dir}...")
    df = pd.read_csv(data_path)

    # 2. 타겟 피험자(우등생) 필터링
    df = df[df['subject_id'].isin(target_subjects)]

    # 3. 비합리적 시나리오(Irrational) 제외 필터링
    allowed_groups = get_group_indices(include_irrational=False)
    allowed_scenarios = [sc for group in allowed_groups for sc in group]
    df = df[df['scenario_id'].isin(allowed_scenarios)]

    if df.empty:
        print("❌ Error: 필터링 후 남은 데이터가 없습니다.")
        return

    # 4. 피험자 간 평균 계산
    SCORE_COLS = ['desire_K', 'desire_L', 'desire_M', 'belief_L', 'belief_M', 'belief_Empty']
    
    df_mean = df.groupby(['scenario_id', 'group_id', 'time_step', 'phase', 'truck_presence'])[SCORE_COLS].mean().reset_index()

    # -------------------------------------------------------------
    # 🌟 [수정] 예전 코드와 100% 동일한 Belief 정규화 로직 적용
    # (평균을 구한 'df_mean'을 바탕으로 1을 빼지 않고 곧바로 비율 계산)
    # -------------------------------------------------------------
    belief_cols = ['belief_L', 'belief_M', 'belief_Empty']
    
    # 1. 각 행(row)별로 L, M, Empty 평균값의 합을 구함
    bel_sum = df_mean[belief_cols].sum(axis=1)
    
    # 2. 0으로 나누는 에러를 방지 (예전 코드의 bel_sum[bel_sum == 0] = 1.0 과 동일)
    bel_sum = bel_sum.replace(0, 1.0)
    
    # 3. 각 항목을 합계로 나누어 확률 분포(0~1)로 변환
    df_mean[belief_cols] = df_mean[belief_cols].div(bel_sum, axis=0)
    # -------------------------------------------------------------

    # 5. LLM X축 정규화
    df_mean = df_mean.groupby('scenario_id', group_keys=False).apply(normalize_scenario_x)

    # 6. BToM 데이터 로드 및 정규화
    df_btom_raw = load_btom_everystep()
    
    df_btom_mean = None
    if df_btom_raw is not None:
        # BToM도 LLM과 동일하게 Irrational 제외
        df_btom_raw = df_btom_raw[df_btom_raw['scenario_id'].isin(allowed_scenarios)]

        # 🌟 [수정] LLM 데이터(df)에서 시나리오별 truck_presence 매핑 사전을 만들어 BToM에 주입
        presence_map = df[['scenario_id', 'truck_presence']].drop_duplicates().set_index('scenario_id')['truck_presence'].to_dict()
        df_btom_raw['truck_presence'] = df_btom_raw['scenario_id'].map(presence_map)

        # BToM X축 정규화
        df_btom_mean = df_btom_raw.groupby('scenario_id', group_keys=False).apply(normalize_scenario_x)

    # 7. 플롯 시각화 및 저장
    print("🎨 Generating Phase-Normalized Plots...")
    
    # (1) Desire 플롯
    plot_score_figure(
        df_data=df_mean,
        df_btom_data=df_btom_mean,
        score_type="Desire",
        title_prefix=f"[{model_name.upper()} - {condition.capitalize()}] Desire Score Timeline",
        cols=['desire_K', 'desire_L', 'desire_M'],
        colors=['#E63946', '#457B9D', "#ACCB20"],
        labels=['Truck K', 'Truck L', 'Truck M'],
        output_dir=target_dir,
        perfect_count=len(target_subjects)
    )

    # (2) Belief 플롯
    plot_score_figure(
        df_data=df_mean,
        df_btom_data=df_btom_mean,
        score_type="Belief",
        title_prefix=f"[{model_name.upper()} - {condition.capitalize()}] Belief Score Timeline",
        cols=['belief_L', 'belief_M', 'belief_Empty'],
        colors=['#457B9D', "#ACCB20", "#8D8E86"],
        labels=['Truck L', 'Truck M', 'None'],
        output_dir=target_dir,
        perfect_count=len(target_subjects)
    )

    # # 🌟 [임시] BToM 단독 궤적 플롯 생성
    # if df_btom_mean is not None:
    #     print("🎨 Generating BToM-only Baseline Plots...")
        
    #     plot_score_figure(
    #         df_data=df_btom_mean,         # 주인공 자리에 BToM 데이터를 넣음
    #         df_btom_data=None,            # 배경 오버레이는 끔
    #         score_type="Desire",
    #         title_prefix=f"[BToM Baseline] Desire Score Timeline",
    #         cols=['desire_K', 'desire_L', 'desire_M'],
    #         colors=['#E63946', '#457B9D', "#ACCB20"],
    #         labels=['Truck K', 'Truck L', 'Truck M'],
    #         output_dir=target_dir,
    #         perfect_count="All"
    #     )

    #     plot_score_figure(
    #         df_data=df_btom_mean, 
    #         df_btom_data=None, 
    #         score_type="Belief",
    #         title_prefix=f"[BToM Baseline] Belief Score Timeline",
    #         cols=['belief_L', 'belief_M', 'belief_Empty'],
    #         colors=['#457B9D', "#ACCB20", "#8D8E86"],
    #         labels=['Truck L', 'Truck M', 'None'],
    #         output_dir=target_dir,
    #         perfect_count="All"
    #     )

    print("✨ Phase Plot generation complete!")