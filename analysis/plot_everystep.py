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
        if phase_name == 'See G2': return 3
        if phase_name == 'Stop': return 4
        if phase_name == 'Return G1': return 5
        if phase_name == 'Selected': return 6
    elif group_id == 2: # Check-Stay
        if phase_name == 'Pass G1': return 2
        if phase_name == 'See G2': return 3
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
        labels.update({2: 'Pass G1', 3: 'See G2', 4: 'Stop', 5: 'Return G1', 6: 'Selected'})
    elif group_id == 2:
        labels.update({2: 'Pass G1', 3: 'See G2', 4: 'Stop', 5: 'Appr G2', 6: 'Selected'})
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
# 2. 통합 플롯 시각화 함수 (Raw/Delta, All/Extremes 동적 처리)
# =========================================================================
def plot_score_figure(df_data, df_btom_data, df_mae, scope, value_type, score_type, title_prefix, cols, colors, labels, output_dir, perfect_count):
    """
    value_type: 'raw'(LLM 원본) 또는 'delta'(LLM - BToM) -> 이 값에 따라 Y축 규격이 자동 변경됨
    score_type: 'Desire' 또는 'Belief'
    cols: 그릴 컬럼 리스트 (예: ['desire_K', 'desire_L', ...])
    colors: 선 색상 리스트
    labels: 범례 라벨 리스트
    """
    if df_btom_data is None:
        print("❌ Error: Delta(잔차) 그래프를 그리려면 BToM 데이터가 반드시 필요합니다.")
        return

    fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharey=True)
    axes = axes.flatten()
    
    btom_cols = ['scenario_id', 'time_step'] + cols

    # 🌟 [NEW] Delta 계산 및 병합 (Delta가 필요할 때만 수행)
    plot_data = df_data
    merged_with_btom = pd.merge(df_data, df_btom_data[btom_cols], on=['scenario_id', 'time_step'], suffixes=('', '_btom'))
    merged_with_btom = merged_with_btom.sort_values(by=['scenario_id', 'time_step']).reset_index(drop=True)

    if value_type == "delta":
        delta_cols = []
        for col in cols:
            delta_col = f"{col}_delta"
            merged_with_btom[delta_col] = merged_with_btom[col] - merged_with_btom[f"{col}_btom"]
            delta_cols.append(delta_col)
        
        plot_cols = delta_cols # Delta 컬럼을 그림
        plot_data = merged_with_btom
    else:
        plot_cols = cols       # 원본 컬럼(desire_K 등)을 그림
        # Raw 모드일 때는 df_data를 쓰지만, BToM 정답 배경을 그리려면 merged_with_btom이 필요함 (아래에서 처리)

    for i in range(1, 8): # Group 1 ~ 7
        ax = axes[i-1]

        # 데이터를 그룹별로 필터링 (X축 꼬임 방지를 위해 sort 강제)
        group_df = plot_data[plot_data['group_id'] == i].sort_values(by=['scenario_id', 'time_step'])
        
        if group_df.empty:
            continue
            
        # 🌟 [NEW] BToM 정답 데이터를 배경 회색 점선으로 그리기 (Raw 모드일 때만)
        if value_type == "raw":
            group_btom_df = df_btom_data[df_btom_data['group_id'] == i].sort_values(by=['scenario_id', 'time_step'])
            btom_scenarios = group_btom_df['scenario_id'].unique()
            for sc_id in btom_scenarios:
                # Raw 모드일 때 정규화된 X축 값을 가져오기 위해 merged 데이터를 활용
                sc_merged_data = merged_with_btom[(merged_with_btom['scenario_id'] == sc_id) & (merged_with_btom['group_id'] == i)].sort_values('time_step')
                
                # 병합 과정에서 누락된 시나리오가 있을 수 있으므로 체크
                if not sc_merged_data.empty:
                    for col_btom_idx, color in zip(range(len(cols)), colors):
                        col_btom_name = f"{cols[col_btom_idx]}_btom"
                        ax.plot(sc_merged_data['x_norm'], sc_merged_data[col_btom_name], 
                                color=color, linestyle=':', alpha=0.3, linewidth=1, zorder=1)

        # 현재 그릴 시나리오 목록 추출
        scenarios = group_df['scenario_id'].unique()
        
        # [Scope 분기] extremes 모드일 경우 그릴 시나리오 리스트 필터링 (Raw 모드일 때는 all만 하도록 예외 처리)
        if scope == "extremes" and df_mae is not None and value_type == "delta":
            group_name_str = BEHAVIOR_GROUPS.get(i, "")
            group_mae_df = df_mae[df_mae['group_desc'] == group_name_str]
            target_scenarios = group_mae_df['scenario_id'].unique()
            scenarios = [s for s in scenarios if s in target_scenarios]

        # Delta 모드일 때 Y=0 기준선
        if value_type == "delta":
            ax.axhline(y=0, color='black', linewidth=1.5, zorder=3, alpha=0.8)

        # 각 시나리오별로 trajectory를 연하게(alpha=0.3) 겹쳐 그림
        for sc_id in scenarios:
            # 🌟 [수정 포인트 2] 선을 그리기 직전에 다시 한 번 time_step 오름차순으로 꽉 묶어줍니다.
            sc_data = group_df[group_df['scenario_id'] == sc_id].sort_values('time_step')

            # 기본 스타일 세팅 (all 모드, delta/raw 모두)
            l_alpha = 0.5
            l_width = 2
            l_style = '-'
            
            # 🌟 [Scope 분기] Top 3 / Bottom 3 스타일 차별화
            if scope == "extremes" and df_mae is not None and value_type == "delta":
                rank_info = df_mae[df_mae['scenario_id'] == sc_id]['rank_type'].values
                if len(rank_info) > 0:
                    if 'Best' in rank_info[0]:
                        # Top 3 (오차가 적은 애들): 약간 투명한 실선 (배경처럼 얌전하게)
                        l_alpha = 0.4
                        l_width = 1.5
                        l_style = '-'      
                    elif 'Worst' in rank_info[0]:
                        # Bottom 3 (오차가 폭발한 애들): 진하고 굵은 점선 (마커 삭제하여 깔끔하게)
                        l_alpha = 0.9
                        l_width = 2
                        l_style = '--'

            # 기존 LLM 데이터 그리기
            for d_col, color, label in zip(plot_cols, colors, labels):
                ax.plot(sc_data['x_norm'], sc_data[d_col], 
                        color=color, alpha=l_alpha, linewidth=l_width, linestyle=l_style,
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
        # extremes 모드일 때는 표시 개수를 (n=6), all 모드일 때는 전체 개수로 표시
        title_suffix = f"\n(n={len(scenarios)} {'extremes' if (scope=='extremes' and value_type=='delta') else 'all'} scenarios)"
        ax.set_title(f"{group_name}{title_suffix}", fontsize=11, fontweight='bold')
        ax.grid(True, axis='y', linestyle=':', alpha=0.3)

        # 🌟 [NEW] value_type에 따라 Y축 규격 및 라벨 동적 통일
        y_prefix = "Δ " if value_type == "delta" else ""
        y_suffix = "\n(+) Overest / (-) Underest" if value_type == "delta" else ""

        # Y축 규격
        if score_type == "Desire":
            if value_type == "delta":
                # Desire는 차이가 -6 ~ +6 까지 발생할 수 있음
                ax.set_ylim(-6.5, 6.5)
                ax.set_yticks(range(-6, 7, 2))
            else: # value_type == "raw"
                # +1 ~ +7
                ax.set_ylim(0.5, 7.5)
                ax.set_yticks(range(1, 8))

            if i == 1 or i == 5:
                ax.set_ylabel(f"{y_prefix}Desire Rating (1-7){y_suffix}", fontweight='bold')
        
        elif score_type == "Belief":
            if value_type == "delta":
                # Belief는 확률 차이이므로 -1.0 ~ +1.0 까지 발생
                ax.set_ylim(-1.05, 1.05)
                ax.set_yticks(np.linspace(-1, 1, 5))
                ax.set_yticklabels([f"{val:.2f}" for val in np.linspace(-1, 1, 5)])
            else: # value_type == "raw"
                # Belief 0-1, Desire와 칸 수를 맞추기 위해 7분할
                ax.set_ylim(-0.05, 1.05)
                ax.set_yticks(np.linspace(0, 1, 7))
                ax.set_yticklabels([f"{val:.2f}" for val in np.linspace(0, 1, 7)])
            
            if i == 1 or i == 5:
                ax.set_ylabel(f"{y_prefix}Belief Prob (0-1){y_suffix}", fontweight='bold')

        # # Y축 7칸 규격 통일 (Desire 1~7 vs Belief 0~1)
        # if score_type == "Desire":
        #     ax.set_ylim(1, 7)
        #     ax.set_yticks(range(1, 8))
        #     if i == 1 or i == 5:
        #         ax.set_ylabel("Desire Rating (1-7)", fontweight='bold')
        # elif score_type == "Belief":
        #     ax.set_ylim(0, 1.05)
        #     # 0부터 1까지 7개의 눈금(Tick)을 생성하여 Desire와 칸을 동일하게 맞춤
        #     ax.set_yticks(np.linspace(0, 1, 7))
        #     ax.set_yticklabels([f"{val:.2f}" for val in np.linspace(0, 1, 7)])
        #     if i == 1 or i == 5:
        #         ax.set_ylabel("Belief Probability", fontweight='bold')
      
    # 8번째 빈 Subplot 삭제
    fig.delaxes(axes[7])
    
    # 전체 제목 및 레이아웃 설정
    mode_text = "Raw Trajectory" if value_type == "raw" else f"Delta (Scope: {scope.upper()})"
    fig.suptitle(f"{title_prefix} {score_type} {mode_text} for {perfect_count} 'Perfect' Subjects", 
                 fontsize=18, fontweight='bold')
    
    # 커스텀 범례 생성 (트럭 색상 + Presence 조건)
    legend_elements = []
    for color, label in zip(colors, labels):
        legend_elements.append(mlines.Line2D([0], [0], color=color, lw=3, label=label))
    legend_elements.append(mlines.Line2D([], [], color='none', label=' ')) # 공백 추가

    # scope 및 value_type에 따른 동적 범례 추가
    if value_type == "raw":
        # Raw 모드 범례
        legend_elements.append(mlines.Line2D([0], [0], color='gray', linestyle='-', lw=2, alpha=0.5, label=f'{perfect_count} Subjects Avg'))
        legend_elements.append(mlines.Line2D([0], [0], color='black', linestyle=':', lw=1, alpha=0.3, label='BToM Baseline (GT)'))
    else:
        # Delta 모드 범례 (all vs extremes)
        if scope == "extremes":
            legend_elements.append(mlines.Line2D([0], [0], color='gray', linestyle='-', lw=1.5, alpha=0.6, label='Top 3 (Best Fit)'))
            legend_elements.append(mlines.Line2D([0], [0], color='black', linestyle='--', lw=2.5, alpha=0.9, label='Bottom 3 (Worst Fit)'))

    fig.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(0.95, 0.1), fontsize=11, frameon=True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 🌟 [NEW] 파일 저장명 차별화 (raw 모드와 delta 모드)
    fn_prefix = "raw_plot" if value_type == "raw" else f"delta_plot"
    # raw 모드일 때는 scope 표시 안 함 (어차피 all이니까)
    fn_scope = "" if value_type == "raw" else f"_{scope}"
        # [추가] BToM만 그렸을 때 파일명 충돌 방지 (Optional)
    prefix = "btom_baseline_" if df_btom_data is None and "Baseline" in title_prefix else ""

    # 저장
    save_path = os.path.join(output_dir, f"{prefix}{fn_prefix}_{score_type.lower()}_phase{fn_scope}.png")
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved {score_type} plot (Type:{value_type}, Scope:{scope}) to {save_path}")
    plt.show()

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

    # 3. 비합리적 시나리오(Irrational) 포함 필터링
    allowed_groups = get_group_indices(include_irrational=True)
    allowed_scenarios = [sc for group in allowed_groups for sc in group]
    df = df[df['scenario_id'].isin(allowed_scenarios)]

    if df.empty:
        print("❌ Error: 필터링 후 남은 데이터가 없습니다.")
        return

    # 4. 피험자 간 평균 계산
    SCORE_COLS = ['desire_K', 'desire_L', 'desire_M', 'belief_L', 'belief_M', 'belief_Empty']
    
    df_mean = df.groupby(['scenario_id', 'group_id', 'time_step', 'phase'])[SCORE_COLS].mean().reset_index()

    # -------------------------------------------------------------
    # 예전 코드와 100% 동일한 Belief 정규화 로직 적용
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

        # BToM X축 정규화
        df_btom_mean = df_btom_raw.groupby('scenario_id', group_keys=False).apply(normalize_scenario_x)

    # MAE Features 파일 로드 (Extremes 모드 전용)
    mae_path = os.path.join(target_dir, "scenario_mae_features.csv")
    df_mae = pd.read_csv(mae_path) if os.path.exists(mae_path) else None

    # =========================================================================
    # 🌟 [메인 루프 설정] 유저 요구사항 반영: Raw, Delta All, Delta Extremes를 한 번에 생성
    # =========================================================================
    # (Scope, ValueType) 조합 리스트
    plot_tasks = [
        ("all", "raw"),         # 1. LLM 원본 전체 조망 (Raw 데이터 Y축 1~7/0~1)
        ("all", "delta"),       # 2. 잔차 전체 조망 (Delta 데이터 Y축 -6~+6 / -1~+1)
        ("extremes", "delta"),  # 3. 잔차 정밀 진단 (Top/Bottom 3 강조)
    ]
    
    print("\n🎨 Generating 3 types of everystep plots per score type (Raw, Delta All, Delta Extremes)...")
    
    # 루프를 돌며 조합별 플롯 생성
    for scope_task, value_type_task in plot_tasks:
        
        # extremes 모드인데 MAE 파일이 없으면 안전하게 스킵
        if scope_task == "extremes" and df_mae is None:
            print(f"  ⏩ [Skip] MAE features csv not found. Skipping {scope_task}/{value_type_task} plot.")
            continue

        print(f"\n   -> Drawing Phase Plot: Scope={scope_task.upper()}, Value={value_type_task.upper()}...")

        # (1) Desire 플롯 호출
        plot_score_figure(
            df_data=df_mean,
            df_btom_data=df_btom_raw,
            df_mae=df_mae,
            scope=scope_task,
            value_type=value_type_task,
            score_type="Desire",
            title_prefix=f"[{model_name.upper()} - {condition.capitalize()}]",
            cols=['desire_K', 'desire_L', 'desire_M'],
            colors=['#E63946', '#457B9D', "#ACCB20"],
            labels=['Truck K', 'Truck L', 'Truck M'],
            output_dir=target_dir,
            perfect_count=len(target_subjects)
        )

        # (2) Belief 플롯 호출
        plot_score_figure(
            df_data=df_mean,
            df_btom_data=df_btom_raw,
            df_mae=df_mae,
            scope=scope_task,
            value_type=value_type_task,
            score_type="Belief",
            title_prefix=f"[{model_name.upper()} - {condition.capitalize()}]",
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

    print("\n✨ All 6 everystep plots (Raw & Delta combinations) generation complete!")