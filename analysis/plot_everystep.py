import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

# 현재 스크립트(analysis 폴더)의 상위 경로를 파이썬 탐색 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) # 상위 폴더 (프로젝트 루트)

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.prepare_everystep import load_reference_everystep
from src.config import BASE_RESULTS_DIR, BEHAVIOR_GROUPS, get_group_indices

# =========================================================================
# 🌟 [NEW] VPA 시각화를 위한 색상 및 설정
# =========================================================================
VPA_COLOR_MAP = {
    'btom': '#E63946',             
    'truebelief': '#F5A623',       
    'nocost': "#DC91EB",           
    'motionheuristic': "#B8969A",
    'hindsight': "#747ED2"   
}
VPA_REFS_DESIRE = ['btom', 'truebelief', 'nocost', 'motionheuristic']
VPA_REFS_BELIEF = ['btom', 'truebelief', 'nocost', 'motionheuristic', 'hindsight']

# =========================================================================
# 1. Phase-Normalization (위상 정규화) 로직
# =========================================================================
def get_phase_index(group_id, phase_name):
    """
    각 그룹별로 Phase가 발생해야 하는 논리적 순서(Integer Bin)를 매핑합니다.
    서로 다른 시나리오라도 같은 Phase면 같은 X축 구간(예: 1.0 ~ 2.0)에 놓이게 됩니다.
    💡 [수정] Prior를 제거하고 Start를 0으로 설정하여 전체 인덱스를 1씩 앞당김
    """
    # if phase_name == 'Prior': return 0
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
    
    return 9 # Unknown

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
# 2. 통합 플롯 시각화 함수 (Raw/Delta, All/Extremes, VPA 오버레이)
# =========================================================================
def plot_score_figure(df_data, df_btom_data, df_extremes, df_normal_stats, scope, value_type, indicator, 
                      score_type, title_prefix, cols, colors, labels, output_dir, perfect_count, vpa_data=None):
    """
    value_type: 'raw'(LLM 원본) 또는 'delta'(LLM - BToM) -> 이 값에 따라 Y축 규격이 자동 변경됨
    indicator: 'Global_Bias', 'TrueBelief' 등 어느 지표에 대한 Extremes를 그릴지 식별
    score_type: 'Desire' 또는 'Belief'
    cols: 그릴 컬럼 리스트 (예: ['desire_K', 'desire_L', ...])
    colors: 선 색상 리스트
    labels: 범례 라벨 리스트
    df_normal_stats: Normal 모드의 평균(mean)과 표준편차(std)가 담긴 딕셔너리 (raw 모드에서만 사용)
    """
    if df_btom_data is None:
        print("❌ Error: Delta(잔차) 그래프를 그리려면 BToM 데이터가 반드시 필요합니다.")
        return

    fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharey=True)
    plt.subplots_adjust(hspace=0.3)

    axes = axes.flatten()
    
    btom_cols = ['scenario_id', 'time_step'] + cols

    # Delta 계산 및 병합 (Delta가 필요할 때만 수행)
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

    for i in range(1, 8): # Group 1 ~ 7
        ax = axes[i-1]

        # 데이터를 그룹별로 필터링 (X축 꼬임 방지를 위해 sort 강제)
        group_df = plot_data[plot_data['group_id'] == i].sort_values(by=['scenario_id', 'time_step'])
        
        if group_df.empty:
            continue
            
        # 🌟 BToM 정답 데이터를 배경 회색 점선으로 그리기 (Raw 모드일 때만)
        if value_type == "raw":
            group_btom_df = df_btom_data[df_btom_data['group_id'] == i].sort_values(by=['scenario_id', 'time_step'])
            btom_scenarios = group_btom_df['scenario_id'].unique()
            for sc_id in btom_scenarios:

                # # Raw 모드일 때 정규화된 X축 값을 가져오기 위해 merged 데이터를 활용
                # sc_merged_data = merged_with_btom[(merged_with_btom['scenario_id'] == sc_id) & (merged_with_btom['group_id'] == i)].sort_values('time_step')
                
                # merged_with_btom 대신 BToM의 원본 정규화 데이터를 직접 사용
                sc_btom = group_btom_df[group_btom_df['scenario_id'] == sc_id]

                # 병합 과정에서 누락된 시나리오가 있을 수 있으므로 체크
                if not sc_btom.empty:
                    for col_btom_idx, color in zip(range(len(cols)), colors):
                        pass
                        # ax.plot(sc_btom['x_norm'], sc_btom[cols[col_btom_idx]], 
                        #         color=color, linestyle=':', alpha=0.5, linewidth=1, zorder=1)

        # 현재 그릴 시나리오 목록 추출
        scenarios = group_df['scenario_id'].unique()
        
        # 🌟 [Scope 분기] extremes 모드일 경우: 해당 indicator에 존재하는 시나리오만 남김
        if scope == "extremes" and df_extremes is not None and value_type == "delta" and indicator is not None:
            ind_df = df_extremes[df_extremes['indicator_name'] == indicator]
            target_scenarios = ind_df['scenario_id'].unique()
            scenarios = [s for s in scenarios if s in target_scenarios]

        # Delta 모드일 때 Y=0 기준선
        if value_type == "delta":
            ax.axhline(y=0, color='black', linewidth=1.5, zorder=3, alpha=0.8)

        # X축 위치 산정을 위해 그룹 내 전체 시나리오의 최대/최소값을 먼저 구함
        max_x_in_group = group_df['x_norm'].max()
        # min_x_in_group = group_df['x_norm'].min()

        # 각 시나리오별로 trajectory를 연하게(alpha=0.3) 겹쳐 그림
        for sc_id in scenarios:
            # 선을 그리기 직전에 다시 한 번 time_step 오름차순으로 꽉 묶어줍니다.
            sc_data = group_df[group_df['scenario_id'] == sc_id].sort_values('time_step')

            # 기본 스타일 세팅 (all 모드, delta/raw 모두)
            l_alpha = 0.5
            l_width = 2
            l_style = '-'

            # 💡 [NEW] 비합리적 경로(Irrational Paths) 스타일 오버라이드
            irrational_scenarios = [11, 12, 22, 71, 72]
            if sc_id in irrational_scenarios:
                l_style = '-.'  # 쇄선 (Dash-dot) 사용
                l_alpha = 0.4   # 일반 경로보다 살짝 더 투명하게 하여 배경에 스며들게 함
                l_width = 1.5   # 굵기도 살짝 줄임
            
            # [Scope 분기] 해당 indicator에서의 Rank Type(Best/Worst)에 따른 스타일 차별화
            if scope == "extremes" and df_extremes is not None and value_type == "delta" and indicator is not None:
                # 조건에 맞는 해당 시나리오의 랭크 추출
                rank_info = df_extremes[(df_extremes['scenario_id'] == sc_id) & (df_extremes['indicator_name'] == indicator)]['rank_type'].values
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

            # 🌟 1. Everystep 궤적(선) 그리기
            for d_col, color, label in zip(plot_cols, colors, labels):
                ax.plot(sc_data['x_norm'], sc_data[d_col], 
                        color=color, alpha=l_alpha, linewidth=l_width, linestyle=l_style,
                        label=label if sc_id == scenarios[0] else "") # 범례는 한 번만
        
        # 🌟 2. Normal 모드 응답 포인트 추가 (Raw 모드일 때만, 그룹 통합 통계)
        if value_type == "raw" and df_normal_stats is not None:
            # i는 현재 Group ID (1~7)
            # 논리적 X축 위치 산정 (Desire = 그룹 내 마지막 스텝, Belief = 그룹 내 첫 스텝)
            x_target = max_x_in_group
            
            # 딕셔너리에서 그룹별 통계 가져오기
            for c_idx, color in enumerate(colors):
                if score_type == "Desire":
                    y_mean = df_normal_stats['des_mean'][c_idx, i-1] # 0-based index for groups
                    y_err = df_normal_stats['des_std'][c_idx, i-1]
                else:
                    y_mean = df_normal_stats['bel_mean'][c_idx, i-1]
                    y_err = df_normal_stats['bel_std'][c_idx, i-1]
                
                # 값이 유효한지 확인 후 에러바 출력
                if not np.isnan(y_mean):
                    ax.errorbar(x_target, y_mean, yerr=y_err, fmt='D', color=color, 
                                markersize=8, capsize=4, elinewidth=2, alpha=1.0, 
                                markeredgecolor='white', markeredgewidth=1, zorder=6)
        
        # =========================================================
        # 🌟 [수정] X축 눈금 라벨을 더 아래로 밀어냅니다 (pad 조절)
        # =========================================================
        # X축 꾸미기 (점선 및 라벨)
        phase_labels = get_group_phase_labels(i)

        # 1. Phase 경계선(회색 점선)은 원래대로 정수 위치(구간의 시작점)에 그립니다.
        keys_list = list(phase_labels.keys())
        for x_val in keys_list:
            ax.axvline(x=x_val, color='gray', linestyle='--', alpha=0.3)

        # 2. X축 눈금(Tick)의 위치를 각 구간의 정중앙(+0.5)으로 옮깁니다.
        tick_positions = [x + 0.6 for x in keys_list]
        ax.set_xticks(tick_positions)

        # 🌟 [수정] VPA 데이터가 있어서 리본을 그릴 때만 여백을 18로 늘리고, 아닐 때는 기본값(4) 유지
        has_ribbon = (value_type == "raw" and vpa_data is not None)
        current_pad = 18 if has_ribbon else 4

        # 3. 라벨을 출력하고, 글씨만 공중에 예쁘게 떠 있도록 튀어나온 눈금선(tick mark)을 숨깁니다.
        # pad=15~20을 주어 텍스트를 밑으로 내리고 Ribbon이 들어갈 틈을 확보
        ax.tick_params(axis='x', length=0, pad=current_pad) 
        ax.set_xticklabels(list(phase_labels.values()), rotation=45, ha='right', fontsize=9)

        # =====================================================================
        # 🌟 VPA Ribbon을 플롯 바깥(X축 아래)에 배치
        # =====================================================================
        if value_type == "raw" and vpa_data is not None:
            group_vpa = vpa_data.get(i, {})
            
            for phase_idx in keys_list:
                phase_vpa = group_vpa.get(phase_idx, {}).get(score_type, {}).get('unique', {})
                if not phase_vpa: continue
                
                # 1등 모델 찾기
                winner = max(phase_vpa, key=phase_vpa.get)
                max_r2 = phase_vpa[winner]
                
                if max_r2 > 0:
                    # 1. 하단 Ribbon (Alpha 적용)
                    # 설명력(0.0~0.5)을 Alpha(0.2~0.9)로 선형 변환하여 그라데이션 효과
                    alpha_val = min(0.9, max(0.2, max_r2 * 2.0))

                    # 💡 clip_on=False를 통해 Axes(테두리) 바깥쪽(y=-0.08 위치)에 사각형을 그림
                    rect = mpatches.Rectangle(
                        xy=(phase_idx, -0.07),       # X는 데이터 좌표(phase_idx), Y는 비율(-7% 위치)
                        width=1.0,                   # 너비는 1 구간
                        height=0.04,                 # 띠의 두께
                        transform=ax.get_xaxis_transform(),
                        color=VPA_COLOR_MAP[winner],
                        alpha=alpha_val,
                        clip_on=False                # 🌟 필수: 플롯 영역 밖으로 나가도 잘리지 않게 함
                    )
                    ax.add_patch(rect)
                
                # # 2. 상단 미니 막대그래프 (Inset Bar Chart)
                # # 데이터 좌표계(X축)와 Axes 좌표계(Y축 비율)를 섞어 씀 (y=1.05부터 위로 25% 크기)
                # ax_inset = ax.inset_axes([phase_idx + 0.1, 1.02, 0.8, 0.3], transform=ax.get_xaxis_transform())
                
                # bar_width = 0.8 / len(vpa_refs)
                # x_positions = np.arange(len(vpa_refs)) * bar_width
                
                # for r_idx, ref in enumerate(vpa_refs):
                #     val = phase_vpa.get(ref, 0)
                #     is_winner = (ref == winner and val > 0)
                    
                #     bc = VPA_COLOR_MAP[ref]
                #     ec = 'black' if is_winner else 'white'
                #     lw = 1.0 if is_winner else 0.5
                #     al = 1.0 if is_winner else 0.5
                    
                #     ax_inset.bar(x_positions[r_idx], val, width=bar_width*0.85, 
                #                  color=bc, edgecolor=ec, linewidth=lw, alpha=al)
                
                # # 미니 차트 규격 통일 및 외곽선 제거
                # ax_inset.set_ylim(0, 0.5)
                # ax_inset.axis('off')

        # 제목 및 Y축 설정
        group_name = BEHAVIOR_GROUPS.get(i, f"Group {i}")
        # extremes 모드일 때는 표시 개수를 (n=6), all 모드일 때는 전체 개수로 표시
        title_suffix = f"\n(n={len(scenarios)} {'extremes' if (scope=='extremes' and value_type=='delta') else 'all'} scenarios)"
        ax.set_title(f"{group_name}{title_suffix}", fontsize=11, fontweight='bold')
        ax.grid(True, axis='y', linestyle=':', alpha=0.3)

        # value_type에 따라 Y축 규격 및 라벨 동적 통일
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
      
    # 8번째 빈 Subplot 삭제
    fig.delaxes(axes[7])
    
    # 전체 제목 설정
    vpa_title = " w/ VPA Overlays" if (vpa_data is not None and value_type == "raw") else ""
    mode_text = f"Raw Trajectory{vpa_title}" if value_type == "raw" else f"Delta (Scope: {scope.upper()}{' - ' + indicator if indicator else ''})"
    fig.suptitle(f"{title_prefix} {score_type} {mode_text} for {perfect_count} 'Perfect' Subjects", 
                 fontsize=18, fontweight='bold', y=0.98 if vpa_data is not None else 0.95)
    
    # 커스텀 범례 생성 (트럭 색상 + Presence 조건)
    legend_elements = []
    for color, label in zip(colors, labels):
        legend_elements.append(mlines.Line2D([0], [0], color=color, lw=3, label=label))
    legend_elements.append(mlines.Line2D([], [], color='none', label=' ')) # 공백 추가

    # scope 및 value_type에 따른 동적 범례 추가
    if value_type == "raw":
        # Raw 모드 범례
        legend_elements.append(mlines.Line2D([0], [0], color='gray', linestyle='-', lw=2, alpha=0.5, label=f'{perfect_count} Subjects Avg'))
        legend_elements.append(mlines.Line2D([0], [0], color='gray', linestyle='-.', lw=1.5, alpha=0.4, label='Irrational Paths'))
        # legend_elements.append(mlines.Line2D([0], [0], color='black', linestyle=':', lw=1, alpha=0.3, label='BToM Baseline'))

        # Normal Mode 범례 추가
        if df_normal_stats is not None:
            legend_elements.append(mlines.Line2D([0], [0], marker='D', color='gray', linestyle='None', 
                                                 markersize=6, alpha=0.8, label='End-step (Mean±Std)'))

        # 💡 VPA가 활성화된 경우 VPA 범례 및 농도 스케일 추가
        if vpa_data is not None:
            legend_elements.append(mlines.Line2D([], [], color='none', label='--- VPA Models ---'))
            vpa_refs = VPA_REFS_DESIRE if score_type == "Desire" else VPA_REFS_BELIEF
            for ref in vpa_refs:
                legend_elements.append(mpatches.Patch(color=VPA_COLOR_MAP[ref], label=ref.capitalize()))
        
    else:
        # Delta 모드 범례 (all vs extremes)
        if scope == "extremes":
            legend_elements.append(mlines.Line2D([0], [0], color='gray', linestyle='-', lw=1.5, alpha=0.6, label='Best N (Low Error)'))
            legend_elements.append(mlines.Line2D([0], [0], color='black', linestyle='--', lw=2.5, alpha=0.9, label='Worst N (High Error)'))

    # 레이아웃 마감
    fig.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(0.95, 0.1), fontsize=11, frameon=True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 파일 저장명 차별화 (raw 모드와 delta 모드)
    fn_prefix = "raw_vpa_plot" if (vpa_data is not None and value_type == "raw") else ("raw_plot" if value_type == "raw" else f"delta_plot")
    # raw 모드일 때는 scope 표시 안 함 (어차피 all이니까)
    fn_scope = "" if value_type == "raw" else (f"_{scope}_{indicator}" if indicator else f"_{scope}")
    # BToM만 그렸을 때 파일명 충돌 방지 (Optional)
    prefix = "btom_baseline_" if df_btom_data is None and "Baseline" in title_prefix else ""

    # 저장
    save_path = os.path.join(output_dir, f"{prefix}{fn_prefix}_{score_type.lower()}_phase{fn_scope}_nobackground.png")
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved {score_type} plot (Type:{value_type}, Ind:{indicator}, Scope:{scope}) to {save_path}")
    plt.close()

# =========================================================================
# 3. 메인 분석 함수 (run_analysis.py에서 호출할 엔트리포인트)
# =========================================================================
def run_plot_everystep(model_name, condition, target_subjects, output_dir=None, enable_vpa=False):
    """
    Args:
        model_name: 모델 이름 (예: gpt-4o)
        condition: 실험 조건 (예: vanilla, oneshot)
        target_subjects: 완벽하게 78개 시나리오를 통과한 피험자 리스트 (인지모델은 [0])
        output_dir: 저장할 폴더 경로 (run_analysis에서 제공)
    """
    # 1. 저장 디렉토리 동적 확인
    if output_dir is None:
        output_dir = os.path.join(BASE_RESULTS_DIR, model_name, condition, "everystep")
        
    allowed_groups = get_group_indices(include_irrational=True)
    allowed_scenarios = [sc for group in allowed_groups for sc in group]
    
    ref_models = ["btom", "truebelief", "nocost", "motionheuristic", "hindsight"]
    SCORE_COLS = ['desire_K', 'desire_L', 'desire_M', 'belief_L', 'belief_M', 'belief_Empty']

    # -------------------------------------------------------------
    # 🌟 [A] VPA 데이터 동적 로드 (enable_vpa 활성화 시)
    # -------------------------------------------------------------
    vpa_data_dict = None
    if enable_vpa and model_name.lower() not in ref_models:
        vpa_pkl_path = os.path.join(parent_dir, "results", f"everystep_cumulative_vpa_results_{condition}.pkl")
        if os.path.exists(vpa_pkl_path):
            print(f"📥 Loading VPA overlays from: {os.path.basename(vpa_pkl_path)}")
            master_vpa = pd.read_pickle(vpa_pkl_path)
            if model_name in master_vpa:
                vpa_data_dict = master_vpa[model_name].get('Cumulative', {})
            else:
                print(f"⚠️ Warning: Model '{model_name}' not found in VPA data.")
        else:
            print(f"⚠️ Warning: VPA data file not found ({vpa_pkl_path}). Overlays will be skipped.")

    # -------------------------------------------------------------
    # 🌟 [B] Target 데이터 로드 (인지 모델 vs LLM 동적 분기)
    # -------------------------------------------------------------
    if model_name.lower() in ref_models:
        # 인지 모델일 경우: load_reference_everystep을 이용해 즉시 데이터 확보
        print(f"📥 Loading Target Everystep data for {model_name.upper()}...")
        df_mean = load_reference_everystep(model_name.lower())
        
        if df_mean is None:
            print(f"❌ Error: {model_name.upper()}의 타겟 데이터를 불러올 수 없습니다.")
            return
            
        df_mean = df_mean[df_mean['scenario_id'].isin(allowed_scenarios)]
        df_mean = df_mean.groupby('scenario_id', group_keys=False).apply(normalize_scenario_x)
        
        title_prefix = f"[{model_name.upper()}]"
        
    else:
        # LLM일 경우: CSV에서 읽어서 평균 및 정규화
        data_path = os.path.join(output_dir, "everystep_valid_only.csv")
        
        if not os.path.exists(data_path):
            print(f"❌ Error: Valid-only data not found at {data_path}")
            return
            
        print(f"📥 Loading Target Everystep CSV data for {model_name.upper()}...")
        df = pd.read_csv(data_path)

        df = df[df['subject_id'].isin(target_subjects)]
        df = df[df['scenario_id'].isin(allowed_scenarios)]

        df_mean = df.groupby(['scenario_id', 'group_id', 'time_step', 'phase'])[SCORE_COLS].mean().reset_index()

        # Belief 정규화
        belief_cols = ['belief_L', 'belief_M', 'belief_Empty']

        # 1. 1~7점 척도를 0~6점 척도로 Shift (음수 방지)
        df_mean_shifted = df_mean[belief_cols] - 1
        df_mean_shifted = np.maximum(df_mean_shifted, 0)

        # 2. 각 행(row)별로 L, M, Empty 평균값의 합을 구하고, 0으로 나누는 에러 방지
        bel_sum = df_mean_shifted.sum(axis=1).replace(0, 1.0)

        # 3. 각 항목을 합계로 나누어 확률 분포(0~1)로 변환
        df_mean[belief_cols] = df_mean_shifted.div(bel_sum, axis=0)

        # X축 정규화
        df_mean = df_mean.groupby('scenario_id', group_keys=False).apply(normalize_scenario_x)
        
        title_prefix = f"[{model_name.upper()} - {condition.capitalize()}]"

    # -------------------------------------------------------------
    # 🌟 [C] Normal 모드 데이터 로드 및 통계 연산 (Group 단위)
    # -------------------------------------------------------------
    df_normal_stats = None
    if model_name.lower() not in ref_models:
        normal_dir = os.path.join(BASE_RESULTS_DIR, model_name, condition)
        normal_pkl_path = os.path.join(normal_dir, "model_data.pkl")
        
        if os.path.exists(normal_pkl_path):
            print(f"📥 Loading Normal Mode data from {normal_pkl_path} for comparison overlay...")
            normal_data_dict = pd.read_pickle(normal_pkl_path)
            
            # 1. 3D 배열 가져오기 (Rating, Scenario, Subject)
            des_inf_3d = normal_data_dict.get('des_inf')
            bel_inf_3d = normal_data_dict.get('bel_inf')
            
            if des_inf_3d is not None and bel_inf_3d is not None:
                # 2. 우등생(Target Subjects) 필터링
                # CSV 파일명 순서와 매칭되므로 subject_id - 1 을 인덱스로 사용
                # 범위를 벗어나는 에러를 방지하기 위해 유효한 인덱스만 추출
                target_idxs = [s - 1 for s in target_subjects if (s - 1) < des_inf_3d.shape[2]]
                
                des_inf_filtered = des_inf_3d[:, :, target_idxs]
                bel_inf_filtered = bel_inf_3d[:, :, target_idxs]
                
                # 3. 우등생 피험자들에 대한 "시나리오별 평균"을 먼저 구함
                des_scen_mean = np.nanmean(des_inf_filtered, axis=2) # (3, 78)
                
                bel_shifted = np.maximum(bel_inf_filtered - 1, 0)
                bel_sums = np.nansum(bel_shifted, axis=0, keepdims=True)
                bel_sums[bel_sums == 0] = 1.0
                bel_norm_filtered = bel_shifted / bel_sums
                bel_scen_mean = np.nanmean(bel_norm_filtered, axis=2) # (3, 78)
                
                # 4. 💡 "그룹 단위" 평균 및 표준편차 산출
                n_groups = 7
                # 그룹 내 시나리오 인덱스 매핑 (0-based)
                group_inds = [[sid - 1 for sid in group] for group in get_group_indices(include_irrational=False)]
                
                des_group_mean = np.zeros((3, n_groups))
                des_group_std = np.zeros((3, n_groups))
                bel_group_mean = np.zeros((3, n_groups))
                bel_group_std = np.zeros((3, n_groups))
                
                for gi in range(n_groups):
                    g_idxs = group_inds[gi]
                    # 해당 그룹에 속한 시나리오들의 평균의 '평균'과 '표준편차'
                    des_group_mean[:, gi] = np.nanmean(des_scen_mean[:, g_idxs], axis=1)
                    des_group_std[:, gi] = np.nanstd(des_scen_mean[:, g_idxs], axis=1, ddof=1)
                    
                    bel_group_mean[:, gi] = np.nanmean(bel_scen_mean[:, g_idxs], axis=1)
                    bel_group_std[:, gi] = np.nanstd(bel_scen_mean[:, g_idxs], axis=1, ddof=1)
                
                # 5. 결과를 딕셔너리로 묶어서 플롯 함수에 전달
                df_normal_stats = {
                    'des_mean': des_group_mean, 'des_std': des_group_std,
                    'bel_mean': bel_group_mean, 'bel_std': bel_group_std
                }
            else:
                print("⚠️ Notice: Normal PKL 데이터 내에 3D 배열이 존재하지 않습니다.")
        else:
            print(f"⚠️ Notice: Normal mode data not found at {normal_pkl_path}. Overlay will be skipped.")

    # -------------------------------------------------------------
    # 🌟 [D] BToM 데이터 로드 (배경 점선 / Delta 베이스라인 용도)
    # -------------------------------------------------------------
    print("📥 Loading BToM Everystep data for baseline reference...")
    df_btom_raw = load_reference_everystep('btom')
    
    if df_btom_raw is not None:
        df_btom_raw = df_btom_raw[df_btom_raw['scenario_id'].isin(allowed_scenarios)]
        
        # 💡 BToM 데이터도 X축(Phase) 정규화를 수행
        df_btom_raw = df_btom_raw.groupby('scenario_id', group_keys=False).apply(normalize_scenario_x)

        # 만약 Target 자체가 BToM이라면, Delta(잔차)는 수학적으로 완벽한 0이 됨
        if model_name.lower() == 'btom':
            print("💡 Notice: Target이 BToM이므로 Delta 모드 시 모든 값이 0으로 그려집니다.")

    # -------------------------------------------------------------
    # 🌟 [E] 동적으로 생성된 Extremes 파일 스캔 및 로드
    # -------------------------------------------------------------
    # (인지 모델을 실행할 때는 extremes 파일이 없으므로 자동 스킵됨)
    extremes_pattern = os.path.join(output_dir, "extremes_top*_summary.csv")
    extremes_files = glob.glob(extremes_pattern)
    df_extremes = None
    
    if extremes_files:
        # 파일이 여러 개일 경우 가장 최근에 수정된 파일 선택
        latest_file = max(extremes_files, key=os.path.getctime)
        print(f"📥 Found Extremes summary! Loading from: {os.path.basename(latest_file)}")
        df_extremes = pd.read_csv(latest_file)
    else:
        print("⚠️ Warning: No extremes summary found. Extremes plots will be skipped.")

    # =========================================================================
    # 🌟 [메인 루프 설정] Raw, Delta All, Delta Extremes(지표(Indicator)별)를 한 번에 생성
    # =========================================================================
    # (Scope, ValueType) 조합 리스트
    plot_tasks = [
        ("all", "raw", None),         # 1. LLM 원본 전체 조망 (Raw 데이터 Y축 1~7/0~1)
        ("all", "delta", None),       # 2. 잔차 전체 조망 (Delta 데이터 Y축 -6~+6 / -1~+1)
        # ("extremes", "delta"),      # 3. 잔차 정밀 진단 (Top/Bottom 3 강조)
    ]
    
    # df_extremes 데이터가 있다면 안에 들어있는 각 평가 지표별로 Task 추가
    if df_extremes is not None:
        indicators = df_extremes['indicator_name'].unique()
        for ind in indicators:
            plot_tasks.append(("extremes", "delta", ind))

    print(f"\n🎨 Generating {len(plot_tasks)} sets of everystep plots per score type...")
    
    # 루프를 돌며 조합별 플롯 생성
    for scope_task, value_type_task, indicator_task in plot_tasks:
        
        ind_text = f"[{indicator_task}] " if indicator_task else ""
        print(f"\n   -> Drawing Phase Plot: Scope={scope_task.upper()}, Value={value_type_task.upper()} {ind_text}...")
        
        # (1) Desire 플롯 호출
        plot_score_figure(
            df_data=df_mean,
            df_btom_data=df_btom_raw,
            df_extremes=df_extremes,
            df_normal_stats=df_normal_stats,
            scope=scope_task,
            value_type=value_type_task,
            indicator=indicator_task,
            score_type="Desire",
            title_prefix=title_prefix,
            cols=['desire_K', 'desire_L', 'desire_M'],
            colors=['#E63946', '#457B9D', "#ACCB20"],
            labels=['Truck K', 'Truck L', 'Truck M'],
            output_dir=output_dir,
            perfect_count=len(target_subjects),
            vpa_data=vpa_data_dict
        )

        # (2) Belief 플롯 호출
        plot_score_figure(
            df_data=df_mean,
            df_btom_data=df_btom_raw,
            df_extremes=df_extremes,
            df_normal_stats=df_normal_stats,
            scope=scope_task,
            value_type=value_type_task,
            indicator=indicator_task,
            score_type="Belief",
            title_prefix=title_prefix,
            cols=['belief_L', 'belief_M', 'belief_Empty'],
            colors=['#457B9D', "#ACCB20", "#8D8E86"],
            labels=['Truck L', 'Truck M', 'None'],
            output_dir=output_dir,
            perfect_count=len(target_subjects),
            vpa_data=vpa_data_dict
        )

    print("\n✨ All everystep plots generation complete!")