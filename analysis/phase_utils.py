"""
Stepwise(Every-step / Prefix-step) 분석에서 함께 쓰는 phase 정규화와 집계 함수.
원래 plot_everystep.py(phase)와 plot_current_vs_initial_belief.py(집계)에 있던 것을 옮겨 옴.
이 둘은 analysis/ver0_everystep/으로 보관되었고, 현재 스크립트는 이 모듈에서 가져옴.
"""
import numpy as np
import pandas as pd


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


def keep_complete(df):
    """모든 time step이 있는 (subject, scenario)만 남김."""
    from src.dataset import df_btom
    expected = df_btom.groupby("scenario_id").size()
    got = df.groupby(["subject_id", "scenario_id"]).size().reset_index(name="n")
    got["exp"] = got["scenario_id"].map(expected)
    ok = got[got["n"] == got["exp"]][["subject_id", "scenario_id"]]
    dropped = got[got["n"] != got["exp"]]
    for r in dropped.itertuples():
        print(f"  ⚠️ drop subject {r.subject_id} scenario {r.scenario_id}: {r.n}/{r.exp} steps")
    return df.merge(ok, on=["subject_id", "scenario_id"])


def to_prob(block):
    """1~7 척도 -> (x-1)/sum, 논문 bel_inf_mean_norm과 동일한 처리."""
    shifted = np.maximum(block - 1, 0)
    total = shifted.sum(axis=1).replace(0, 1.0)
    return shifted.div(total, axis=0)


def phase_means(sub, col):
    """시나리오별 phase 평균 -> 시나리오 간 평균. x = 구간 중앙."""
    per_sc = sub.groupby(["scenario_id", "phase_idx"])[col].mean().reset_index()
    out = per_sc.groupby("phase_idx")[col].mean()
    return out.index.values + 0.5, out.values


