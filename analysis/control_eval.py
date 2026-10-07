import pandas as pd

def ground_truth(df_scenario):
    """generate_scenario_prompt(mode='control')과 동일한 로직으로 정답 산출."""
    t_pos = int(df_scenario['time_step'].iloc[len(df_scenario) // 2])
    vis2 = df_scenario[df_scenario['visible_goal2'] == 1]
    t_vis = int(vis2['time_step'].min()) if not vis2.empty else None
    t_occ = max(1, (t_vis - 1)) if t_vis else int(df_scenario['time_step'].max())

    r_pos = df_scenario[df_scenario['time_step'] == t_pos].iloc[0]
    r_occ = df_scenario[df_scenario['time_step'] == t_occ].iloc[0]
    r_last = df_scenario.iloc[-1]

    # Q4: final location
    if (r_last['agent_x'] == r_last['goal1_x']) and (r_last['agent_y'] == r_last['goal1_y']):
        final = "Spot 1"
    elif (r_last['agent_x'] == r_last['goal2_x']) and (r_last['agent_y'] == r_last['goal2_y']):
        final = "Spot 2"
    else:
        final = "Neither"

    return {
        'q1_position': {'x': int(r_pos['agent_x']), 'y': int(r_pos['agent_y'])},
        'q2_spot2_visible': 'yes' if int(r_occ['visible_goal2']) == 1 else 'no',
        'q3_first_observed_step': t_vis if t_vis else 0,
        'q4_final_location': final,
    }

def score(pred, gt):
    """문항별 정오 판정."""
    out = {}
    try:
        out['q1'] = (int(pred['q1_position']['x']) == gt['q1_position']['x'] and
                     int(pred['q1_position']['y']) == gt['q1_position']['y'])
    except Exception:
        out['q1'] = False
    out['q2'] = str(pred.get('q2_spot2_visible', '')).strip().lower() == gt['q2_spot2_visible']
    try:
        out['q3'] = int(pred['q3_first_observed_step']) == gt['q3_first_observed_step']
    except Exception:
        out['q3'] = False
    out['q4'] = str(pred.get('q4_final_location', '')).strip().lower() == gt['q4_final_location'].lower()
    return out