"""
Control 실험(공간/궤적 이해 점검) 문항 정의.

LLM이 ToM 실험에서 받았던 것과 동일한 Map Configuration + 전체 궤적 로그를 주고,
로그와 좌표를 제대로 이해했는지 사실 질문으로 점검함.

이 모듈 하나에 다음을 모아 프롬프트와 채점이 같은 정의를 공유하게 함:
  - sample_probes  : 문항에 쓰일 시점(t)을 (seed, subject, scenario)로 고정된 난수로 추출
  - control_task_text : 프롬프트에 들어갈 질문 + JSON 템플릿
  - ground_truth   : 정답 계산
  - score_response : 응답 채점

[격자 경계에 관한 주의]
원래 ToM 프롬프트의 Map Configuration에는 격자 크기(15x5)가 명시되어 있지 않음.
건물이 격자 가장자리에 닿는 배치(x=1~14, x=2~15)에서는, 프롬프트만 보면
격자 밖으로 돌아가는 더 짧은 경로가 가능해 보임. 설정을 그대로 전달하는 원칙에 따라
프롬프트에 격자 크기를 추가하지 않고, 최단거리 정답을 두 가지로 계산함:
  - bounded   : 실제 실험 격자(1~15, 1~5) 안에서의 최단거리 (= 실험의 실제 세계)
  - unbounded : 격자 경계 없이 건물만 피하는 최단거리 (= 프롬프트만으로 추론 가능한 값)
"""
import json
import random
from collections import deque

GRID_X = (1, 15)
GRID_Y = (1, 5)
_UNBOUNDED_PAD = 20  # 경계 없는 BFS의 탐색 범위 (충분히 넓게)

CHANGE_OPTIONS = ("closer", "farther", "same")
LOCATION_OPTIONS = ("Spot 1", "Spot 2", "Neither")


# =========================================================================
# 거리 계산 (prepare_everystep.get_true_distance와 같은 규칙: 상하좌우, 건물 칸 통과 불가)
# =========================================================================
def _in_wall(x, y, wall):
    wx, wy, ww, wh = wall
    return wx <= x < wx + ww and wy <= y < wy + wh


def shortest_moves(start, target, wall, bounded=True):
    """start -> target 최단 이동 횟수. bounded=False면 격자 경계를 무시함."""
    if bounded:
        xlo, xhi, ylo, yhi = GRID_X[0], GRID_X[1], GRID_Y[0], GRID_Y[1]
    else:
        xlo, xhi = GRID_X[0] - _UNBOUNDED_PAD, GRID_X[1] + _UNBOUNDED_PAD
        ylo, yhi = GRID_Y[0] - _UNBOUNDED_PAD, GRID_Y[1] + _UNBOUNDED_PAD
    start, target = tuple(map(int, start)), tuple(map(int, target))
    queue, seen = deque([(start, 0)]), {start}
    while queue:
        (x, y), d = queue.popleft()
        if (x, y) == target:
            return d
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if not (xlo <= nx <= xhi and ylo <= ny <= yhi):
                continue
            if _in_wall(nx, ny, wall) or (nx, ny) in seen:
                continue
            seen.add((nx, ny))
            queue.append(((nx, ny), d + 1))
    return None


# =========================================================================
# 시나리오 정보 추출
# =========================================================================
def _scenario_info(df_scenario):
    df = df_scenario.sort_values('time_step')
    r0 = df.iloc[0]
    wall = tuple(int(r0[c]) for c in ('wall_start_x', 'wall_start_y', 'wall_width', 'wall_height'))
    spots = {1: (int(r0['goal1_x']), int(r0['goal1_y'])), 2: (int(r0['goal2_x']), int(r0['goal2_y']))}
    steps = [int(t) for t in df['time_step']]
    pos = {int(t): (int(x), int(y)) for t, x, y in zip(df['time_step'], df['agent_x'], df['agent_y'])}
    vis2 = {int(t): int(v) for t, v in zip(df['time_step'], df['visible_goal2'])}
    return df, wall, spots, steps, pos, vis2


# =========================================================================
# Probe 시점 추출
# =========================================================================
def sample_probes(df_scenario, subject_idx=1, seed=0):
    """
    문항에 쓰일 시점을 뽑음. (seed, subject, scenario)로 시드가 고정되어 재실행/재개해도 동일함.
      t1  : 좌표를 묻는 시점 (전체 중 무작위)
      t2a, t2b : Spot 2 가시 여부를 묻는 두 시점. 가능하면 '안 보임' 1개 + '보임' 1개를 뽑고
                 제시 순서는 무작위. 한쪽이 없으면 같은 쪽에서 서로 다른 두 시점.
      t9  : 직전 스텝 대비 거리 변화를 묻는 시점 (2번째 스텝 이후)
    """
    _, _, _, steps, _, vis2 = _scenario_info(df_scenario)
    sc_id = int(df_scenario['scenario_id'].iloc[0])
    rng = random.Random(f"control-{seed}-{subject_idx}-{sc_id}")

    t1 = rng.choice(steps)

    hidden = [t for t in steps if vis2[t] == 0]
    visible = [t for t in steps if vis2[t] == 1]
    if hidden and visible:
        pair = [rng.choice(hidden), rng.choice(visible)]
    else:
        pool = hidden or visible
        pair = rng.sample(pool, 2) if len(pool) >= 2 else [pool[0], pool[0]]
    rng.shuffle(pair)

    t9 = rng.choice(steps[1:])
    return {'probe_t1': t1, 'probe_t2a': pair[0], 'probe_t2b': pair[1], 'probe_t9': t9,
            'probe_seed': seed}


# =========================================================================
# 프롬프트 문항
# =========================================================================
def control_task_text(probes, max_steps):
    t1, t2a, t2b, t9 = probes['probe_t1'], probes['probe_t2a'], probes['probe_t2b'], probes['probe_t9']
    return f"""This log has {max_steps} time steps. Answer the following questions about the map and the log.
        These are questions about the movement log itself, not about the student's preferences or thoughts.

        [Movement rules for questions 5-9]
        - In one time step, the student either stays in place or moves to an adjacent cell (up, down, left, or right; no diagonal moves).
        - The student cannot enter cells occupied by the building.
        - A "move" is a time step at which the student's position differs from the previous time step.
        - The "distance" to a spot is the smallest number of moves needed to reach it without entering the building.

        1. What were the student's coordinates at Time Step {t1}?
        2. (a) At Time Step {t2a}, was Spot 2 visible to the student? Answer "yes" or "no".
           (b) At Time Step {t2b}, was Spot 2 visible to the student? Answer "yes" or "no".
        3. At which time step did the student FIRST see Spot 2? If Spot 2 was never visible, answer 0.
        4. At the LAST time step, where was the student? Answer exactly one of: "Spot 1", "Spot 2", or "Neither".
        5. From the student's starting position (Time Step 1), what is the smallest number of moves needed to reach Spot 1?
        6. From the student's starting position (Time Step 1), what is the smallest number of moves needed to reach Spot 2?
        7. Over the whole log, how many moves did the student make?
        8. Over the whole log, at how many time steps did the student stay at the same position as in the previous time step?
        9. Compared with Time Step {t9 - 1}, at Time Step {t9} did the student's distance to Spot 1 become "closer", "farther", or stay the "same"? Answer the same for Spot 2.

        Return the result in the following JSON structure:
        {{
            "q1_position": {{ "x": int, "y": int }},
            "q2a_spot2_visible": "yes" or "no",
            "q2b_spot2_visible": "yes" or "no",
            "q3_first_observed_step": int,
            "q4_final_location": "Spot 1" or "Spot 2" or "Neither",
            "q5_shortest_moves_spot1": int,
            "q6_shortest_moves_spot2": int,
            "q7_total_moves": int,
            "q8_stay_steps": int,
            "q9_spot1_change": "closer" or "farther" or "same",
            "q9_spot2_change": "closer" or "farther" or "same"
        }}
        """


# =========================================================================
# 정답
# =========================================================================
def ground_truth(df_scenario, probes):
    """probe 시점에 대한 정답 dict. 최단거리 문항(q5, q6)은 bounded / unbounded 두 정답을 모두 둠."""
    _, wall, spots, steps, pos, vis2 = _scenario_info(df_scenario)
    probes = {k: int(v) for k, v in probes.items() if k.startswith('probe_t')}
    start, last = pos[steps[0]], pos[steps[-1]]

    seen = [t for t in steps if vis2[t] == 1]
    final = "Spot 1" if last == spots[1] else "Spot 2" if last == spots[2] else "Neither"
    moves = sum(pos[a] != pos[b] for a, b in zip(steps[:-1], steps[1:]))

    def change(spot):
        t = probes['probe_t9']
        before = shortest_moves(pos[t - 1], spots[spot], wall)
        after = shortest_moves(pos[t], spots[spot], wall)
        return "closer" if after < before else "farther" if after > before else "same"

    gt = {
        'q1_position': {'x': pos[probes['probe_t1']][0], 'y': pos[probes['probe_t1']][1]},
        'q2a_spot2_visible': "yes" if vis2[probes['probe_t2a']] else "no",
        'q2b_spot2_visible': "yes" if vis2[probes['probe_t2b']] else "no",
        'q3_first_observed_step': seen[0] if seen else 0,
        'q4_final_location': final,
        'q5_shortest_moves_spot1': shortest_moves(start, spots[1], wall, bounded=True),
        'q6_shortest_moves_spot2': shortest_moves(start, spots[2], wall, bounded=True),
        'q7_total_moves': moves,
        'q8_stay_steps': (len(steps) - 1) - moves,
        'q9_spot1_change': change(1),
        'q9_spot2_change': change(2),
        # 격자 경계를 모르는 경우의 정답 (설명은 모듈 docstring 참고)
        'q5_shortest_moves_spot1_unbounded': shortest_moves(start, spots[1], wall, bounded=False),
        'q6_shortest_moves_spot2_unbounded': shortest_moves(start, spots[2], wall, bounded=False),
    }
    return gt


# =========================================================================
# 채점
# =========================================================================
ITEMS = ['q1_position', 'q2a_spot2_visible', 'q2b_spot2_visible', 'q3_first_observed_step',
         'q4_final_location', 'q5_shortest_moves_spot1', 'q6_shortest_moves_spot2',
         'q7_total_moves', 'q8_stay_steps', 'q9_spot1_change', 'q9_spot2_change']
NUMERIC_ITEMS = ['q3_first_observed_step', 'q5_shortest_moves_spot1', 'q6_shortest_moves_spot2',
                 'q7_total_moves', 'q8_stay_steps']


def _as_obj(v):
    """CSV에 JSON 문자열로 저장된 값을 되돌림."""
    if isinstance(v, str):
        s = v.strip()
        if s.startswith('{') or s.startswith('['):
            try:
                return json.loads(s)
            except Exception:
                return v
    return v


def _as_int(v):
    try:
        if v is None or (isinstance(v, float) and v != v):
            return None
        return int(round(float(v)))
    except Exception:
        return None


def _as_label(v):
    if isinstance(v, bool):
        return "yes" if v else "no"
    if v is None or (isinstance(v, float) and v != v):
        return ""
    return str(v).strip().strip('"').lower()


def score_response(pred, gt):
    """
    Returns: dict
      {item}_correct : bool (q5, q6는 bounded 정답 기준)
      q5/q6 _correct_unbounded, _correct_either : 격자 경계 해석별 정오
      {numeric item}_abs_err : |응답 - 정답| (숫자 문항)
      n_correct, n_items
    """
    out = {}
    p1 = _as_obj(pred.get('q1_position'))
    try:
        out['q1_position_correct'] = (_as_int(p1['x']) == gt['q1_position']['x'] and
                                      _as_int(p1['y']) == gt['q1_position']['y'])
    except Exception:
        out['q1_position_correct'] = False

    for k in ('q2a_spot2_visible', 'q2b_spot2_visible', 'q9_spot1_change', 'q9_spot2_change'):
        out[f'{k}_correct'] = _as_label(pred.get(k)) == gt[k]
    out['q4_final_location_correct'] = _as_label(pred.get('q4_final_location')) == gt['q4_final_location'].lower()

    for k in NUMERIC_ITEMS:
        v = _as_int(pred.get(k))
        out[f'{k}_correct'] = v is not None and v == gt[k]
        out[f'{k}_abs_err'] = abs(v - gt[k]) if v is not None and gt[k] is not None else None

    for k in ('q5_shortest_moves_spot1', 'q6_shortest_moves_spot2'):
        v = _as_int(pred.get(k))
        out[f'{k}_correct_unbounded'] = v is not None and v == gt[f'{k}_unbounded']
        out[f'{k}_correct_either'] = out[f'{k}_correct'] or out[f'{k}_correct_unbounded']

    flags = [out[f'{k}_correct'] for k in ITEMS]
    out['n_correct'], out['n_items'] = int(sum(flags)), len(flags)
    return out
