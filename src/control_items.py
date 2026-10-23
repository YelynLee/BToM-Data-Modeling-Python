"""
Control 실험(공간/궤적·질문 이해 점검) 문항 정의.

LLM이 ToM 실험에서 받았던 것과 동일한 Map Configuration + 전체 궤적 로그를 주고,
로그와 좌표, 그리고 ToM 질문 문구 자체를 제대로 이해했는지 사실 질문으로 점검함.

이 모듈 하나에 다음을 모아 프롬프트와 채점이 같은 정의를 공유하게 함:
  - sample_probes     : 문항에 쓰일 시점(t)과 보기 순서를 (seed, subject, scenario)로 고정된 난수로 추출
  - control_task_text : 프롬프트에 들어갈 질문 + JSON 템플릿
  - ground_truth      : 정답 계산
  - score_response    : 응답 채점
  - upgrade_legacy_row: 이전 문항 번호 체계(v1: 9문항, v2: 13문항)로 저장된 결과를 현재 번호로 변환

[문항 구성 (v3)]  비슷한 문항끼리 묶어 순서대로 배치
  1  FIRST 시점 좌표            2  LAST 시점 좌표
  3  t=1에 가려진 spot           4  LAST 시점 위치 (Spot 1 / Spot 2 / Neither)  ※ 2와 같은 내용을 범주로 다시 물음
  5  특정 두 시점의 Spot 2 가시 여부 (안 보이는 시점 1개 + 보이는 시점 1개, 순서 무작위)
  6  Spot 2를 처음 본 스텝
  7  시작점 -> Spot 1 최단 이동   8  시작점 -> Spot 2 최단 이동
  9  전체 이동 횟수              10 제자리 스텝 수
  11 특정 시점의 거리 변화 (Spot 1, Spot 2)
  12 질문 해석 객관식: (a) End-step / Prefix-step 문구, (b) Every-step 문구

[격자 경계에 관한 주의]
원래 ToM 프롬프트의 Map Configuration에는 격자 크기(15x5)가 명시되어 있지 않음.
건물이 격자 가장자리에 닿는 배치(x=1~14, x=2~15)에서는, 프롬프트만 보면
격자 밖으로 돌아가는 더 짧은 경로가 가능해 보임. 설정을 그대로 전달하는 원칙에 따라
프롬프트에 격자 크기를 추가하지 않고, 최단거리 정답(q7, q8)을 두 가지로 계산함:
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

# -------------------------------------------------------------------------
# 질문 해석 문항 (q12): ToM 실험의 실제 질문 문구를 보여주고 무엇을 묻는지 고르게 함.
# 보기는 표준 id(a~d)로 정의하고, 화면에 보이는 순서(A~D)는 probe마다 섞음. 정답은 항상 'a'.
#   q12a: End-step / Prefix-step 문구
#   q12b: Every-step 문구 (특정 Time Step t에 대해 답한다고 가정)
#         b = current belief 해석, c = hindsight 해석 (미래 로그까지 사용)
# -------------------------------------------------------------------------
Q12A_OPTIONS = {
    "a": "What the student believed at Time Step 1 about which truck (if any) was in the occluded spot",
    "b": "What the student believes at the last time step about which truck (if any) is in Spot 2",
    "c": "Which truck (if any) is actually in Spot 2",
    "d": "How much the student likes each truck",
}
Q12B_OPTIONS = {
    "a": "What the student believed at Time Step 1, judged only from the log up to Time Step {t}",
    "b": "What the student believes at Time Step {t}, judged only from the log up to Time Step {t}",
    "c": "What the student believed at Time Step 1, judged from the whole log including later time steps",
    "d": "Which truck (if any) is actually in Spot 2",
}
LETTERS = "ABCD"


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


def belief_options_str(df_scenario):
    """ToM 프롬프트와 같은 belief 보기 문자열 (t=1에 보인 트럭은 제외). 예: 'L, M, and Empty'"""
    r0 = df_scenario.sort_values('time_step').iloc[0]
    seen = None
    for truck in ("K", "L", "M"):
        tx, ty = r0.get(f"{truck}_x", 0), r0.get(f"{truck}_y", 0)
        if r0['visible_goal1'] == 1 and tx == r0['goal1_x'] and ty == r0['goal1_y']:
            seen = truck
        if r0['visible_goal2'] == 1 and tx == r0['goal2_x'] and ty == r0['goal2_y']:
            seen = truck
    opts = [o for o in ("K", "L", "M", "Empty") if o != seen]
    return ", ".join(opts[:-1]) + ", and " + opts[-1]


# =========================================================================
# Probe 시점 / 보기 순서 추출
# =========================================================================
PROBE_COLS = ['probe_t5a', 'probe_t5b', 'probe_t11', 'probe_t12', 'probe_q12a_order', 'probe_q12b_order']


def sample_probes(df_scenario, subject_idx=1, seed=0):
    """
    문항에 쓰일 시점과 보기 순서를 뽑음. (seed, subject, scenario)로 시드가 고정되어 재실행/재개해도 동일함.
      t5a, t5b : Spot 2 가시 여부를 묻는 두 시점. 가능하면 '안 보임' 1개 + '보임' 1개, 제시 순서는 무작위.
                 한쪽이 없으면(18개 시나리오는 끝까지 못 봄) 같은 쪽에서 서로 다른 두 시점.
      t11      : 직전 스텝 대비 거리 변화를 묻는 시점 (2번째 스텝 이후)
      t12      : Every-step 문구 해석 문항의 기준 시점. 해석이 실제로 갈리는 시점(G2를 본 이후,
                 마지막 스텝 이전)을 우선 사용
    ※ 이전 버전과 같은 시점이 나오도록 난수 추출 순서를 유지함 (첫 추출은 이전 q1용 시점으로, 현재는 쓰지 않음)
    """
    _, _, _, steps, _, vis2 = _scenario_info(df_scenario)
    sc_id = int(df_scenario['scenario_id'].iloc[0])
    rng = random.Random(f"control-{seed}-{subject_idx}-{sc_id}")

    rng.choice(steps)  # (이전 버전 q1 시점; 추출 순서 유지용)

    hidden = [t for t in steps if vis2[t] == 0]
    visible = [t for t in steps if vis2[t] == 1]
    if hidden and visible:
        pair = [rng.choice(hidden), rng.choice(visible)]
    else:
        pool = hidden or visible
        pair = rng.sample(pool, 2) if len(pool) >= 2 else [pool[0], pool[0]]
    rng.shuffle(pair)

    t11 = rng.choice(steps[1:])

    after_seen = [t for t in steps if visible and t >= visible[0] and t < steps[-1]]
    t12 = rng.choice(after_seen) if after_seen else rng.choice(steps[1:])
    order_a = "".join(rng.sample("abcd", 4))
    order_b = "".join(rng.sample("abcd", 4))
    return {'probe_t5a': pair[0], 'probe_t5b': pair[1], 'probe_t11': t11, 'probe_t12': t12,
            'probe_q12a_order': order_a, 'probe_q12b_order': order_b, 'probe_seed': seed}


def _mc_lines(options, order, **fmt):
    return "\n".join(f"           {LETTERS[i]}. {options[cid].format(**fmt)}" for i, cid in enumerate(order))


# =========================================================================
# 프롬프트 문항
# =========================================================================
def control_task_text(probes, max_steps, options_str="L, M, and Empty"):
    t5a, t5b, t11, t12 = probes['probe_t5a'], probes['probe_t5b'], probes['probe_t11'], probes['probe_t12']
    q12a = (f'At the FIRST time step, rate the student\'s likelihood for {options_str} '
            f'being in the occluded spot at t=1.')
    q12b = (f'At EVERY time step, rate the student\'s likelihood for {options_str} '
            f'being in the occluded spot at t=1 only given the current information.')
    return f"""This log has {max_steps} time steps. Answer the following questions about the map and the log.
        These are questions about the movement log itself.

        [Movement rules]
        - In one time step, the student either stays in place or moves to an adjacent cell (up, down, left, or right; no diagonal moves).
        - The student cannot enter cells occupied by the building.
        - A "move" is a time step at which the student's position differs from the previous time step.
        - The "distance" to a spot is the smallest number of moves needed to reach it without entering the building.

        1. At the FIRST time step, where was the student? Give the coordinates.
        2. At the LAST time step, where was the student? Give the coordinates.
        3. Which parking spot was the occluded spot at t=1? Answer exactly one of: "Spot 1", "Spot 2", "Both", or "Neither".
        4. At the LAST time step, where was the student? Answer exactly one of: "Spot 1", "Spot 2", or "Neither".
        5. (a) At time step {t5a}, was Spot 2 visible to the student? Answer "yes" or "no".
           (b) At time step {t5b}, was Spot 2 visible to the student? Answer "yes" or "no".
        6. At which time step did the student FIRST see Spot 2? If Spot 2 was never visible, answer 0.
        7. From the student's starting position (Time Step 1), what is the smallest number of moves needed to reach Spot 1?
        8. From the student's starting position (Time Step 1), what is the smallest number of moves needed to reach Spot 2?
        9. Over the whole log, how many moves did the student make?
        10. Over the whole log, at how many time steps did the student stay at the same position as in the previous time step?
        11. Compared with Time Step {t11 - 1}, at Time Step {t11} did the student's distance to Spot 1 become "closer", "farther", or stay the "same"? Answer the same for Spot 2.
        12. (a) Consider this question about the log above: "{q12a}"
               What does this question ask you to rate? Choose one letter.
{_mc_lines(Q12A_OPTIONS, probes['probe_q12a_order'])}
            (b) Now consider this question, answered for Time Step {t12}: "{q12b}"
               For Time Step {t12}, what does this question ask you to rate? Choose one letter.
{_mc_lines(Q12B_OPTIONS, probes['probe_q12b_order'], t=t12)}

        Return the result in the following JSON structure:
        {{
            "q1_first_position": {{ "x": int, "y": int }},
            "q2_last_position": {{ "x": int, "y": int }},
            "q3_occluded_spot": "Spot 1" or "Spot 2" or "Both" or "Neither",
            "q4_final_location": "Spot 1" or "Spot 2" or "Neither",
            "q5a_spot2_visible": "yes" or "no",
            "q5b_spot2_visible": "yes" or "no",
            "q6_first_observed_step": int,
            "q7_shortest_moves_spot1": int,
            "q8_shortest_moves_spot2": int,
            "q9_total_moves": int,
            "q10_stay_steps": int,
            "q11_spot1_change": "closer" or "farther" or "same",
            "q11_spot2_change": "closer" or "farther" or "same",
            "q12a_question_meaning": "A" or "B" or "C" or "D",
            "q12b_question_meaning": "A" or "B" or "C" or "D"
        }}
        """


# =========================================================================
# 정답
# =========================================================================
def ground_truth(df_scenario, probes):
    """probe 시점에 대한 정답 dict. 최단거리 문항(q7, q8)은 bounded / unbounded 두 정답을 모두 둠.
    probes에 없는 항목(이전 버전 결과)의 정답은 None."""
    _, wall, spots, steps, pos, vis2 = _scenario_info(df_scenario)
    t = {k: int(v) for k, v in probes.items()
         if k.startswith('probe_t') and v is not None and v == v}
    start, last = pos[steps[0]], pos[steps[-1]]

    seen = [s for s in steps if vis2[s] == 1]
    final = "Spot 1" if last == spots[1] else "Spot 2" if last == spots[2] else "Neither"
    moves = sum(pos[a] != pos[b] for a, b in zip(steps[:-1], steps[1:]))

    vis1 = {int(a): int(v) for a, v in zip(df_scenario['time_step'], df_scenario['visible_goal1'])}
    hidden_t1 = [n for n, v in ((1, vis1[steps[0]]), (2, vis2[steps[0]])) if v == 0]
    occluded = "Both" if len(hidden_t1) == 2 else f"Spot {hidden_t1[0]}" if hidden_t1 else "Neither"

    def change(spot):
        if 'probe_t11' not in t:
            return None
        s_ = t['probe_t11']
        before = shortest_moves(pos[s_ - 1], spots[spot], wall)
        after = shortest_moves(pos[s_], spots[spot], wall)
        return "closer" if after < before else "farther" if after > before else "same"

    def letter(order_key):
        order = probes.get(order_key)
        return LETTERS[order.index("a")] if isinstance(order, str) else None  # 정답 'a'의 화면상 글자

    return {
        'q1_first_position': {'x': start[0], 'y': start[1]},
        'q2_last_position': {'x': last[0], 'y': last[1]},
        'q3_occluded_spot': occluded,
        'q4_final_location': final,
        'q5a_spot2_visible': ("yes" if vis2[t['probe_t5a']] else "no") if 'probe_t5a' in t else None,
        'q5b_spot2_visible': ("yes" if vis2[t['probe_t5b']] else "no") if 'probe_t5b' in t else None,
        'q6_first_observed_step': seen[0] if seen else 0,
        'q7_shortest_moves_spot1': shortest_moves(start, spots[1], wall, bounded=True),
        'q8_shortest_moves_spot2': shortest_moves(start, spots[2], wall, bounded=True),
        'q9_total_moves': moves,
        'q10_stay_steps': (len(steps) - 1) - moves,
        'q11_spot1_change': change(1),
        'q11_spot2_change': change(2),
        'q12a_question_meaning': letter('probe_q12a_order'),
        'q12b_question_meaning': letter('probe_q12b_order'),
        # 격자 경계를 모르는 경우의 정답 (설명은 모듈 docstring 참고)
        'q7_shortest_moves_spot1_unbounded': shortest_moves(start, spots[1], wall, bounded=False),
        'q8_shortest_moves_spot2_unbounded': shortest_moves(start, spots[2], wall, bounded=False),
    }


# =========================================================================
# 이전 버전 결과 변환
# =========================================================================
# v1(9문항) / v2(13문항) 키 -> 현재(v3) 키. v1·v2의 q1(무작위 시점 좌표)은 현재 문항에 없어서 제외.
LEGACY_KEY_MAP = {
    'q2a_spot2_visible': 'q5a_spot2_visible', 'q2b_spot2_visible': 'q5b_spot2_visible',
    'q3_first_observed_step': 'q6_first_observed_step',
    'q5_shortest_moves_spot1': 'q7_shortest_moves_spot1', 'q6_shortest_moves_spot2': 'q8_shortest_moves_spot2',
    'q7_total_moves': 'q9_total_moves', 'q8_stay_steps': 'q10_stay_steps',
    'q9_spot1_change': 'q11_spot1_change', 'q9_spot2_change': 'q11_spot2_change',
    'q10_occluded_spot': 'q3_occluded_spot', 'q11_first_position': 'q1_first_position',
    'q12_last_position': 'q2_last_position',
    'q13a_question_meaning': 'q12a_question_meaning', 'q13b_question_meaning': 'q12b_question_meaning',
    'probe_t2a': 'probe_t5a', 'probe_t2b': 'probe_t5b', 'probe_t9': 'probe_t11', 'probe_t13': 'probe_t12',
    'probe_q13a_order': 'probe_q12a_order', 'probe_q13b_order': 'probe_q12b_order',
}


def result_version(columns):
    cols = set(columns)
    if 'q1_first_position' in cols:
        return 'v3'
    if 'q13a_question_meaning' in cols:
        return 'v2'
    return 'v1'


def upgrade_legacy_row(row):
    """이전 버전(v1, v2) 결과 행을 현재 키로 변환. 현재 버전 행은 그대로 반환."""
    if 'q1_first_position' in row or 'q2_last_position' in row:
        return row
    out = {k: v for k, v in row.items() if k not in LEGACY_KEY_MAP and k not in ('q1_position', 'probe_t1')}
    for old, new in LEGACY_KEY_MAP.items():
        if old in row:
            out[new] = row[old]
    return out


# =========================================================================
# 채점
# =========================================================================
ITEMS = ['q1_first_position', 'q2_last_position', 'q3_occluded_spot', 'q4_final_location',
         'q5a_spot2_visible', 'q5b_spot2_visible', 'q6_first_observed_step',
         'q7_shortest_moves_spot1', 'q8_shortest_moves_spot2', 'q9_total_moves', 'q10_stay_steps',
         'q11_spot1_change', 'q11_spot2_change', 'q12a_question_meaning', 'q12b_question_meaning']
POSITION_ITEMS = ['q1_first_position', 'q2_last_position']
LABEL_ITEMS = ['q3_occluded_spot', 'q4_final_location', 'q5a_spot2_visible', 'q5b_spot2_visible',
               'q11_spot1_change', 'q11_spot2_change']
NUMERIC_ITEMS = ['q6_first_observed_step', 'q7_shortest_moves_spot1', 'q8_shortest_moves_spot2',
                 'q9_total_moves', 'q10_stay_steps']
MEANING_ITEMS = [('q12a_question_meaning', 'probe_q12a_order'), ('q12b_question_meaning', 'probe_q12b_order')]


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
    pred: 응답 행 (probe 보기 순서 포함). gt: ground_truth 결과.
    Returns: dict
      {item}_correct : bool (q7, q8은 bounded 정답 기준). 정답이 None인 문항은 키를 만들지 않음
      q7/q8 _correct_unbounded, _correct_either : 격자 경계 해석별 정오
      {numeric item}_abs_err : |응답 - 정답|
      q12a/q12b _choice : 고른 해석의 표준 id (a=정답, b=current 해석, c=q12a 실제 트럭 / q12b hindsight 해석, d=기타)
      n_correct, n_items
    """
    pred = upgrade_legacy_row(dict(pred))
    # 응답에 열 자체가 없는 문항(이전 버전 결과)은 채점에서 제외. 열은 있는데 값이 비면 오답으로 처리.
    gt = {k: (v if (k in pred or k.endswith('_unbounded')) else None) for k, v in gt.items()}
    out = {}
    for k in POSITION_ITEMS:
        if gt.get(k) is None:
            continue
        p = _as_obj(pred.get(k))
        try:
            out[f'{k}_correct'] = (_as_int(p['x']) == gt[k]['x'] and _as_int(p['y']) == gt[k]['y'])
        except Exception:
            out[f'{k}_correct'] = False

    for k in LABEL_ITEMS:
        if gt.get(k) is not None:
            out[f'{k}_correct'] = _as_label(pred.get(k)) == gt[k].lower()

    for k in NUMERIC_ITEMS:
        if gt.get(k) is None:
            continue
        v = _as_int(pred.get(k))
        out[f'{k}_correct'] = v is not None and v == gt[k]
        out[f'{k}_abs_err'] = abs(v - gt[k]) if v is not None else None

    for k in ('q7_shortest_moves_spot1', 'q8_shortest_moves_spot2'):
        if gt.get(k) is None:
            continue
        v = _as_int(pred.get(k))
        out[f'{k}_correct_unbounded'] = v is not None and v == gt[f'{k}_unbounded']
        out[f'{k}_correct_either'] = out[f'{k}_correct'] or out[f'{k}_correct_unbounded']

    for k, order_key in MEANING_ITEMS:
        if gt.get(k) is None:
            continue
        ans = _as_label(pred.get(k)).upper()[:1]
        order = pred.get(order_key)
        out[f'{k}_correct'] = ans == gt[k]
        out[f'{k}_choice'] = order[LETTERS.index(ans)] if isinstance(order, str) and ans in LETTERS else None

    flags = [out[f'{k}_correct'] for k in ITEMS if f'{k}_correct' in out]
    out['n_correct'], out['n_items'] = int(sum(flags)), len(flags)
    return out
