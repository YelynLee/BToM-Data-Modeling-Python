from dataset import df_btom
import pandas as pd

SYSTEM_PROMPT_BASE = """
You are a participant in a psychology experiment designed to infer intentions from behavior.
Please make judgments based strictly on the provided setting descriptions.

[Experiment Settings]
1. Context: Students (represented as face icons) at the Manitoba Institute of Technology (MIT) love food trucks. They already know the locations of the buildings (black squares) and the parking spots (yellow squares).
2. Trucks: Three different trucks (Mexican=M, Korean=K, Lebanese=L) compete for parking spots daily. There are 2 parking spots in total. Spots can be occupied by a truck or be empty.
3. Visibility: Students CANNOT see the 'grey area' occluded by the building (Invisible). They CAN see the 'white area' (Visible). For parking spots they cannot currently see, students rely on prior experience to know, believe, or guess which truck is there.
4. Motivation: Students are very hungry and want to eat immediately, but they are willing to walk further if there is a food truck they prefer more.

The logs provided represent the decision-making process of different students on different days as they move around the campus to decide where to have lunch.

[Output Format]
Please provide the response in JSON format. Do not include any markdown formatting like ```json.
"""

SYSTEM_PROMPT_EVERY = """
You are a participant in a psychology experiment designed to infer intentions from behavior.
Please make judgments based strictly on the provided setting descriptions.

[Experiment Settings]
1. Context: Students (represented as face icons) at the Manitoba Institute of Technology (MIT) love food trucks. They already know the locations of the buildings (black squares) and the parking spots (yellow squares).
2. Trucks: Three different trucks (Mexican=M, Korean=K, Lebanese=L) compete for parking spots daily. There are 2 parking spots in total. Spots can be occupied by a truck or be empty.
3. Visibility: Students CANNOT see the 'grey area' occluded by the building (Invisible). They CAN see the 'white area' (Visible). For parking spots they cannot currently see, students rely on prior experience to know, believe, or guess which truck is there.
4. Motivation: Students are very hungry and want to eat immediately, but they are willing to walk further if there is a food truck they prefer more.
5. Time step: Crucially, agents only know what they have seen SO FAR.

The logs provided represent the decision-making process of different students on different days as they move around the campus to decide where to have lunch.

[STRICT RULES]
1. Sequential Processing: You must process the log strictly from Time Step 1 to the end.
2. NO Look-ahead: When analyzing Time Step 't', you must NOT use any information from future steps (t+1, t+2...). 
   - Pretend you are watching a live video feed and do not know the ending.
3. Update: Update the estimates ONLY based on the accumulated evidence up to the current step.
4. YES Look-back: When you're doing the 'belief analysis', you must retrospectively consider the agent's initial belief at "time_step": 1.
5. NO LAZINESS (CRITICAL): You must output data for EVERY SINGLE time step present in the log. Do NOT skip, abbreviate, or use placeholders like "...".

[Output Format]
Please provide the response in JSON format. Do not include any markdown formatting like ```json.
"""

# =========================================================================
# 🌟 [NEW] 역방향 추론용 System Prompt
# =========================================================================
SYSTEM_PROMPT_REVERSE = """
You are a cognitive scientist analyzing the internal mental states of a student looking for a food truck.
Instead of seeing their physical movements, you are ONLY given their internal "Desire" and "Belief" scores over time.
Please make judgments based strictly on the provided setting descriptions.

[Experiment Settings]
1. Context: Students (represented as face icons) at the Manitoba Institute of Technology (MIT) love food trucks. They already know the locations of the buildings (black squares) and the parking spots (yellow squares).
2. Trucks: Three different trucks (Mexican=M, Korean=K, Lebanese=L) compete for parking spots daily. There are 2 parking spots in total. Spots can be occupied by a truck or be empty.
3. Visibility: Students CANNOT see the 'grey area' occluded by the building (Invisible). They CAN see the 'white area' (Visible). For parking spots they cannot currently see, students rely on prior experience to know, believe, or guess which truck is there.
4. Motivation: Students are very hungry and want to eat immediately, but they are willing to walk further if there is a food truck they prefer more.

The logs provided represent the decision-making process of different students on different days as they move around the campus to decide where to have lunch.

[Output Format]
Please provide the response in JSON format. Do not include any markdown formatting like ```json.
"""

# =========================================================================
# 🌟 [NEW] Control용 System Prompt
# =========================================================================
SYSTEM_PROMPT_CONTROL = """
You are a participant in a psychology experiment about reading movement logs.
Please answer strictly from the provided setting descriptions.

[Experiment Settings]
1. Context: Students (represented as face icons) at the Manitoba Institute of Technology (MIT) love food trucks. They already know the locations of the buildings (black squares) and the parking spots (yellow squares).
2. Trucks: Three different trucks (Mexican=M, Korean=K, Lebanese=L) compete for parking spots daily. There are 2 parking spots in total. Spots can be occupied by a truck or be empty.
3. Visibility: Students CANNOT see the 'grey area' occluded by the building (Invisible). They CAN see the 'white area' (Visible). For parking spots they cannot currently see, students rely on prior experience to know, believe, or guess which truck is there.

[Output Format]
Please provide the response in JSON format. Do not include any markdown formatting like ```json.
"""

# ver 3/16: Check-Goback 중 실제 예시(일단 모델 상관없이 scenario 1번) & subject/btom 1번 피험자 응답을 가져옴
# 외에도 그룹을 달리하거나, 개수를 늘리는 등 조작이 필요
ONE_SHOT_EXAMPLE = """
[Reference Example: How to analyze the Log]
To help you understand, here is an example of how a human observer scored a scenario.

[Map Configuration]
- Parking Spot 1: Located at (1, 1)
- Parking Spot 2: Located at (15, 5)
- Obstacle (Building): Starts at (2, 3) with size 13x1
- Trucks: Truck K at Spot 1, Truck L at Spot 2

[Chronological Log]
Time Step 1: Agent at (2, 2) | Spot 1 is Visible (Observed: Truck K) | Spot 2 is NOT Visible (Occluded)
Time Step 2: Agent at (1, 2) | Spot 1 is Visible (Observed: Truck K) | Spot 2 is NOT Visible (Occluded)
Time Step 3: Agent at (1, 3) | Spot 1 is Visible (Observed: Truck K) | Spot 2 is NOT Visible (Occluded)
Time Step 4: Agent at (1, 4) | Spot 1 is Visible (Observed: Truck K) | Spot 2 is Visible (Observed: Truck L)
Time Step 5: Agent at (1, 4) | Spot 1 is Visible (Observed: Truck K) | Spot 2 is Visible (Observed: Truck L)
Time Step 6: Agent at (1, 3) | Spot 1 is Visible (Observed: Truck K) | Spot 2 is NOT Visible (Occluded)
Time Step 7: Agent at (1, 2) | Spot 1 is Visible (Observed: Truck K) | Spot 2 is NOT Visible (Occluded)
Time Step 8: Agent at (1, 1) | Spot 1 is Visible (Observed: Truck K) | Spot 2 is NOT Visible (Occluded)

[Example JSON Output]
{
  "desire_scores": { "K": 4, "L": 3, "M": 6 },
  "belief_scores": { "time_step": 1, "L": 4, "M": 6, "Empty": 4 }
}
"""

ONE_SHOT_EXAMPLE_EVERY = """
[Reference Example: How to analyze the Log]
To help you understand, here is an example of how a human observer scored a scenario.

[Map Configuration]
- Parking Spot 1: Located at (1, 1)
- Parking Spot 2: Located at (15, 5)
- Obstacle (Building): Starts at (2, 3) with size 13x1
- Trucks: Truck K at Spot 1, Truck L at Spot 2

[Chronological Log]
Time Step 1: Agent at (2, 2) | Spot 1 is Visible (Observed: Truck K) | Spot 2 is NOT Visible (Occluded)
Time Step 2: Agent at (1, 2) | Spot 1 is Visible (Observed: Truck K) | Spot 2 is NOT Visible (Occluded)
Time Step 3: Agent at (1, 3) | Spot 1 is Visible (Observed: Truck K) | Spot 2 is NOT Visible (Occluded)
Time Step 4: Agent at (1, 4) | Spot 1 is Visible (Observed: Truck K) | Spot 2 is Visible (Observed: Truck L)
Time Step 5: Agent at (1, 4) | Spot 1 is Visible (Observed: Truck K) | Spot 2 is Visible (Observed: Truck L)
Time Step 6: Agent at (1, 3) | Spot 1 is Visible (Observed: Truck K) | Spot 2 is NOT Visible (Occluded)
Time Step 7: Agent at (1, 2) | Spot 1 is Visible (Observed: Truck K) | Spot 2 is NOT Visible (Occluded)
Time Step 8: Agent at (1, 1) | Spot 1 is Visible (Observed: Truck K) | Spot 2 is NOT Visible (Occluded)
   
[Example JSON Output]
[
  {
    "time_step": 1,
    "desire_scores": { "K": 4, "L": 4, "M": 4 },
    "belief_scores": { "L": 4, "M": 4, "Empty": 4 }
  },
  {
    "time_step": 2,
    "desire_scores": { "K": 4, "L": 4, "M": 4 },
    "belief_scores": { "L": 4, "M": 4, "Empty": 4 }
  },
  {
    "time_step": 3,
    "desire_scores": { "K": 2, "L": 4, "M": 4 },
    "belief_scores": { "L": 5, "M": 5, "Empty": 4 }
  },
  {
    "time_step": 4,
    "desire_scores": { "K": 2, "L": 4, "M": 4 },
    "belief_scores": { "L": 5, "M": 5, "Empty": 3 }
  },
  {
    "time_step": 5,
    "desire_scores": { "K": 2, "L": 3, "M": 5 },
    "belief_scores": { "L": 4, "M": 6, "Empty": 3 }
  },
  {
    "time_step": 6,
    "desire_scores": { "K": 3, "L": 2, "M": 6 },
    "belief_scores": { "L": 3, "M": 6, "Empty": 3 }
  },
  {
    "time_step": 7,
    "desire_scores": { "K": 3, "L": 2, "M": 6 },
    "belief_scores": { "L": 2, "M": 6, "Empty": 2 }
  },
  {
    "time_step": 8,
    "desire_scores": { "K": 4, "L": 2, "M": 6 },
    "belief_scores": { "L": 2, "M": 6, "Empty": 2 }
  }
]
"""

def generate_scenario_prompt(df_scenario, condition='vanilla', mode='normal'):
    """
    Args:
        df_scenario: 시나리오 데이터프레임
        condition: 'vanilla', 'reasoning', 'oneshot'
        mode: 'normal', 'everystep', 'reverse'
    Returns:
        system_prompt, user_prompt
    """
    # 🌟 [추가] 시나리오의 전체 타임스텝 수 계산
    max_steps = int(df_scenario['time_step'].max())
    row0 = df_scenario.iloc[0]

    # =========================================================================
    # 🌟 [NEW] Mode: Reverse (역방향 추론 방식)
    # =========================================================================
    if mode == 'reverse':
        system_prompt = SYSTEM_PROMPT_REVERSE
        
        # 1. Map Info (역방향 추론에서도 기본 위치 정보는 제공)
        static_info = f"""
        [Map Configuration]
        - Parking Spot 1 (G1) is always closer and seen to the student at t=1
        - Parking Spot 2 (G2) is further away and hidden behind the building at t=1
        - The student can move aside the building and check G2, then approach or return to the truck he/she prefers
        - Truck K is always at G1 and Truck L, M, or Empty can be at G2
        """
        
        # 2. Mental State Logs (좌표나 시야 정보 없이 오직 점수만 제공)
        logs = []
        for _, row in df_scenario.iterrows():
            t = int(row['time_step'])
            # 소수점 3자리로 포맷팅
            d_str = f"Desire(K: {row['desire_K']:.3f}, L: {row['desire_L']:.3f}, M: {row['desire_M']:.3f})"
            b_str = f"Belief(L: {row['belief_L']:.3f}, M: {row['belief_M']:.3f}, Empty: {row['belief_Empty']:.3f})"
            logs.append(f"Time Step {t}: {d_str} | {b_str}")
            
        dynamic_logs = "\n".join(logs)
        
        task_instruction = f"""
        # [TARGET SCENARIO START]
        # Now, please analyze tne scenario provided below.

        {static_info}

        [Mental State Log]
        - Desire: The relative preference score for each truck (Scale: 1 = Dislike strongly to 7 = Like strongly)
        - Belief: The probability of what is hiding at G2 at t=1 (Scale: 0.0 = Definitely not there to 1.0 = Definitely there, sum to 1)
        {dynamic_logs}

        Based on the changes in Belief and Desire, please perform the following tasks:

        1. Select the timestep interval where the student passed Spot 1 (G1) but didn't see the truck at Spot 2 (G2) yet.
        
        2. After checking Spot 2, infer if the student approached G2 or returned to G1 and also select the timestep interval of this action.

        3. Based on the overall trajectory, infer if the truck at Spot 2 was present or not.

        Return the result in the following JSON structure:
        {{
            "pass_G1_interval": [start_time, end_time],
            "post_check_action": "Return G1",   // Choose exactly between: "Return G1" or "Approach G2"
            "post_check_interval": [start_time, end_time],
            "G2_status": "present"              // Choose exactly between: "present" or "absent"
        }}
        """
        prompt_content = task_instruction
        return system_prompt, prompt_content

    # =========================================================================
    # 기존 Mode 처리 (normal, everystep)
    # =========================================================================
    # 💡 [NEW/수정] Spot을 기준으로 어떤 트럭이 있는지 검사 (빈 자리 처리 포함)
    spot1_truck = "No truck"
    spot2_truck = "No truck"
    
    for truck in ['K', 'L', 'M']:
        # 트럭의 좌표를 안전하게 가져옴
        t_x = df_scenario.get(f'{truck}_x', pd.Series([0])).iloc[0]
        t_y = df_scenario.get(f'{truck}_y', pd.Series([0])).iloc[0]
        
        # Spot 1 위치와 트럭 좌표가 같으면
        if t_x == row0['goal1_x'] and t_y == row0['goal1_y']:
            spot1_truck = f"Truck {truck}"
        # Spot 2 위치와 트럭 좌표가 같으면
        elif t_x == row0['goal2_x'] and t_y == row0['goal2_y']:
            spot2_truck = f"Truck {truck}"

    # 최종 문자열 생성 (항상 두 Spot의 상태를 명시)
    trucks_info_str = f"{spot1_truck} at Spot 1, {spot2_truck} at Spot 2"
    
    # 1. Static Map Info (첫 번째 행 기준)
    static_info = f"""
    [Map Configuration]
    - Parking Spot 1: Located at ({row0['goal1_x']}, {row0['goal1_y']})
    - Parking Spot 2: Located at ({row0['goal2_x']}, {row0['goal2_y']})
    - Obstacle (Building): Starts at ({row0['wall_start_x']}, {row0['wall_start_y']}) with size {row0['wall_width']}x{row0['wall_height']}
    - Trucks: {trucks_info_str}
    """

    # 2. Dynamic Trajectory Log
    logs = []

    # --- t=1 시점에서 관찰된 트럭 파악하기 ---
    visible_truck_at_t1 = None

    for _, row in df_scenario.iterrows():
        t = row['time_step']
        agent_pos = f"({row['agent_x']}, {row['agent_y']})"
        
        # --- Observation Logic (시야에 따른 정보 제공) ---
        # Goal 1
        if row['visible_goal1'] == 1:
            if row['K_x'] == row['goal1_x'] and row['K_y'] == row['goal1_y']: obs1 = "Truck K"
            elif row['L_x'] == row['goal1_x'] and row['L_y'] == row['goal1_y']: obs1 = "Truck L"
            elif row['M_x'] == row['goal1_x'] and row['M_y'] == row['goal1_y']: obs1 = "Truck M"
            else: obs1 = "Empty"
            g1_str = f"Visible (Observed: {obs1})"
        else:
            g1_str = "NOT Visible (Occluded)"

        # Goal 2
        if row['visible_goal2'] == 1:
            if row['K_x'] == row['goal2_x'] and row['K_y'] == row['goal2_y']: obs2 = "Truck K"
            elif row['L_x'] == row['goal2_x'] and row['L_y'] == row['goal2_y']: obs2 = "Truck L"
            elif row['M_x'] == row['goal2_x'] and row['M_y'] == row['goal2_y']: obs2 = "Truck M"
            else: obs2 = "Empty"
            g2_str = f"Visible (Observed: {obs2})"
        else:
            g2_str = "NOT Visible (Occluded)"

        # t=1일 때 무엇이 보였는가? (Observed Truck 저장)
        if t == 1:
            if "Truck K" in obs1 or "Truck K" in obs2: visible_truck_at_t1 = "K"
            elif "Truck L" in obs1 or "Truck L" in obs2: visible_truck_at_t1 = "L"
            elif "Truck M" in obs1 or "Truck M" in obs2: visible_truck_at_t1 = "M"

        log_line = f"Time Step {t}: Agent at {agent_pos} | Spot 1 is {g1_str} | Spot 2 is {g2_str}"
        logs.append(log_line)
    
    dynamic_logs = "\n".join(logs) # 모든 time step 내용을 한 번에 제공

    # --- 질문 목록 동적 생성 ---
    # 기본 옵션
    belief_options = ["K", "L", "M", "Empty"]
    
    # t=1에 이미 보인 트럭은 옵션에서 제거
    if visible_truck_at_t1 in belief_options:
        belief_options.remove(visible_truck_at_t1)
    
    # 프롬프트에 넣을 문자열 생성 (예: "L, M, and Empty")
    options_str = ", ".join(belief_options[:-1]) + ", and " + belief_options[-1]
    
    # JSON 템플릿 문자열 동적 생성
    # 예: "L": int, "M": int, "Empty": int
    json_fields = ", ".join([f'"{opt}": int' for opt in belief_options])

    # =========================================================================
    # 3. Instruction 구성 (매트릭스 로직)
    # =========================================================================
    # =========================================================================
    # [NEW] Mode: Control (spatial/visibility parsing validation)
    # =========================================================================
    if mode == 'control':
        system_prompt = SYSTEM_PROMPT_CONTROL

        # probe time steps
        t_pos = int(df_scenario['time_step'].iloc[len(df_scenario) // 2])   # Q1
        vis2 = df_scenario[df_scenario['visible_goal2'] == 1]
        t_vis = int(vis2['time_step'].min()) if not vis2.empty else None
        # Q2: a step strictly before first visibility (else the first step)
        t_occ = max(1, (t_vis - 1)) if t_vis else int(df_scenario['time_step'].max())

        task_instruction = f"""
        # [TARGET SCENARIO START]
        # Now, please read the scenario provided below and answer factual questions about it.

        {static_info}

        [Chronological Log]
        {dynamic_logs}

        This log has {max_steps} time steps. Answer the following questions about what the log
        states. These are questions about the log itself, not about the student's preferences or
        thoughts.

        1. What were the student's coordinates at Time Step {t_pos}?
        2. At Time Step {t_occ}, was Spot 2 visible to the student? Answer "yes" or "no".
        3. At which time step did the student FIRST observe the contents of Spot 2?
        If Spot 2 was never visible, answer 0.
        4. At the LAST time step, where was the student?
        Answer exactly one of: "Spot 1", "Spot 2", or "Neither".

        Return the result in the following JSON structure:
        {{
            "q1_position": {{ "x": int, "y": int }},
            "q2_spot2_visible": "yes",
            "q3_first_observed_step": int,
            "q4_final_location": "Spot 1"
        }}
        """
        return system_prompt, task_instruction
    
    # [1] Mode: Everystep (신규 방식 - 모든 스텝 분석)
    if mode == 'everystep':
        system_prompt = SYSTEM_PROMPT_EVERY

        prefix = ""
        if condition == "oneshot":
            prefix = f"""
            {ONE_SHOT_EXAMPLE_EVERY}
                
            # ============================================================================
            # [END OF EXAMPLE]
            # The example above is for reference only. Do NOT use its data for the task below.
            # ============================================================================
            """

        step_instruction = f"""
        # [TARGET SCENARIO START]
        # Now, please analyze tne scenario provided below.

        {static_info}

        [Chronological Log]
        {dynamic_logs}

        Based on the chronological situation above, please perform the following tasks:

        [CRITICAL REQUIREMENT]
        This log has exactly {max_steps} time steps. Your JSON array MUST contain exactly {max_steps} objects. Do NOT use "..." or skip any steps.

        1. At EVERY time step, rate the student's preference for Truck K, L, and M.
            (Scale: 1 = Dislike strongly to 7 = Like strongly)

        2. At EVERY time step, rate the student's likelihood for {options_str} being in the occluded spot at t=1 only given the current information.
            (Scale: 1 = Definitely not there to 7 = Definitely there)
  
        Return the result in the following JSON structure:
        [
            {{
                "time_step": 1,
                "desire_scores": {{ "K": int, "L": int, "M": int }},
                "belief_scores": {{ {json_fields} }}
            }},
            {{
                "time_step": 2,
                ...
            }}
        ]
        """
        prompt_content = f"{prefix}\n{step_instruction}"
            
    # [2] Mode: Normal (기존 방식 - 마지막 스텝만 분석)
    else:
        system_prompt = SYSTEM_PROMPT_BASE

        prefix = ""
        if condition == "oneshot":
            prefix = f"""
            {ONE_SHOT_EXAMPLE}
                
            # ============================================================================
            # [END OF EXAMPLE]
            # The example above is for reference only. Do NOT use its data for the task below.
            # ============================================================================
            """

        task_instruction = f"""
        # [TARGET SCENARIO START]
        # Now, please analyze tne scenario provided below.

        {static_info}

        [Chronological Log]
        {dynamic_logs}

        Based on the chronological situation above, please perform the following tasks:

        1. At the LAST time step, rate the student's preference for Truck K, L, and M.
            (Scale: 1 = Dislike strongly to 7 = Like strongly)
        
        2. At the FIRST time step, rate the student's likelihood for {options_str} being in the occluded spot at t=1.
            (Scale: 1 = Definitely not there to 7 = Definitely there)

        Return the result in the following JSON structure:
        {{
            "desire_scores": {{ "K": int, "L": int, "M": int }},
            "belief_scores": {{ "time_step": 1, {json_fields} }}
        }}
        """
        prompt_content = f"{prefix}\n{task_instruction}"

    return system_prompt, prompt_content

if __name__ == "__main__":
    # === 시나리오 테스트 ===
    target_id = 1

    print(f"시나리오 {target_id}번의 prompt 확인을 시작합니다.\n")

    target_df = df_btom[df_btom['scenario_id'] == target_id]

    if not target_df.empty:
        # 1. 메타 데이터 확인 (Group Desc, Truck Presence)
        row0 = target_df.iloc[0]
        group_desc = row0['group_desc']
        
        trucks = []
        if row0['K_x'] != 0 or row0['K_y'] != 0: trucks.append('K')
        if row0['L_x'] != 0 or row0['L_y'] != 0: trucks.append('L')
        if row0['M_x'] != 0 or row0['M_y'] != 0: trucks.append('M')
        truck_str = ", ".join(trucks) + " present" if trucks else "No trucks"

        print(f"=== Scenario {target_id} Metadata ===")
        print(f"Group: {group_desc}")
        print(f"Trucks: {truck_str}")
        print("=" * 60)
        print("\n")
        
        # 🌟 [수정된 부분] 테스트하고 싶은 모드들을 리스트로 정의
        test_modes = ['normal', 'everystep']
        
        # 각 모드별로 프롬프트를 생성하고 출력
        for current_mode in test_modes:
            print(f"🚀 [MODE: {current_mode.upper()}] 🚀".center(60, '='))
            
            # 모드 명시하여 프롬프트 생성
            sys_prompt, user_prompt = generate_scenario_prompt(target_df, condition='vanilla', mode=current_mode)
            
            print("\n[Generated System Prompt]")
            print(sys_prompt.strip())
            print("-" * 60)
            
            print("\n[Generated User Prompt]")
            print(user_prompt.strip())
            print("-" * 60)
            print("\n\n")
            
    else:
        print(f"Scenario {target_id} not found in dataset.")