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
Please provide the response in JSON format Do not include any markdown formatting like ```json.
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
4. Consistency: If no new information is gained at a step, maintain the previous estimate.
5. NO LAZINESS (CRITICAL): You must output data for EVERY SINGLE time step present in the log. Do NOT skip, abbreviate, or use placeholders like "...".

[Output Format]
Please provide the response in JSON format strictly.
"""

# 일단은 Check-Goback 조건의 예시 하나
# 이후로는 그룹을 달리하거나, 개수를 늘리는 등 조작이 필요
ONE_SHOT_EXAMPLE = """
[Reference Example: How to analyze the Log]
To help you understand, here is an example of how a human observer analyzes a scenario.

[Map Configuration]
- Spot 1 (Visible): Truck K
- Spot 2 (Occluded): Truck L
- Note: Truck M is NOT present in this map.

[Chronological Log]
Time Step 1-2: Agent moves towards Spot 1 (Visible: K)
Time Step 3: Agent moves past Spot 1 (Visible: K) 
Time Step 4-5: Moves towards occluded Spot 2
Time Step 6: Agent arrives at Spot 2 (Visible: L)
Time Step 7: Agent moves past Spot 2 (Visible: L) 
Time Step 8-9: Agent moves towards to Spot 1
Time Step 10: Agent arrives at Spot 1 (Visible: K)

[Reasoning Logic]
1. Desire Analysis:
   - At Step 3, the agent ignored the visible Truck K, implying he/she hoped for a better option (L or M) in the blind spot.
   - At Step 7, he/she saw Truck L but rejected it.
   - At Step 10, he/she returned to K. This confirms preference order: M (hoped for) > K (settled for) > L (rejected).

2. Belief Analysis:
   - At Step 3, the agent walked away from a visible K. This action assumes an initial belief that his/her favorite truck (L or M) was in the occluded spot 2.
   - At Step 7, it can be assumed that he/she anticipated M but not L in the occluded spot 2 at t=1.
   - At Step 10, the initial belief that M would be in the occluded spot 2 is assumed to be the same.

[Example JSON Output]
{
  "desire_reasoning": "Passing visible K implies hope for L or M. Rejecting L to return to K confirms M > K > L.",
  "desire_scores": { "K": 5, "M": 7, "L": 2 },
  "belief_reasoning": "Ignoring visible K and rejecting L strongly suggests a high belief at t=1 that truck M is hidden at spot 2.",
  "belief_scores": { "time_step": 1, "M": 7, "L": 2, "Empty": 1 }
}
"""

ONE_SHOT_EXAMPLE_EVERY = """
[Reference Example: How to analyze the Log]
To help you understand, here is an example of how a human observer analyzes a scenario.

[Map Configuration]
- Spot 1 (Visible): Truck K
- Spot 2 (Occluded): Truck L
- Note: Truck M is NOT present in this map.

[Chronological Log]
Time Step 1-2: Agent moves towards Spot 1 (Visible: K)
Time Step 3: Agent moves past Spot 1 (Visible: K) 
Time Step 4-5: Moves towards occluded Spot 2
Time Step 6: Agent arrives at Spot 2 (Visible: L)
Time Step 7: Agent moves past Spot 2 (Visible: L) 
Time Step 8-9: Agent moves towards to Spot 1
Time Step 10: Agent arrives at Spot 1 (Visible: K)

[Reasoning Logic]
1. Desire Analysis:
   - At Step 3, the agent ignored the visible Truck K, implying he/she hoped for a better option (L or M) in the blind spot.
   - At Step 7, he/she saw Truck L but rejected it.
   - At Step 10, he/she returned to K. This confirms preference order: M (hoped for) > K (settled for) > L (rejected).

2. Belief Analysis:
   - At Step 3, the agent walked away from a visible K. This action assumes an initial belief that his/her favorite truck (L or M) was in the occluded spot 2.
   - At Step 7, it can be assumed that he/she anticipated M but not L in the occluded spot 2 at t=1.
   - At Step 10, the initial belief that M would be in the occluded spot 2 is assumed to be the same.
   
[Example JSON Output]
[
  {
    "time_step": 1,
    "desire_reasoning": "No action has been taken. No strong inference on desire can be made yet.",
    "desire_scores": { "K": 4, "M": 4, "L": 4 },
    "belief_reasoning": "At t=1, K is visible while others are invisible. No action has been taken to infer belief yet.",
    "belief_scores": { "M": 4, "L": 4, "Empty": 4 }
  },
  {
    "time_step": 2,
    "desire_reasoning": "Moving towards Spot 1. It may show preference on K, but no strong inference can be made yet.",
    "desire_scores": { "K": 6, "M": 4, "L": 4 },
    "belief_reasoning": "At t=1, K was visible while others were invisible. No enough evidence to infer belief on the occluded spot.",
    "belief_scores": { "M": 4, "L": 4, "Empty": 4 }
  },
  {
    "time_step": 3,
    "desire_reasoning": "Passed visible K to check the occluded spot. Missing truck L or M is likely preferred over K.",
    "desire_scores": { "K": 3, "M": 6, "L": 6 },
    "belief_reasoning": "At t=1, K was visible while others were invisible. Moving past K suggests expecting L or M.",
    "belief_scores": { "M": 6, "L": 6, "Empty": 1 }
  },
  "... (Include an object for EVERY time step 4, 5, 6 following the same structure) ...",
  {
    "time_step": 7,
    "desire_reasoning": "Spot 2 became visible, showing L. Passing K earlier and now rejecting L implies the missing M is the most preferred.",
    "desire_scores": { "K": 5, "M": 7, "L": 2 },
    "belief_reasoning": "K was initially visible, others occluded. Continuing past K and rejecting L strongly implies intial expectation of missing M.",
    "belief_scores": { "M": 7, "L": 2, "Empty": 1 }
  },
  "... (Include an object for EVERY time step 8, 9 following the same structure) ...",
  {
    "time_step": 10,
    "desire_reasoning": "Spot 2 was checked. Returning to K after rejecting L implies missing M was desired, but K is preferred over L.",
    "desire_scores": { "K": 5, "M": 7, "L": 2 },
    "belief_reasoning": "K was visible, L and M occluded at t=1. Leaving K and rejecting L meant hoping for missing M at spot 2.",
    "belief_scores": { "M": 7, "L": 2, "Empty": 1 }
  }
]
"""

def generate_scenario_prompt(df_scenario, condition='reasoning', mode='normal'):
    """
    Args:
        df_scenario: 시나리오 데이터프레임
        condition: 'vanilla', 'reasoning', 'oneshot'
        mode: 'normal', 'everystep'
    Returns:
        system_prompt, user_prompt
    """
    # 🌟 [추가] 시나리오의 전체 타임스텝 수 계산
    max_steps = int(df_scenario['time_step'].max())

    # 1. Static Map Info (첫 번째 행 기준)
    row0 = df_scenario.iloc[0]
    static_info = f"""
    [Map Configuration]
    - Parking Spot 1: Located at ({row0['goal1_x']}, {row0['goal1_y']})
    - Parking Spot 2: Located at ({row0['goal2_x']}, {row0['goal2_y']})
    - Obstacle (Building): Starts at ({row0['wall_start_x']}, {row0['wall_start_y']}) with size {row0['wall_width']}x{row0['wall_height']}
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
    belief_options = ["K", "M", "L", "Empty"]
    
    # t=1에 이미 보인 트럭은 옵션에서 제거
    if visible_truck_at_t1 in belief_options:
        belief_options.remove(visible_truck_at_t1)
    
    # 프롬프트에 넣을 문자열 생성 (예: "M, L, and Empty")
    options_str = ", ".join(belief_options[:-1]) + ", and " + belief_options[-1]
    
    # JSON 템플릿 문자열 동적 생성
    # 예: "K": int, "M": int, "Empty": int
    json_fields = ", ".join([f'"{opt}": int' for opt in belief_options])

    # =========================================================================
    # 3. Instruction 구성 (매트릭스 로직)
    # =========================================================================
    # [1] Mode: Everystep (신규 방식 - 모든 스텝 분석)
    if mode == 'everystep':
        system_prompt = SYSTEM_PROMPT_EVERY

        if condition == 'vanilla': # vanilla
            step_instruction = f"""
            # [TARGET SCENARIO START]
            # Now, please analyze tne scenario provided below.

            {static_info}

            [Chronological Log]
            {dynamic_logs}

            Based on the chronological situation above, please perform the following tasks:

            [CRITICAL REQUIREMENT]
            This log has exactly {max_steps} time steps. Your JSON array MUST contain exactly {max_steps} objects. Do NOT use "..." or skip any steps.

            1. At EVERY time step, rate the student's preference for Truck K, M, and L.
                (Scale: 1 = Dislike strongly to 7 = Like strongly)
            2. At EVERY time step, rate the student's likelihood for {options_str} being in the occluded spot at t=1.
                (Scale: 1 = Definitely not there to 7 = Definitely there)
  
            Return the result in the following JSON structure:
            [
                {{
                    "time_step": 1,
                    "desire_scores": {{ "K": int, "M": int, "L": int }},
                    "belief_scores": {{ {json_fields} }}
                }},
                {{
                    "time_step": 2,
                    ...
                }}
            ]
            """
            prompt_content = step_instruction

        else: # reasoning or oneshot
            # One-Shot일 경우 예시 추가
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

            1. At EVERY time step, judge the student's preference in regard to the entire path.
            - Integrate your analysis of (a) and (b) into a single concise paragraph (strictly under 40 words). Do NOT use bullet points or separate lines.
                (a) Based on the visibility at t=1, did he/she pass a visible truck to go to an occluded one after t=1?
                    That means, is there any time step when spot 2 becomes visible?
                (b) Identify the truck(s) NOT mentioned in the log. Based on the path, what can be inferred about his/her preference for this missing truck(s)?
            - Then, rate the student's preference for Truck K, M, and L.
                (Scale: 1 = Dislike strongly to 7 = Like strongly)
       
            2. At EVERY time step, judge the student's initial (t=1) belief.
                - Integrate your analysis of (a) and (b) into a single concise paragraph (strictly under 40 words). Do NOT use bullet points or separate lines.
                    (a) Which truck was visible and invisible at t=1?
                    (b) Based on the subsequent path after t=1, did he/she likely expect the missing truck(s) to be in the OCCLUDED area at t=1?
                - Then, rate the likelihood for {options_str} being in the occluded spot at t=1.
                    (Scale: 1 = Definitely not there to 7 = Definitely there)

            Return the result in the following JSON structure:
            [
                {{
                    "time_step": 1,
                    "desire_reasoning": "(a)..., (b)...",
                    "desire_scores": {{ "K": int, "M": int, "L": int }},
                    "belief_reasoning": "(a)..., (b)...",
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

        # (A) Vanilla: Reasoning 불필요, 점수만 요청
        if condition == 'vanilla':
            task_instruction = f"""
            # [TARGET SCENARIO START]
            # Now, please analyze tne scenario provided below.

            {static_info}

            [Chronological Log]
            {dynamic_logs}

            Based on the chronological situation above, please perform the following tasks:

            1. At the LAST time step, rate the student's preference for Truck K, M, and L.
                (Scale: 1 = Dislike strongly to 7 = Like strongly)
        
            2. At the FIRST time step, rate the student's likelihood for {options_str} being in the occluded spot at t=1.
                (Scale: 1 = Definitely not there to 7 = Definitely there)

            Return the result in the following JSON structure:
            {{
                "desire_scores": {{ "K": int, "M": int, "L": int }},
                "belief_scores": {{ "time_step": 1, {json_fields} }}
            }}
            """
            prompt_content = task_instruction

        # (B) Reasoning, One-Shot
        else:
            # One-Shot일 경우 예시 추가
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

            1. At the LAST time step, judge the student's preference in regard to the entire path.
                - Integrate your analysis of (a) and (b) into a single concise paragraph (strictly under 40 words). Do NOT use bullet points or separate lines.
                    (a) Based on the visibility at t=1, did he/she pass a visible truck to go to an occluded one after t=1?
                        That means, is there any time step when spot 2 becomes visible?
                    (b) Identify the truck(s) NOT mentioned in the log. Based on the path, what can be inferred about his/her preference for this missing truck(s)?
                - Then, rate the student's preference for Truck K, M, and L.
                    (Scale: 1 = Dislike strongly to 7 = Like strongly)
        
            2. At the FIRST time step, judge the student's initial (t=1) belief.
                - Integrate your analysis of (a) and (b) into a single concise paragraph (strictly under 40 words). Do NOT use bullet points or separate lines.
                    (a) Which truck was visible and invisible at t=1?
                    (b) Based on the subsequent path after t=1, did he/she likely expect the missing truck(s) to be in the OCCLUDED area at t=1?
                - Then, rate the likelihood for {options_str} being in the occluded spot at t=1.
                    (Scale: 1 = Definitely not there to 7 = Definitely there)

            Return the result in the following JSON structure:
            {{
                "desire_reasoning": "(a)..., (b)...",
                "desire_scores": {{ "K": int, "M": int, "L": int }},
                "belief_reasoning": "(a)..., (b)...",
                "belief_scores": {{ "time_step": 1, {json_fields} }}
            }}
            """
            prompt_content = f"{prefix}\n{task_instruction}"

    return system_prompt, prompt_content