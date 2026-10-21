# BToM-Data-Modeling-Python
Post-research after the following paper [Rational quantitative attribution of beliefs, desires and percepts in human mentalizing](https://www.nature.com/articles/s41562-017-0064)


## 1. Project Overview
- Objective: <br>Building an automated pipeline and golden dataset of food truck task for verifying LLM's Theory of Mind (ToM) capabilities
- Key Points:
   - **Schema Flattening**: <br>Unnested MATLAB cell arrays and structs into a 2D tabular format
   - **Integrity**: <br>Integrated QA routines to validate the data across the pipeline
   - **Robust Pipeline**: <br>Designed dynamic execution paths using quality thresholds
   - **Semantic Labeling**: <br>Applied BFS algorithm to assign semantic phase labels to coordinates


## 2. System Requirements
- OS: Tested on Windows
- Language: Python 3.12
- Dependencies: refer to 'requirements.txt'


## 3. Installation Guide
1. Clone the repository
```bash
   git clone https://github.com/YelynLee/BToM-Data-Modeling-Python.git
``` 
2. Install the required packages
```bash
   pip install -r requirements.txt
``` 


## 4. Instructions for Use

### Directory Structure
```
BToM_LLM
|--main_experiment.py (the main entry point to run the experiments)
|--run_analysis.py (the main entry point to run the analyses)
|--data
|   |--human
|   |   |--human_data.pkl
|   |   |--...
|   |--btom
|   |   |--btom_data.pkl
|   |   |--...
|   |--truebelief
|   |   |--truebelief_data.pkl
|   |--nocost
|   |   |--nocost_data.pkl
|   |--motionheuristic
|   |   |--motionheuristic_data.pkl
|--src
|   |--config.py
|   |--utils.py
|   |--dataset.py
|   |--prompts.py
|   |--api_client.py
|   |--data_processor.py
|   |--prepare_everystep.py
|   |--...
|--results
|   |--gpt-4o
|   |--gemini-2.5-flash
|   |--deepseek-chat
|   |--claude-opus-4-6
|   |--...
|--analysis
|   |--plot_bars.py
|   |--plot_scatter.py
|   |--plot_rmse_corr.py
|   |--plot_rsa.py
|   |--plot_everystep.py
|   |--find_best_beta.py
|   |--...
```


### Arguments Reference
- To process the data,

| Argument | Description | Available Options |
| :--- | :--- | :--- |
| `--model` | Target model for data processing | `human`, `btom`, `truebelief`, `gpt-4o`, etc. |
| `--condition` | Experiment condition | `vanilla`(default), `reasoning`, `oneshot` |
| `--mode` | Experiment option | `normal`(default), `everystep` |
| `--ref_only` | Convert reference data (MAT)  | this is needed for human, btom, truebelief, nocost, motionheuristic |
| `--beta` | Target beta score of btom | 2.5(default) |


- To run the experiments,

| Argument | Description | Available Options |
| :--- | :--- | :--- |
| `--model` | Target model for experiment | `human`, `btom`, `gpt-4o`, `gemini-2.5-flash`, etc. |
| `--condition` | Experiment condition | `vanilla`(default), `reasoning`, `oneshot` |
| `--mode` | Experiment option | `normal`(default), `everystep` |
| `--subjects` | Number of virtual subjects | 16(default) |


- To run the analyses,

| Argument | Description | Available Options |
| :--- | :--- | :--- |
| `--model` | Target model for analysis | `human`, `btom`, `gpt-4o`, `gemini-2.5-flash`, etc. |
| `--baseline` | Baseline to compare against | `human`(default), `btom`, `truebelief`, `nocost`, `motionheuristic` |
| `--condition` | Experiment condition | `vanilla`(default), `reasoning`, `oneshot` |
| `--mode` | Experiment option | `normal`(default), `everystep` |
| `--type` | Type of plot to generate | `all`(default), `bar`, `scatter`, `rmse`, `rsa`, `phase` |


### Execution Example
- You can use the preprocessed .pkl data, but to process from the original .mat (human, btom) or .csv (models) to .pkl, enter:
```bash
# if the target model is reference data (human, btom, truebelief, nocost, motionheuristic), you need --ref_only
python src/data_processor.py --ref_only --model truebelief --beta 9.0
python src/data_processor.py --model gpt-4o --condition vanilla --mode normal
```


- To run the experiments, enter:
```bash
python main_experiment.py --model gpt-4o --condition oneshot --mode normal --subjects 16
```


- Stepwise variants (Prefix-step / current belief), enter:
```bash
# 프롬프트만 미리 확인 (API 호출 없음)
python main_experiment.py --model claude-opus-4-6 --mode prefixstep --scenario_ids 1 --current_belief --preview

# Prefix-step: (scenario, t)마다 로그를 1..t로 잘라 End-step과 동일한 질문 -> results/{model}/vanilla/prefixstep/
python main_experiment.py --model claude-opus-4-6 --mode prefixstep --scenarios check --subjects 8

# Every-step / Prefix-step에서 initial belief와 현재(current) belief를 함께 질문 (질문 순서는 호출마다 무작위)
#   -> .../everystep_upto_cur/, .../prefixstep_cur/
python main_experiment.py --model claude-opus-4-6 --mode everystep --current_belief --scenarios check
python main_experiment.py --model claude-opus-4-6 --mode prefixstep --current_belief --scenarios check

# 논문에 쓴 기존 Every-step 문구 그대로 재현 -> .../everystep/
python main_experiment.py --model claude-opus-4-6 --mode everystep --legacy_wording

# 후처리 / 분석도 같은 플래그로 폴더를 찾음
python src/data_processor.py --model claude-opus-4-6 --condition vanilla --mode prefixstep --current_belief
python run_analysis.py --model claude-opus-4-6 --mode prefixstep --current_belief --type phase
```

| Argument (main_experiment.py) | Description |
| :--- | :--- |
| `--mode prefixstep` | (scenario, t)마다 독립 API 호출. 로그는 Time Step 1..t, Map Configuration은 전체 공개. 질문은 End-step과 동일 |
| `--scenarios check` | Check-GoBack / Check-Stay / Check-Partial 55개 시나리오만 (irrational 경로 제외) |
| `--scenario_ids 1,6,40` | 지정한 시나리오만 (`--scenarios`보다 우선) |
| `--current_belief` | initial belief(t=1)와 에이전트의 현재 belief를 따로 질문. CSV에 `belief_now_*` 컬럼 추가 |
| `--belief_order` | `random`(기본) / `initial_first` / `now_first`: 두 belief 질문의 제시 순서. random은 (seed, subject, scenario, t)로 시드를 고정해 호출마다 정하고 CSV `belief_order` 컬럼에 기록 |
| `--legacy_wording` | everystep / prefixstep의 belief 문구를 이전 버전으로 사용. 기본 문구는 `only given the information up to step t` |
| `--mask_hidden` | Map Configuration에서 Spot 2의 트럭 정체를 숨김 (Spot 2가 보일 때만 로그로 드러남) |
| `--preview` | API 호출 없이 생성될 프롬프트만 출력 |

결과 폴더: Every-step은 새 문구(기본)일 때 `everystep_upto[_cur]`, `--legacy_wording`일 때 기존 `everystep`에 저장되어 논문 데이터와 섞이지 않습니다. End-step(normal) 문구는 바뀌지 않았습니다.

체크포인트는 작업 단위(시나리오, prefixstep은 (시나리오, t))마다 저장되므로 중간에 끊겨도 같은 명령으로 이어서 실행됩니다.


- Control experiment (공간/궤적 이해 점검) & reasoning 설정, enter:
```bash
# End-step과 같은 설정 + 전체 궤적을 주고 사실 질문 9종(11개 응답)을 물음. 78개 시나리오, 시나리오당 1회 호출
python main_experiment.py --model claude-opus-4-6 --mode control --thinking off --subjects 8
python analysis/control_eval.py --model claude-opus-4-6 --condition vanilla --variant thinking_off

# Claude thinking / effort / temperature 제어 (결과는 하위 폴더로 분리: thinking_off, thinking_off_temp1.0, effort_low ...)
python main_experiment.py --model claude-opus-4-6 --mode prefixstep --current_belief --scenarios check --thinking off
python main_experiment.py --model claude-opus-4-6 --mode prefixstep --scenarios check --thinking off --temperature 1.0
```

| Argument (main_experiment.py) | Description |
| :--- | :--- |
| `--mode control` | 문항 정의·정답·채점은 `src/control_items.py`. probe 시점은 (seed, subject, scenario)로 고정되어 CSV에 저장 |
| `--thinking` | `default`(기존과 동일: Opus adaptive / Sonnet 없음) / `on`(adaptive) / `off`(disabled). 현재 Claude만 지원 |
| `--temperature` | 기본은 기존 값(0.7, thinking 사용 시 1.0 고정). thinking off에서 1.0으로 맞출 때 `--temperature 1.0` |
| `--effort` | Claude에도 `output_config.effort`로 전달 (`low`/`medium`/`high`/`max`; `none`은 thinking off로 처리) |

참고: 원래 Map Configuration에는 격자 크기(15×5)가 없어서, 건물이 가장자리에 닿는 배치에서는 프롬프트만으로 격자 밖 우회 경로가 가능해 보입니다. control 채점은 최단거리 문항(q5, q6)을 실제 격자 기준과 경계 없는 기준 두 가지로 모두 기록합니다.


- To run the analyses, enter:
```bash
# if the target model is reference data (human, btom, truebelief, nocost, motionheuristic), you don't need condition and mode
python run_analysis.py --model btom --baseline human --type scatter
python run_analysis.py --model gpt-4o --baseline btom --condition reasoning --mode normal --type rsa
python run_analysis.py --model gemini-2.5-flash --baseline btom --condition vanilla --mode everystep --type phase
```


- To plot the scenario-level bias, enter:
```bash
python analysis/plot_bias.py --model gemini-2.5-flash --condition vanilla --n_extremes 5 --wall_x 2 --wall_width 13
```


## 5. Results (In Progress)
- Please refer to the plot images in /results