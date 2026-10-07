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

# Every-step / Prefix-step에서 initial belief와 현재(current) belief를 함께 질문 -> .../everystep_cur/, .../prefixstep_cur/
python main_experiment.py --model claude-opus-4-6 --mode everystep --current_belief --scenarios check
python main_experiment.py --model claude-opus-4-6 --mode prefixstep --current_belief --belief_order now_first --scenarios check

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
| `--belief_order` | `initial_first`(기본) / `now_first`: 두 belief 질문의 제시 순서 (counterbalance) |
| `--mask_hidden` | Map Configuration에서 Spot 2의 트럭 정체를 숨김 (Spot 2가 보일 때만 로그로 드러남) |
| `--preview` | API 호출 없이 생성될 프롬프트만 출력 |

체크포인트는 작업 단위(시나리오, prefixstep은 (시나리오, t))마다 저장되므로 중간에 끊겨도 같은 명령으로 이어서 실행됩니다.


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