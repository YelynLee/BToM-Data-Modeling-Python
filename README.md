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


- Stepwise experiments (Prefix-step), enter:
```bash
# 프롬프트만 미리 확인 (API 호출 없음)
python main_experiment.py --model claude-opus-4-6 --mode prefixstep --scenario_ids 1 --thinking off --preview

# Prefix-step: (scenario, t)마다 로그를 1..t로 잘라 End-step 질문을 그대로 던짐 -> .../prefixstep/thinking_off/
#   Batch API(50% 할인, 요청마다 독립 처리). 중단돼도 같은 명령으로 다시 실행하면 이어서 진행
python main_experiment.py --model claude-opus-4-6 --mode prefixstep --scenarios goback --subjects 5 --thinking off --batch
python main_experiment.py --model claude-opus-4-6 --mode prefixstep --scenarios rational --subjects 5 --thinking off --batch  # 73개로 확장

# 2x2 (로그 전체/잘림 x 호출당 판단 하나/여러 개): Every-step-single, 같은 시점에 Every-step 재실행(토큰 기록)
python main_experiment.py --model claude-opus-4-6 --mode singlestep --scenarios all --subjects 5 --batch
python main_experiment.py --model claude-opus-4-6 --mode everystep --scenarios all --subjects 5 --batch --tag rerun
# 마지막 스텝 비교 / 토큰 요약
python analysis/final_step_compare.py --model claude-opus-4-6 --input <everystep> <singlestep> <prefixstep> --labels ... --out <csv>
python analysis/token_summary.py --input <everystep/rerun> <singlestep> <prefixstep> --labels ... --out <csv>

# 후처리 / 분석
python src/data_processor.py --model claude-opus-4-6 --condition vanilla --mode prefixstep
python analysis/compare_stepwise_bootstrap.py --input <everystep_valid_only.csv> <prefixstep 폴더> --labels "Every-step" "Prefix-step" --groups 1,4 --out_dir <출력 폴더>
```

| Argument (main_experiment.py) | Description |
| :--- | :--- |
| `--mode singlestep` | Every-step-single: (scenario, t)마다 독립 호출, 로그는 전체, Every-step 질문을 시점 t 하나에만 물음 (정보는 Every-step과 같고 형식은 Prefix-step과 같음) |
| `--truncate_log` | singlestep 전용: 질문 문구는 그대로, 로그만 1..t로 자름 (`singlestep_trunc`). singlestep과 비교하면 정보 접근 효과만 분리 |
| `--tag NAME` | 결과 폴더 끝에 하위 폴더 추가 (예: `--mode everystep --tag rerun` -> `everystep/rerun`), 기존 결과와 분리 |
| `--mode prefixstep` | (scenario, t)마다 독립 API 호출. 로그는 Time Step 1..t, Map Configuration은 전체 공개. 질문은 End-step과 동일 |
| `--scenarios` | `all`(78) / `rational`(irrational 제외 73, 논문 분석 범위) / `check`(55) / `goback`(Check-GoBack 22) |
| `--scenario_ids 1,6,40` | 지정한 시나리오만 (`--scenarios`보다 우선) |
| `--batch` | Batch API로 실행 (Claude / OpenAI / Gemini, 50% 할인). `batch_state.json`에 batch 정보를 저장해 재실행 시 재제출 없이 결과를 가져오고, 실패한 요청만 다시 제출 |
| `--workers N` | Batch가 없는 모델(DeepSeek 등)용: 단건 호출을 N개씩 동시에 보냄. 응답마다 저장되므로 중단 후 이어서 실행 가능 |
| `--new_wording` | (실험용) everystep / prefixstep에 `only given the information up to step t` 문구 사용 → `*_upto` 폴더. 기본은 논문과 같은 문구 |
| `--current_belief` | (부차적) initial belief와 함께 "At the LAST time step, ... being in Spot 2" 질문 추가. CSV에 `belief_now_*` |
| `--belief_order` | `--current_belief`일 때 두 질문의 순서. `random`(기본, 시드 고정) / `initial_first` / `now_first` |
| `--mask_hidden` | Map Configuration에서 Spot 2의 트럭 정체를 숨김 |
| `--preview` | API 호출 없이 생성될 프롬프트만 출력 |

결과 폴더: 논문과 같은 문구가 기본이며 `everystep/`, `prefixstep/`에 저장됩니다. `--new_wording` 결과는 `*_upto/`로 분리됩니다.
체크포인트는 작업 단위(시나리오, prefixstep은 (시나리오, t))마다 저장되므로 중간에 끊겨도 같은 명령으로 이어서 실행됩니다.

- 다른 모델 / 응답 원문·토큰, enter:
```bash
# 같은 요청 내용(src/llm_requests.py)을 단건 호출과 Batch가 함께 사용
python main_experiment.py --model gpt-5.4 --mode prefixstep --scenarios all --subjects 5 --effort low --batch
python main_experiment.py --model gemini-2.5-pro --mode prefixstep --scenarios all --subjects 5 --batch
python main_experiment.py --model deepseek-reasoner --mode prefixstep --scenarios all --subjects 5 --workers 8

# 이미 끝난 Batch의 응답 원문·토큰을 다시 받아 CSV에 채움 (재호출 없음, 점수는 그대로)
#   보관 기간: Claude 29일, OpenAI 30일, Gemini 6주
python -m src.fetch_batch_responses --dir results/claude-opus-4-6/vanilla/prefixstep/thinking_off

# 응답 원문을 그룹 × phase로 층화 추출해 읽기 (+ Spot 2를 보기 전에 실제 내용을 언급하는지 휴리스틱 표시)
python analysis/sample_responses.py --dir results/claude-opus-4-6/vanilla/prefixstep/thinking_off --n 2
```
새로 저장되는 열: `response_text`(응답 원문, Every-step은 시나리오의 첫 행에만), `n_judgements`(한 응답이 답한 스텝 수),
`input_tokens`, `output_tokens`, `reasoning_tokens`, `thinking_chars`.
판단 하나당 출력량은 `output_tokens / n_judgements`로 비교합니다.
토큰 정의는 제공사마다 다릅니다: OpenAI·DeepSeek·Claude의 `output_tokens`는 추론 토큰을 포함하고, Gemini의 `output_tokens`(candidates)는 포함하지 않으므로 Gemini는 `output_tokens + reasoning_tokens`가 전체 출력입니다.
Gemini 2.5의 단건 호출은 논문과 같이 OpenAI 호환 엔드포인트를, Batch는 네이티브 엔드포인트를 사용합니다(같은 system instruction·temperature·JSON 출력).


- Control experiment (공간/궤적·질문 이해 점검) & reasoning 설정, enter:
```bash
# End-step과 같은 설정 + 전체 궤적을 주고 사실 질문 13종(16개 응답)을 물음. 시나리오당 1회 호출
python main_experiment.py --model claude-opus-4-6 --mode control --scenarios all --thinking off --subjects 8 --batch
python analysis/control_eval.py --model claude-opus-4-6 --condition vanilla --variant thinking_off
```

| Argument (main_experiment.py) | Description |
| :--- | :--- |
| `--mode control` | 문항 정의·정답·채점은 `src/control_items.py`. probe 시점과 보기 순서는 (seed, subject, scenario)로 고정되어 CSV에 저장 |
| `--thinking` | `default`(추론 기능 끔: Opus는 `off`와 같고 결과는 `thinking_off` 폴더 / Sonnet은 thinking 파라미터 없음) / `on`(adaptive, 별도 옵션, `thinking_on` 폴더) / `off`(disabled). 현재 Claude만 지원. ※ 2026-10-28 이전 Opus `default`는 adaptive thinking이었음: 논문 End-step·Every-step Opus 결과와 그 이전의 하위 폴더 없는 Opus 결과는 adaptive. 같은 폴더에 다른 thinking 설정 결과가 있으면 실행을 멈춤 |
| `python -m src.cancel_batch --dir <폴더>` | 진행 중인 batch 취소 (처리된 요청만 청구, 결과는 남음) |
| `--temperature` | 기본은 기존 값(0.7, thinking 사용 시 1.0 고정). thinking off에서 1.0으로 맞출 때 `--temperature 1.0` |
| `--effort` | Claude에도 `output_config.effort`로 전달 (`low`/`medium`/`high`/`max`; `none`은 thinking off로 처리) |

control 문항 (v3): q1/q2 FIRST/LAST 좌표, q3 t=1에 가려진 spot, q4 LAST 위치(범주), q5 두 시점의 Spot 2 가시 여부,
q6 Spot 2를 처음 본 스텝, q7/q8 시작점에서 Spot 1/2까지 최단 이동, q9 전체 이동, q10 제자리 스텝, q11 특정 시점 거리 변화,
q12 질문 해석 객관식 (a: End-step·Prefix 문구, b: Every-step 문구; 선택지 b = current 해석, c = hindsight 해석).
이전 9문항·13문항 결과도 `control_eval.py`가 현재 번호로 변환해 채점합니다.


- Statistics (리뷰어 대응), enter:
```bash
# End-step: 사람 신뢰도(split-half, 상한선), 레퍼런스·LLM별 r(사람) + CI, LLM - BToM 차이(paired bootstrap, Williams, Holm),
#           Better - Worse, End vs Every 최종 스텝 (pooled / 모델 평균 두 방식). 기본 73개 시나리오
python analysis/endstep_stats.py --condition vanilla --out_dir results/stats

# Stepwise 조건 비교: rm_corr, VPA(고유 R² + β), hindsight 지수(전체/구간별), |LLM - BToM|
#   2단계 bootstrap(시나리오 -> 피험자) CI, 첫 조건 대비 paired 차이 CI / p, 시나리오 단위 sign-flip 순열 검정,
#   --match_subjects k: 첫 조건에서 k명씩 뽑아 피험자 수 차이의 영향 확인
#   전체 VPA 좌우 비교 그림: stepwise_vpa_full.png (desire / belief, 고유 R² + β)
#   --phase_vpa: (보조) See G2 이후 구간별 VPA (Check 그룹 1,2,4) -> stepwise_phase_vpa.png(고유 R²), stepwise_phase_vpa_simple.png(단독 R²)
python analysis/compare_stepwise_bootstrap.py \
  --input results/claude-opus-4-6/vanilla/everystep/everystep_valid_only.csv results/claude-opus-4-6/vanilla/prefixstep/thinking_off \
  --labels "Every-step" "Prefix-step" --groups 1,4 --match_subjects 5 --out_dir results/claude-opus-4-6/vanilla/compare

# Control: 문항별 정확도·q12 해석 선택 비율의 CI, 시나리오별 사실 정확도 vs End-step ToM 오차 상관
python analysis/control_eval.py --model claude-opus-4-6 --variant thinking_off --tom_link
```
공통 함수(층화 시나리오 bootstrap, Williams/Steiger, sign-flip, Holm, split-half)는 `analysis/stats_utils.py`.

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