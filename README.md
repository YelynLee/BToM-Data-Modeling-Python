# BToM-Data-Modeling-Python
Post-research after the following paper [Rational quantitative attribution of beliefs, desires and percepts in human mentalizing](https://www.nature.com/articles/s41562-017-0064)

## 1. Project Overview
- Objective: Building an automated pipeline and golden dataset for validating LLM's Theory of Mind (ToM) capabilities
- Key Points:
   - Schema Flattening:
      Unnested MATLAB cell arrays and structs into a 2D tabular format
   - Integrity:
      Integrated QA routines to validate the data across the pipeline
   - Robust Pipeline:
      Designed dynamic execution paths using quality thresholds
   - Semantic Labeling:
      Applied BFS algorithm to assign semantic phase labels to coordinates

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

To run the experiments, enter:
```bash
python main_experiment.py --model gpt-4o --condition oneshot --mode normal --subjects 16
```

To run the analyses, enter:
```bash
# if the target model is reference data (human, btom, truebelief, nocost, motionheuristic), you don't need condition and mode
python run_analysis.py --model btom --baseline human --type scatter
python run_analysis.py --model gpt-4o --baseline btom --condition reasoning --mode normal --type rsa
```

## 5. Results (In Progress)
- refer to the plot images in /results