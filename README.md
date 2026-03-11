# BToM-Data-Modeling-Python
Post-research after the following paper [Rational quantitative attribution of beliefs, desires and percepts in human mentalizing](https://www.nature.com/articles/s41562-017-0064)

## 1. Project Overview
- Objective: Building an automated pipeline and golden dataset for validating LLM's Theory of Mind (ToM) capabilities
- Key Points:
    - Integrity
    - Dynamic Pipeline
    - Semantic Labeling

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
- The repo is structured as follows:
```
BToM_LLM
|--main_experiment.py (the main entry point to run the experiments)
|--run_analysis.py ()
|--data ()
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
|--src ()
|   |--config.py
|   |--utils.py
|   |--dataset.py
|   |--prompts.py
|   |--api_client.py
|   |--data_processor.py
|   |--prepare_everystep.py
|   |--...
|--results ()
|   |--gpt-4o
|   |--gemini-2.5-flash
|   |--deepseek-chat
|   |--claude-opus-4-6
|   |--...
|--analysis ()
|   |--plot_bars.py
|   |--plot_scatter.py
|   |--plot_rmse_corr.py
|   |--plot_rsa.py
|   |--plot_everystep.py
|   |--find_best_beta.py
|   |--...

```

- You can use the preprocessed .pkl data, but to process from the original .csv or .mat to .pkl, enter:
```bash
python src/data_processor.py --ref_only --model truebelief --beta 9.0
python src/data_processor.py --model gpt-4o --condition vanilla --mode normal
```

To run the experiment, enter:
```bash
python main_experiment.py --model gpt-4o --condition oneshot --mode normal --subjects 16
```

To analyze, enter:
```bash
python run_analysis.py --model btom --baseline human --type scatter
python run_analysis.py --model gpt-4o --baseline btom --condition reasoning --mode normal --type rsa
```

## 5. Results (On Process)