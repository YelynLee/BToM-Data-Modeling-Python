# ver0_everystep — Every-step 시기의 분석 (보관용)

ICLR 2027 제출본과 리뷰어 대응 중 Every-step 조건과 Hindsight 모델에 쓰던 스크립트입니다.
새 설계(End-step + Prefix-step)에서는 쓰지 않지만, 재현을 위해 그대로 둡니다.

| 파일 | 내용 |
|---|---|
| plot_everystep.py | Every-step phase 그림 (논문 Fig 5a 원본). `run_analysis.py`가 stepwise 모드에서 아직 호출함 |
| plot_current_vs_initial_belief.py | initial vs current belief 질문 비교 |
| compare_end_vs_everystep.py, calc_everystep_correlation.py, plot_everystep_correlation.py | End-step vs Every-step 상관 |
| stepwise_path_specific_regression.py | Every-step 경로별 회귀 |
| plot_hindsight_index_all.py | 부록 Fig 9 (Every-step hindsight index) |
| hindsight_lambda.py, plot_lambda_curves.py, lambda_sensitivity_conclusions.py, hindsight_index_runs.py | Hindsight λ 적합·민감도 (부록 Fig 10, Table 6–7) |

결과물: `results/stats/ver0_everystep/`. 공통 phase 함수는 `analysis/phase_utils.py`로 옮겼습니다.
