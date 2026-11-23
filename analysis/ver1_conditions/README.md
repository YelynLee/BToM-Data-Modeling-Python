# ver1_conditions — 조건 분해 분석 (보관용)

리뷰어 대응에서 Every-step / Every-step-single / Single-trunc / Prefix-step을 서로 비교하던 스크립트입니다.

| 파일 | 내용 |
|---|---|
| condition_contrasts.py | 인접 조건 대비, sign-flip 순열 검정 + Holm |
| factor_decomposition_explore.py | 형식·정보·문구 분해 |
| truth_leak_explore.py | 참 상태 쪽 pull (Spot 2 보이기 전/후) |
| plot_trunc_figures.py, plot_reviewer_figures.py | 리뷰어 응답 그림 R0–R3 |
| plot_prefix_summary.py, plot_update_every_prefix.py | Every-step vs Prefix-step 요약 |

조건별 원자료는 `results/<model>/vanilla/{everystep,singlestep,singlestep_trunc,prefixstep,compare_*}`에 그대로 있습니다
(경로를 바꾸면 로더가 깨지므로 옮기지 않음).
