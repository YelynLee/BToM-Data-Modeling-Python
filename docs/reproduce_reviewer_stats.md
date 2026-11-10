# 리뷰어 대응 통계·그림 재현 가이드

모든 명령은 저장소 루트에서 실행합니다. 결과 폴더 이름은 아래를 가정합니다 (다르면 경로만 바꾸면 됨).

| 조건 | Claude-4.6-Opus (thinking off) | Gemini-2.5-Pro (기본 설정) |
|---|---|---|
| End-step (논문) | `results/claude-opus-4-6/vanilla/model_data.pkl` | `results/gemini-2.5-pro/vanilla/model_data.pkl` |
| Every-step (논문, 16명) | `results/claude-opus-4-6/vanilla/everystep/everystep_valid_only.csv` | `results/gemini-2.5-pro/vanilla/everystep/everystep_valid_only.csv` |
| Every-step 재실행 (5명) | `results/claude-opus-4-6/vanilla/everystep/thinking_off/rerun` | `results/gemini-2.5-pro/vanilla/everystep/rerun` |
| Every-step-single | `results/claude-opus-4-6/vanilla/singlestep/thinking_off` | `results/gemini-2.5-pro/vanilla/singlestep` |
| Prefix-step | `results/claude-opus-4-6/vanilla/prefixstep/thinking_off` | `results/gemini-2.5-pro/vanilla/prefixstep` |
| Control | `results/claude-opus-4-6/vanilla/control/thinking_off` | `results/gemini-2.5-pro/vanilla/control` |

bootstrap은 모든 스크립트에서 기본이 **층화 없는 시나리오 복원추출**(논문 Table 4 방식)이고, `--stratified`를 붙이면 행동 그룹별 층화로 다시 계산합니다 (강건성 점검). 반복 횟수 2,000, seed 0이 기본이라 같은 입력이면 같은 숫자가 나옵니다.

---

## 1. End-step 통계 (통계 섹션)

```bash
python analysis/endstep_stats.py --condition vanilla --out_dir results/stats
python analysis/endstep_stats.py --condition vanilla --out_dir results/stats --stratified   # 강건성
```
출력 `results/stats/endstep_stats_vanilla.csv`의 `section` 열로 찾습니다.

| 확인할 수치 | section | name / metric |
|---|---|---|
| 사람 split-half, LOSO | `1_human_reliability` | `split_half_spearman_brown`, `leave_one_subject_out_r` |
| 사람–BToM 0.91 / 0.78 | `2_reference_vs_human` | name `btom`, `r_with_human` |
| Table 4 (LLM–사람 r, CI) | `3_llm` | `r_with_human` |
| LLM − BToM Δr, Williams, Holm | `4_llm_minus_btom_benchmark` | `p`, `p_holm`, `williams_p_holm` |
| Better − Worse | `5_better_minus_worse` | (기술 통계로만 사용) |
| End vs Every 최종 스텝 | `6_end_vs_every` | 모델별 + `better4_pooled` / `better4_model_mean` / `better4_mean_of_model_r` |
| Worse 그룹 레퍼런스 비교 (Fig 4a, Fig 7) | `7_worse_vs_references` | `worse_group_mean`, 모델별, `r(best) - r(other)` |
| End-step VPA (Fig 4b, Fig 8) | `8_endstep_vpa` | `p4_*` = 논문 예측변수 4개, `p5_*` = Hindsight 포함(belief), `* - *` = 1위 vs 2위 |
| desire/belief 순위 일치 (App E.1, 0.65) | `9_rank_agreement` | `spearman(r_desire, r_belief)` |

### belief 정규화 순서 비교 (평균 후 정규화 vs 정규화 후 평균)
```bash
python - <<'EOF'
import sys, pickle, numpy as np
sys.path[:0] = ['.', 'src', 'analysis']
from src.config import get_group_indices
from stats_utils import pooled_r
ids = sorted(s for g in get_group_indices(include_irrational=False) for s in g); c = np.array(ids) - 1
H = pickle.load(open('data/human/human_data.pkl', 'rb'))['bel_inf_mean_norm'][:, c]
for m in ['claude-opus-4-6', 'gemini-2.5-pro', 'deepseek-reasoner', 'gpt-5.4']:
    d = pickle.load(open(f'results/{m}/vanilla/model_data.pkl', 'rb'))
    A = d['bel_inf_mean_norm'][:, c]                                   # 평균 후 정규화 (논문)
    sh = np.maximum(d['bel_inf'][:, c, :] - 1, 0); s = sh.sum(0, keepdims=True); s[s == 0] = np.nan
    B = np.nanmean(sh / s, axis=2)                                     # 정규화 후 평균 (기존 그림)
    print(m, round(pooled_r(A, H), 3), round(pooled_r(B, H), 3), round(float(np.nanmean(abs(A - B))), 3))
EOF
```

---

## 2. Every-step / Every-step-single / Prefix-step (R1, 표 S1)

모델마다 아래 세 명령을 같은 출력 폴더에 실행합니다 (Opus 예시. Gemini는 경로만 바꿈).
```bash
R=results/claude-opus-4-6/vanilla; O=$R/compare_2x2
# 지표 + Every-step 대비 차이 CI + 순열 검정 + 전체 VPA 그림
python analysis/compare_stepwise_bootstrap.py \
  --input $R/everystep/thinking_off/rerun $R/singlestep/thinking_off $R/prefixstep/thinking_off \
  --labels "Every-step" "Every-step-single" "Prefix-step" --out_dir $O
# 마지막 스텝(t = T) r(사람 / BToM / End-step)
python analysis/final_step_compare.py --model claude-opus-4-6 \
  --input $R/everystep/thinking_off/rerun $R/singlestep/thinking_off $R/prefixstep/thinking_off \
  --labels "Every-step" "Every-step-single" "Prefix-step" --out $O/final_step.csv
# 판단당 생성 토큰
python analysis/token_summary.py \
  --input $R/everystep/thinking_off/rerun $R/singlestep/thinking_off $R/prefixstep/thinking_off \
  --labels "Every-step" "Every-step-single" "Prefix-step" --out $O/tokens.csv
```
- Single vs Prefix 차이의 CI가 필요하면 Single을 첫 입력으로 한 번 더: `--input <single> <prefix> --labels "Every-step-single" "Prefix-step" --out_dir $O/single_vs_prefix`
- 조건 이름(labels)은 위와 똑같이 써야 R1/S1 스크립트가 찾습니다.

R1 그림과 표 S1 (두 모델의 `compare_2x2` 폴더가 준비된 뒤):
```bash
python analysis/plot_reviewer_figures.py --models "Claude-4.6-Opus" "Gemini-2.5-Pro" \
  --compare results/claude-opus-4-6/vanilla/compare_2x2 results/gemini-2.5-pro/vanilla/compare_2x2 \
  --out_dir results/reviewer
```
출력: `R1_decomposition.png`, `S1_stepwise_table.csv` / `.md`

Figure 5a 형식 phase 그림 (End-step 점 자동):
```bash
python analysis/plot_stepwise_phase.py \
  --input $R/everystep/thinking_off/rerun $R/singlestep/thinking_off $R/prefixstep/thinking_off \
  --labels "Every-step" "Every-step-single" "Prefix-step" --score belief --groups 1,4 --out $O/phase_belief.png
```

---

## 3. Control (C1, C2)

```bash
# 문항별 정확도와 CI, 사실 정확도 vs End-step ToM 오차 (control_ci.csv 등)
python analysis/control_eval.py --model claude-opus-4-6 --condition vanilla --variant thinking_off --tom_link
python analysis/control_eval.py --model gemini-2.5-pro --condition vanilla --tom_link
# C1, C2 그림 (control_scored.csv가 있는 폴더를 넣음)
python analysis/plot_reviewer_figures.py --models "Claude-4.6-Opus" "Gemini-2.5-Pro" \
  --control results/claude-opus-4-6/vanilla/control/thinking_off results/gemini-2.5-pro/vanilla/control \
  --endstep results/claude-opus-4-6/vanilla/model_data.pkl results/gemini-2.5-pro/vanilla/model_data.pkl \
  --out_dir results/reviewer
```
출력: `C1_control_accuracy.png`, `C2_control_vs_tom_error.png` / `.csv`
- C1의 빈 마커 = 격자 경계 해석(경계 있음 / 없음) 중 어느 쪽으로든 맞으면 정답. q8은 `control_scored.csv`의 `*_correct_either`, q11은 그림 스크립트에서 다시 채점
- C1/C2는 irrational 5개를 뺀 73개 시나리오 기준이라 `control_summary.csv`(78개)와 소수점이 조금 다를 수 있음

---

## 4. Every-step VPA, 논문 Fig 5b의 모델 4개 (CI + β)

각 모델의 Every-step 결과(논문 16명)를 한 번에 넣고 `--no_paired`(모델 간 차이는 계산하지 않음):
```bash
python analysis/compare_stepwise_bootstrap.py \
  --input results/claude-opus-4-6/vanilla/everystep/everystep_valid_only.csv \
          results/gemini-2.5-pro/vanilla/everystep/everystep_valid_only.csv \
          results/deepseek-reasoner/vanilla/everystep/everystep_valid_only.csv \
          results/deepseek-v4-pro/vanilla/everystep/everystep_valid_only.csv \
  --labels "Claude-4.6-Opus" "Gemini-2.5-Pro" "DeepSeek-R1" "DeepSeek-V4-Pro" \
  --no_paired --title "Every-step condition" --out_dir results/stats/everystep_vpa
```
- `stepwise_vpa_full.png`: 열 = 모델, 위 = desire, 아래 = belief. 막대 = 고유 adjusted R² (95% CI), 숫자 = 표준화 β
- `stepwise_bootstrap_summary.csv`의 `vpa_desire_*`, `vpa_belief_*` 행에 수치
- `everystep_valid_only.csv`가 없으면 먼저: `python run_analysis.py --model deepseek-reasoner --mode everystep --type phase`
- 예측변수: desire는 BToM, TrueBelief, NoCost, MotionHeuristic (Hindsight desire = BToM이라 제외), belief는 Hindsight 포함 5개. 시나리오 73개. 논문 Fig 5b는 78개로 계산되어 소수점이 다를 수 있음

---

## 5. Single-trunc 포함 네 조건 (Opus) 과 그림 R0, R2, R3

```bash
R=results/claude-opus-4-6/vanilla; O=$R/compare_4cond
IN="$R/everystep/thinking_off/rerun $R/singlestep/thinking_off $R/singlestep_trunc/thinking_off $R/prefixstep/thinking_off"
LAB=("Every-step" "Every-step-single" "Single-trunc" "Prefix-step")
python analysis/compare_stepwise_bootstrap.py --input $IN --labels "${LAB[@]}" --out_dir $O
python analysis/final_step_compare.py --model claude-opus-4-6 --input $IN --labels "${LAB[@]}" --out $O/final_step.csv
python analysis/token_summary.py --input $IN --labels "${LAB[@]}" --out $O/tokens.csv
python analysis/truth_leak_explore.py --input $IN --labels "${LAB[@]}" --out_dir $O   # pull (Spot 2 보이기 전/후/끝내 안 보임)
# 그림: R0 설계 도식, R2 Opus 네 조건, R3 모델별 요인 분해 (+ R3_factor_decomposition.csv)
python analysis/plot_trunc_figures.py --design --r2_dir $O --r2_model "Claude Opus 4.6 (thinking off)" \
  --models "Claude Opus 4.6" "Gemini 2.5 Pro" "GPT-5.4" \
  --compare $O results/gemini-2.5-pro/vanilla/compare_2x2 results/gpt-5.4/vanilla/compare_2x2 --out_dir results/reviewer
```
- Single − Single-trunc 차이의 CI와 순열 p는 Single을 첫 입력으로 한 번 더 실행 (`--out_dir $R/compare_single3`)
- `analysis/factor_decomposition_explore.py`: t = T 재현성, run 수준 회귀(정보, 문구, log 토큰)
