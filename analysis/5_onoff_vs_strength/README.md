# 5. 답을 쓰기 전 상태로 정답/오답을 알 수 있는가 (Dream-v0-Base-7B)

모든 분석은 layer 20 출력의 프롬프트 마지막 토큰 상태를 쓴다(`hid_prompt_last.npy`, 저장된 step 0/8/16/24/31).
- step 0은 답 칸이 아무것도 채워지기 전 상태다.
- SAE는 16,384개 latent의 JumpReLU다(`../sae_io.py`). 이 위치에서는 1,484개만 한 번이라도 켜진다.
- 신호는 dev 주어에서 고르고 test 주어에서 평가한다(주어 기준 50/50).
- **같은 사실 안 AUROC**는 정답 행과 오답 행이 둘 다 있는 사실마다 AUROC를 구해 평균 낸 값이다(0.5 = 우연). 주어, 관계, 유명도가 같은 조건에서 표현 차이만 보는 점수다.
- 실행: repo 루트에서 `CUDA_VISIBLE_DEVICES=<gpu> python3 analysis/5_onoff_vs_strength/<script>.py`. 출력은 `log_*.txt`다.

## 결론
- **아는 사실 대 모르는 사실(5/5 대 0/5)은 잘 구분된다**: AUROC 0.88~0.94.
- **같은 사실 안 정답 대 오답은 답을 쓰기 전(step 0)에는 약하다**: 0.56~0.64. 디노이징 step 8에서는 0.75, 끝에서는 0.80까지 오른다.
- SAE를 켜짐/꺼짐으로 읽든 세기로 읽든, latent 수 K를 바꾸든, raw hidden probe보다 낫지 않다.

## 스크립트와 결과
| 스크립트 | 질문 | 결과 |
|---|---|---|
| `onoff_vs_strength.py` (`log.txt`) | SAE를 켜짐/꺼짐(논문 방식) 대 세기로 읽으면? | 5/5 대 0/5: 켜짐/꺼짐 0.888~0.913, 세기 0.880~0.935. 같은 사실 안: 0.557~0.612 |
| `steps_trajectory.py` (`log_steps.txt`) | 디노이징 step별, 그리고 step 사이 변화량은? | raw probe가 step 0에서 0.644, step 8에서 0.749, step 31에서 0.805. 변화량은 도움이 안 됨 |
| `commit_step.py`, `commit_auroc.py` | 답 첫 칸이 언제 채워지는가 | 정답 중앙값 step 2, 오답 30. 1/5~4/5 안에서는 정답 5, 오답 22. 이 값 하나로 같은 사실 안 0.561 |
| `paper_style_top1.py`, `two_signals_by_k.py` | 논문식 앎/모름 latent 하나(MaxMin) | step 0에서 앎 13781, 모름 3932. k가 커질수록 앎 latent가 켜지는 비율이 연속적으로 오름 |
| `gap_score.py` (`log_gap.txt`) | (앎 − 모름) 점수가 높은 오답은 다른 표현에서는 맞히는가 | 0/5 대 1/5 이상: 0.779, 관계 안에서 0.623 |
| `states_types.py`, `three_way.py`, `between_axis.py` | 앎 / 모름 / 아는데 틀림 3분류 | 균형 정확도 0.70~0.74(관계만 쓰면 0.62). 아는데 틀림은 앎-모름 축의 중간에 놓임 |
| `within_fact_types*.py` | 같은 사실 안 정답 대 entity, 정답 대 broken (k별, K = 1/5/20/100) | 정답 대 broken은 step 0 raw probe로 0.835. 정답 대 entity는 0.634. K=1은 약하고 K=5~20부터 평평함 |

이후 실험은 오답 유형을 나누지 않고 정답/오답으로만 본다. 유형별 방향이 거의 같기 때문이다(`../6_steering/README.md`).
