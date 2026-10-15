# 지식 상태 및 Dream–Qwen 출력 비교 분류

대상은 검토 완료된 TriviaQA 100 facts, 500개 동일 질문쌍이다. 정답 여부는 이번 run에서 alias-only 후보 39개를 감사한 뒤 채택한 `official_alias_is_correct`를 사용한다.

## 1. Fact별 지식 상태

| 정답 수 | 분류 코드 | 실험상 해석 |
|---:|---|---|
| 0/5 | `NO_KNOWLEDGE_OBSERVED` | 지식이 관측되지 않음 |
| 1/5 | `SINGLE_HIT_CHANCE_CANDIDATE` | 우연 정답 후보 |
| 2/5–4/5 | `QUESTION_SENSITIVE_KNOWLEDGE` | 지식은 관측되지만 질문 형태에 민감 |
| 5/5 | `CONSISTENT_KNOWLEDGE` | 다섯 질문에서 일관되게 지식이 관측됨 |

| 모델 | 0/5 | 1/5 | 2/5–4/5 | 5/5 | 합계 |
|---|---:|---:|---:|---:|---:|
| Dream | 46 | 5 | 13 | 36 | 100 |
| Qwen | 31 | 8 | 12 | 49 | 100 |

이 명칭은 다섯 prompt에서 관측된 행동을 요약하는 operational label이다. 특히 `0/5`가 내부 지식의 부재를, `1/5`가 우연을 직접 증명하는 것은 아니다.

### 두 모델의 fact 상태 교차표

행은 Dream, 열은 Qwen 상태다.

| Dream \ Qwen | 0/5 | 1/5 | 2/5–4/5 | 5/5 |
|---|---:|---:|---:|---:|
| 0/5 | 29 | 4 | 4 | 9 |
| 1/5 | 0 | 2 | 3 | 0 |
| 2/5–4/5 | 2 | 1 | 3 | 7 |
| 5/5 | 0 | 1 | 2 | 33 |

## 2. 동일 질문에 대한 모델 출력 비교

“같은 형태”는 `Answer:` 제거, Unicode·대소문자·구두점·관사·공백 정규화 후 문자열이 완전히 같은 경우다. 이는 의미적 동치나 오답 원인 분류가 아니라 재현 가능한 surface-form 비교다.

| 상위 결과 | 질문쌍 수 | 비율 |
|---|---:|---:|
| 두 모델 모두 오답 | 189 | 37.8% |
| 한 모델만 정답 | 103 | 20.6% |
| 두 모델 모두 정답 | 208 | 41.6% |

| 세부 분류 | 질문쌍 수 | 비율 |
|---|---:|---:|
| 두 모델 모두 오답 · 같은 출력 형태 | 26 | 5.2% |
| 두 모델 모두 오답 · 다른 출력 형태 | 163 | 32.6% |
| Dream 정답 · Qwen 오답 | 20 | 4.0% |
| Dream 오답 · Qwen 정답 | 83 | 16.6% |
| 두 모델 모두 정답 · 같은 출력 형태 | 162 | 32.4% |
| 두 모델 모두 정답 · 다른 출력 형태 | 46 | 9.2% |

## 생성된 테이블

- `classification_tables/knowledge_status_by_fact.csv`: 모델 × fact 200행 상세
- `classification_tables/knowledge_status_summary.csv`: 모델별 네 지식 상태 집계
- `classification_tables/knowledge_status_cross_model.csv`: 동일 fact의 두 모델 상태 100행
- `classification_tables/knowledge_status_cross_matrix.csv`: 4×4 상태 교차 집계
- `classification_tables/model_pair_by_question.csv`: 동일 질문 500쌍의 출력·정답·형태 상세
- `classification_tables/model_pair_outcome_summary.csv`: 양쪽 오답/한쪽 정답/양쪽 정답 집계
- `classification_tables/model_pair_category_summary.csv`: 요청한 여섯 세부 범주 집계

재생성 명령:

```bash
python TrivaQA_ex/classify_comparisons.py
```
