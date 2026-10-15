# Dream vs Qwen2.5: TriviaQA 5-way 결과

실행일: 2026-09-01 (재실행) / 2026-08-31 (최초)
표본: TriviaQA validation 100 facts, fact당 원본 1개 + 동의 paraphrase 4개
출력: Dream 500개 + Qwen 500개, 누락 0개, refusal 0개

## 이번 재실행에서 바뀐 것

최초 실행(32 토큰, 구 채점) 결과를 검토해 두 가지 측정 결함을 고치고 다시 돌렸다.

1. **gold 정규화** — TriviaQA gold 문자열에 붙은 표면 잡음 세 가지(직함, 괄호 선택부,
   구 형태 답의 선행 전치사)를 제거한 동치 형태를 gold에 추가했다.
   `scoring.py`의 `gold_answer_variants()` / `expand_gold_aliases()`.
   예: `Dr. (James) Parkinson` → `Parkinson`, `On a board` → `board`.
   최초 실행에서 두 모델이 10/10 출력 모두 "Parkinson"이라고 맞혔는데 전부 오답
   처리됐던 fact가 있었다. 파생 alias로 정답 처리된 건에는 `matched_derived_gold`
   플래그가 붙어 전수 감사가 가능하다.
2. **`--max-new-tokens` 32 → 64** — 최초 실행에서 답을 내기 전에 소진된 출력이
   Dream 19개, Qwen 25개였다.

## 세 조건 비교

| 조건 | Dream 정답/500 | Qwen 정답/500 | 둘 다 0/5 |
|---|---:|---:|---:|
| A. 32 토큰 · 구 채점 | 228 (45.6%) | 291 (58.2%) | 29 |
| B. 32 토큰 · gold 수정 | 235 (47.0%) | 299 (59.8%) | 27 |
| C. 64 토큰 · gold 수정 | **239 (47.8%)** | **299 (59.8%)** | **25** |

- gold 수정만으로 Dream +7, Qwen +8. bin이 바뀐 fact는 정확히 2개
  (`Dr. (James) Parkinson`, `On a board`)로, 사전에 지목한 것과 일치했다.
- 토큰 상한 확대는 **Qwen 점수를 전혀 바꾸지 않았다**(299 → 299). Qwen 출력 500개 중
  달라진 것은 32 토큰에서 잘렸던 25개뿐이고, 그 답을 끝까지 받아 보니 전부 오답이었다.
  즉 절단은 점수를 깎고 있던 게 아니라 **측정 자체를 무효로 만들고 있었다.**
- Dream은 +4. 다만 아래 주의사항을 함께 읽어야 한다.

### 주의: Dream에서 max_new_tokens는 단순 길이 상한이 아니다

`model_backends.py`는 Dream diffusion 생성에서 `generation_length`, `steps`,
`block_length`를 모두 `max_new_tokens`로 묶는다. 따라서 32 → 64 변경은 diffusion
궤적 자체를 바꾼다. 실제로 **Dream은 500개 출력이 전부 달라졌고**, Qwen은 잘렸던
25개만 달라졌다. Dream의 A → C 차이는 "절단이 줄어서"가 아니라 "디코딩 설정이
달라져서"이며, 절단 감소분과 분리되지 않는다.

절단(EOS 없이 창을 소진) 출력: Dream 19 → 4, Qwen 25 → 4.

## 최종 fact-level 결과 (조건 C)

| 모델 | 0/5 | 1/5 | 2/5 | 3/5 | 4/5 | 5/5 | 1–4/5 합계 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dream | 40 | 7 | 5 | 6 | 6 | 36 | 24 |
| Qwen2.5 | 29 | 8 | 4 | 2 | 8 | 49 | 22 |

- 질문 단위 정확도: Dream `239/500 = 47.8%`, Qwen `299/500 = 59.8%`
- 안정적 recall(5/5): Dream 36, Qwen 49
- 한 번도 recall되지 않음(0/5): Dream 40, Qwen 29
- 두 모델 모두 0/5인 fact: 25 (최초 29에서 4개 감소)
- Refusal은 두 모델 모두 `0/500`. 두 모델 다 Base 모델이라 모르는 질문에도 거절하지
  않고 오답을 생성한다. "refusal은 주로 0/5에 나온다"는 가설은 이 설정에서는
  발생 건수가 0이라 평가할 수 없다.

## 지식 수준을 통제한 비교

한쪽이 5/5로 안정적이거나 양쪽 다 2–4/5인 칸만 남기면 15개 fact가 된다.
0/5(지식 부재와 구별 불가)와 1/5(우연한 1회 적중과 구별 불가)는 제외했다.

| Dream | Qwen | fact 수 | 해석 |
|---|---|---:|---|
| 2–4/5 | 5/5 | 7 | Dream만 질문 형태에 민감 |
| 5/5 | 2–4/5 | 2 | Qwen만 질문 형태에 민감 |
| 2–4/5 | 2–4/5 | 6 | 두 모델 모두 민감 |

이 15개 fact × 5개 변형 = 75개 질문 인스턴스의 정오답 조합:

| 조합 | 개수 | 비율 |
|---|---:|---:|
| 둘 다 정답 | 47 | 63% |
| Qwen만 정답 | 16 | 21% |
| Dream만 정답 | 5 | 7% |
| 둘 다 오답 · 다른 답 | 5 | 7% |
| 둘 다 오답 · 같은 답 | 2 | 3% |

지식 수준을 통제한 뒤에도 Qwen 우세가 16 대 5로 남는다. 최초 실행(13 대 4)과
같은 방향이다.

## 질문 형태별 정확도 (조건 C)

| 모델 | 원본 v0 | v1 | v2 | v3 | v4 | paraphrase 평균 |
|---|---:|---:|---:|---:|---:|---:|
| Dream | 49% | 47% | 46% | 46% | 51% | 47.5% |
| Qwen2.5 | 66% | 58% | 58% | 58% | 59% | 58.25% |

두 모델 모두 원본 질문이 paraphrase 평균보다 높다. 따라서 1–4/5 현상에는 모델
지식의 불안정성뿐 아니라 질문 생성 분포와 원본 질문 난이도 차이도 섞여 있다.

## 채점 감사

- **파생 gold 매칭 16건** (`matched_derived_gold`): 전수 검토 결과 16건 모두 유효한
  정답이었다. `Parkinson`/`James Parkinson`(9건), `board`/`board game`/
  `A board with 30 squares…`(7건). 반대로 senet 문제의 "Egypt"는 확장 후에도
  오답으로 남아, 확장이 다른 답까지 통과시키지는 않는다.
- **canonical vs official-alias 불일치 36건**: 전부 표기·형태 변형이었다 —
  `U.S.S. MISSOURI`/`Uss Missouri`(10), `Adams Apple`/`Adam's Apple`(10),
  `Clothes`/`Clothing`(7), `Five`/`5`(5), `Theatre`/`A Theater`(3),
  `Assassination`/`Assassinated`(1). 따라서 위 표는 official alias 판정을 쓴다.
- 두 감사 모두 다른 표본이나 모델로 실행할 때 반복해야 한다. TriviaQA의 사용되지
  않은 공식 alias 전체가 항상 유효하다고 보장할 수는 없다.

## 남은 측정 결함

토큰 상한을 올려도 해결되지 않은 것이 있다. 최초 검토에서 "생성 실패"로 분류했던
4개 fact(`sfq_17511`, `qw_15839`, `qb_7289`, `odql_13057`)는 64 토큰에서도 여전히
둘 다 0/5다. 다만 성격이 바뀌었다 — 이제는 잘려서 미측정인 게 아니라, 질문을
되풀이한 뒤 실제로 틀린 답을 낸다(예: Dave Arch 관현악단 → Dream "The Tonight Show
with Jay Leno"). 즉 **측정은 성립했고 답이 틀린 것**이므로 오답 분류 대상에 넣을 수 있다.

공식 alias 목록도 가능한 모든 의미 동치 표현을 포함하지는 않는다. `Adam's rib`에
대한 `the rib of Adam` 같은 출력은 여전히 오답이다. 최종 수치는 재현 가능한
benchmark-style 점수이며 사람의 의미 판정을 대체하지 않는다.

## 실행 조건

- Dream: `Dream-org/Dream-v0-Base-7B`, resolved revision `6572adb5535263e4d1a337b56942ba48b6dee2a9`
- Qwen: `Qwen/Qwen2.5-7B` Base, resolved revision `d149729398750b98c0af14eb82c78cfe92750796`
- 동일 plain short-answer prompt, deterministic generation, 최대 64 tokens
- GPU 배치: Dream 2 shards + Qwen 2 shards, GPU 0–3 (RTX A6000)
- 런타임: torch 2.6.0+cu124, transformers 4.49.0 (`requirements.txt` 고정 버전)
- 질문 파일 SHA-256: `facae9bc2192dc4f3df3b2a86dae87b1aa2903d00b436ef8ad941d42f23b9640`

> transformers 4.49.0이 필요하다. 최신 transformers에서는 Dream의 커스텀 모델링
> 코드가 `ROPE_INIT_FUNCTIONS[self.rope_type]`에서 `KeyError: 'default'`로 죽는다.

## 결과 디렉터리

- `results/dream_qwen2_5_full_t64/` — 조건 C, 현재 기준 결과
- `results/rescore_goldfix/` — 조건 B, 최초 캐시를 새 채점기로 재채점 (gold 수정 효과 분리용)
- `results/dream_qwen2_5_full/` — 조건 A, 최초 실행 (보존)

각 `analysis/` 안의 파일:

- `summary.json`: 모델별 전체 통계
- `fact_results.jsonl`: fact별 canonical 및 official-alias 0/5–5/5
- `variant_results.jsonl`: 출력 1개당 1행, `matched_derived_gold` 포함
- `wrong_outputs.jsonl` / `official_alias_wrong_outputs.jsonl`: 기준별 오답
- `alias_only_candidates.jsonl`: 두 기준이 갈린 36건
- `dream_x_qwen_bins.csv`: 동일 fact 직접 비교
- `cache/*_generations.jsonl`: 원시 generation cache

## 해석 범위

100-fact 탐색 실험이다. 1–4/5는 "모델 내부에 지식이 있음"을 직접 증명하지 않고,
다섯 표면형 중 일부에서만 정답 recall이 관측됐다는 operational label이다. 또한
Dream은 Qwen2.5-7B에서 초기화된 뒤 별도 diffusion pretraining을 거쳤으므로 차이를
architecture 하나의 인과 효과로 해석하지 않는다.

---

# 부록: Dream denoising step / canvas 길이 sweep

실행일: 2026-09-08 · Dream만, 동일 질문 500개 · `--diffusion-steps` 플래그 신규 추가

`model_backends.py`가 `steps`를 `max_new_tokens`에 묶어 두어 길이와 스텝 수를
분리할 수 없었다. `generate()`에 `diffusion_steps` 인자와 `--diffusion-steps`
CLI 플래그를 추가해 두 축을 독립적으로 sweep했다. 기본값 0은 기존 동작
(steps = canvas 길이 = 1 token/step)을 유지한다.

## 결과

| 조건 | tok/step | 정답/500 | 최고 대비 | 상대 연산 | 실측 시간 | EOS 없음 |
|---|---:|---:|---:|---:|---:|---:|
| canvas 64 · steps 4 | 16.0 | 134 | 56.1% | 0.06× | 68s | 76 |
| canvas 64 · steps 8 | 8.0 | 137 | 57.3% | 0.12× | 94s | 64 |
| canvas 64 · steps 16 | 4.0 | 175 | 73.2% | 0.25× | 174s | 36 |
| canvas 64 · steps 32 | 2.0 | 228 | 95.4% | 0.50× | 329s | 7 |
| canvas 64 · steps 64 | 1.0 | 239 | 100.0% | 1.00× | 531s | 4 |
| **canvas 32 · steps 32** | **1.0** | **235** | **98.3%** | **0.34×** | **324s** | 19 |
| canvas 16 · steps 16 | 1.0 | 228 | 95.4% | 0.13× | 153s | 54 |

상대 연산은 `steps × (평균 프롬프트 39.2 + canvas)` 근사값이다.

## 결론 1: 스텝 수가 아니라 token/step이 결정한다

`canvas 64 · steps 32`(2 tok/step, 228)와 `canvas 32 · steps 32`(1 tok/step, 235)는
스텝 수가 같은데 후자가 더 정확하고 연산은 1/3이다. 즉 절대 스텝 수가 아니라
**한 스텝에 몇 토큰을 동시에 확정하는지**가 품질을 정한다.

1 tok/step 아래로 내려가면 급격히 무너진다. 2 tok/step에서 -4.6%, 4 tok/step에서
-23%, 8 tok/step 이하에서는 단어 경계 자체가 깨진다:

```
steps= 4: 'The nameof'
steps= 8: 'The nameof'
steps=16: 'The burning plant'
steps=32: 'The common name of the flowering plant Kniphofia is "Dragon\'s tongue."'
steps=64: 'The common name of the flowering plant Kniphofia is "Dragon\'s Blood."'
```

인접 슬롯이 서로를 보지 못한 채 동시에 확정되기 때문이다. 앞서 관찰한
부품 재조합(Dream 24.7% vs Qwen 9.1%)과 같은 메커니즘의 극단적 형태다.

## 결론 2: 이 실험의 권장 설정은 canvas 32 · steps 32

정확도 98.3%를 유지하면서 연산 0.34×, 실측 시간 39% 단축이다. 근거:

- 이 표본에서 Dream 답의 EOS 위치는 중앙값 3, p90 19 토큰이다. canvas 32면 덮인다.
- canvas 16은 짧다 — EOS 미도달 54건(11%)이고, canvas 64에서 맞았던 정답 중
  최소 7건을 절단으로 놓친다.
- canvas 64는 마지막 1.7%를 위해 연산 3배를 쓴다.

canvas 32에서 잘리는 19건은 전부 Dream이 문장형으로 길게 답한 경우다
(`The yacht that won the Royal Yacht Squadron's regatta…`). 정확도를 1%도
포기할 수 없다면 canvas 64를 쓰고, 그 외에는 32가 낫다.

## 결론 3: 질문 복잡도로 스텝을 미리 배분할 근거는 없다

- 질문 길이 ↔ 출력 길이 피어슨 **r = -0.049** (n=500). 사실상 무상관이다.
- fact 난이도별 필요 출력 길이는 역방향 신호를 준다 — `5/5`(잘 아는 fact)는
  EOS 중앙값 2 · p90 8로 짧고, `0/5`는 중앙값 3 · p90 21, `3/5`–`4/5`는 p90 40+다.
  **모델이 모를 때 오히려 길게 되풀이한다.**
- 즉 필요 길이는 질문의 사전 속성이 아니라 **출력 스타일의 사후 속성**이다.
  사전 배분이 아니라 confidence 기반 사후 조기 종료가 맞는 방향이다.

관련 연구도 모두 사후 신호를 쓴다 — CAI-dLLM은 first-step confidence로 토큰별
예산(8/32/64)을 배분하고, Prophet은 top-2 confidence gap으로 조기 확정한다
(GSM8K 97%, MMLU 99%가 절반 스텝에서 이미 정답, 최대 3.4× 스텝 감소).
우리 측정에서도 절반 스텝(canvas 32 · steps 32)이 98.3%를 유지해 방향이 일치한다.

다만 CAI-dLLM은 **short-answer retrieval이 긴 추론보다 조기 확정에 더 민감**하다고
보고하며 PIQA·Winogrande 같은 짧은 과제에서는 1.4–2.1× 가속에 그쳤다. 우리
sweep도 같은 결론이다 — 답이 3토큰이라는 것은 "3스텝이면 된다"는 뜻이 아니라
그 3토큰이 서로 강하게 의존한다는 뜻이다.

## 결과 디렉터리

- `results/step_sweep_s{4,8,16,32}/` — canvas 64 고정, steps 변화
- `results/canvas_c{32,16}/` — 1 tok/step 유지, canvas 변화
- `results/dream_qwen2_5_full_t64/` — canvas 64 · steps 64 (기준)
