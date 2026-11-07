# 최종 채점과 오답 유형 (Qwen2.5-7B, Dream-v0-Base-7B)

PopQA 사실 11,045개 x 5가지 질문 표현 = 모델당 55,225행. 정답/오답을 먼저 정하고, 오답 행에만 유형을 붙인다.
두 모델에 같은 프롬프트, 같은 규칙을 쓴다. 사람이나 LLM 판정 없이 문자열 규칙과 Wikidata 캐시만 쓰므로 다시 돌리면 같은 결과가 나온다.

```
data/0_prep/            PopQA 원본에서 사용 데이터 만들기: 원본 → 제외 목록 → 남은 사실 → 변형 질문
                        find_popqa_multi_answer_candidates.py  정답이 여러 개인 사실 후보
                        confirm_popqa_temporal_value_changes.py 값이 바뀐 사실 확인
                        generate_paraphrases_openai.py          변형 질문 4개 (gpt-4o-mini)
data/1_raw/             모델 출력(qwen_raw, dream_raw), PopQA 원자료
                        run_inference.py         생성 (두 모델)
                        dream_steps_extract.py   Dream 생성 + step별 은닉 상태 저장
                        qwen_type_extract.py     Qwen 은닉 상태 저장
                        rebuild_dream_answers.py Dream 캔버스 토큰에서 답 다시 만들기
                        dream_hidden_steps/, qwen_hidden/   은닉 상태 (NAS, git 제외)
data/2_grading/         run_grading.py → qwen_grading.jsonl, dream_grading.jsonl
data/3_classification/  classify_rules.py, wd.py, wd_cache.json → *_classification.jsonl(오답), full/*_final_full.jsonl(전체)
data/final_report.html  결과 보고서
analysis/sae_io.py           SAE 불러오기 (가중치는 repo의 sae/ 폴더)
analysis/1_signal_search/   같은 사실 안에서 맞힘 대 틀림 신호 검색
analysis/2_know_but_wrong/  안다 / 아는데 틀린다 / 모른다 구분
```

재현 순서 (생성은 이미 되어 있으므로 2번부터):
```
cd data/1_raw && python3 rebuild_dream_answers.py                # Dream 답 = 첫 특수 토큰 앞 → 첫 줄
cd ../2_grading && python3 run_grading.py                        # 정답/오답, 0/5~5/5
cd ../3_classification && python3 classify_rules.py qwen && python3 classify_rules.py dream
```
각 스크립트는 `--check`로 저장된 결과와 비교할 수 있다(차이 0이어야 함). 분석 두 개는 아직 이 라벨로 다시 돌리지 않았다(이전 결과는 지웠다).

## 0. 데이터와 생성
- 데이터 준비 코드와 파일은 `data/0_prep/`에 있다. 제외 후보를 찾는 코드(`find_…`, `confirm_…`)의 출력에서 최종 제외 목록(`2. popqa_excluded.jsonl`)을 정한 과정은 파일로만 남아 있다.
- PopQA 14,267개 사실 중 정답이 여러 개인 사실(3,203개)과 값이 시간에 따라 바뀐 사실(19개)을 빼고 11,045개를 쓴다.
- 질문 5개: PopQA 원래 질문 1개 + gpt-4o-mini로 만든 변형 4개(답의 종류, 관계, 관계 방향은 유지하고 문장 구조만 바꿈).
- 프롬프트(두 모델 공통, 예시 답 없음):
  `Answer the question with only the direct answer: a short phrase or name, no explanation, no full sentences.\n\nQuestion: {질문}\nAnswer:`
- Qwen: 탐욕 디코딩, 최대 32토큰. Dream: 캔버스 32칸, 32 step, `topk_margin`, 온도 0.
- **답 = 첫 특수 토큰(`<|endoftext|>` 등) 앞까지의 글에서, 앞쪽 공백과 줄바꿈을 지운 뒤 첫 줄.** 그 뒤의 글은 보지 않는다.
  - Qwen은 답을 쓰고 `<|endoftext|>`에서 멈춘다.
  - Dream도 답을 쓰고 `<|endoftext|>`를 두지만, 캔버스 32칸을 채워야 해서 그 뒤에 새 문서(웹 페이지 문구, 수학 문제)를 이어 쓴다. 특수 토큰을 지우고 글자로 바꾸면 답과 새 문서가 붙어 버리므로(예: "KyivTake the 2-minute tour ×") 반드시 특수 토큰에서 먼저 끊는다.
  - 줄바꿈 뒤에는 모델이 스스로 만든 다음 질문("Question: …")이 오는 경우가 대부분이다.
- Dream의 답은 저장된 캔버스 토큰에서 다시 만든다(`1_raw/rebuild_dream_answers.py`, `generation` 열에 캔버스 전체를 특수 토큰과 함께 저장). Qwen은 생성 원문을 저장하지 않아 저장된 답을 쓴다(생성은 `<|endoftext|>`에서 멈추므로 같은 규칙).
- 새로 생성할 때도 `run_inference.py`, `dream_steps_extract.py`가 같은 규칙(첫 특수 토큰에서 끊고 첫 줄)을 쓴다.

## 1. 정답/오답 (`python3 data/2_grading/run_grading.py`)
1. **첫 줄 안에 쓸데없는 글이 있으면 오답**: 웹 잔해(Take the 2-minute tour, You are here 등), 수학 문제("What is the sum", "equation", `$` 등), 띄어쓰기 없이 붙은 새 문장("Paris.What"), 프롬프트·채팅 누출("Question:", "user:" 등), 반복.
2. 아니면 답을 정규화(소문자, 악센트·기호 제거)해서 골드와 모든 별칭(Wikidata alias)과 비교한다. 로마 숫자(I~XII)가 같고, 같거나 유사도(difflib 비율 또는 단어 정렬 후 비율)가 90 이상이면 **정답**.
3. 그 밖은 **오답**. 문장에서 답을 꺼내지 않는다(문장형 답은 대부분 오답).

결과 열: `correct`, `k_correct`(사실의 5개 표현 중 맞힌 수), `fact_group`(0/5~5/5).

## 2. 오답 유형 (`python3 data/3_classification/classify_rules.py qwen|dream`)
오답 행에만, 위에서부터 처음 해당하는 것으로 정한다.

| 순서 | 조건 | type3 | rule |
|---|---|---|---|
| 1 | 빈 답 | broken | broken-empty |
| 2 | 거절 표현(unknown, not specified, I don't know 등, 14단어 이하) 또는 "no", "none" | abstain | abstain |
| 3 | 프롬프트·채팅 누출 | broken | broken-leak |
| 4 | 반복 | broken | broken-loop |
| 5 | 답 뒤에 쓸데없는 글(1번과 같은 목록) | broken | broken-continuation |
| 6 | 답이 될 수 없는 모양: 숫자만, 알파벳 한 글자, 기능어 한 단어("Me", "Yes") | broken | broken-invalid |
| 7 | 답의 내용어가 전부 질문 속 단어(질문 베낌) | broken | broken-copy |
| 8 | 질문 단어와 새 단어가 섞인 이름인데 Wikidata에 후보가 없음(아버지·어머니 관계 제외) | broken | broken-fabricated |
| 9 | 나머지(다른 실재 개체, 다른 개체를 말한 문장 등) | entity | entity |

결과 열: `type3`, `rule`(어느 규칙이 정했는지), `unverified`. `unverified`: 8번 판정에 쓸 이름이 `wd_cache.json`에 없어 확인하지 못한 행(entity로 둠). `--online`으로 조회하면 채워진다.

## 결과
| | 정답 | 오답 | entity | broken | abstain |
|---|---|---|---|---|---|
| Qwen | 10,025 (18.2%) | 45,200 | 96.8% | 2.8% | 0.5% |
| Dream | 8,040 (14.6%) | 47,185 | 84.5% | 14.5% | 1.0% |

0/5 ~ 5/5 (사실 수): Qwen 8,438 / 343 / 245 / 276 / 351 / 1,392, Dream 8,822 / 387 / 240 / 244 / 319 / 1,033.

오답 중 broken 비율 (0/5, 1/5, 2/5, 3/5, 4/5): Qwen 2.7 / 4.1 / 2.2 / 5.3 / 5.4%, Dream 14.8 / 8.2 / 11.2 / 12.9 / 12.2%.

broken 내역: Qwen 질문 베낌 1,039 / 빈 답 116 / 지어낸 이름 49 / 모양 26 / 반복 20 / 누출 4, Dream 질문 베낌 5,632 / 지어낸 이름 522 / 빈 답 480 / 모양 195 / 반복 18 / 누출 9 / 쓸데없는 글 1.

## 알려진 한계
- 정답을 담은 문장형 답("Marco Ameglio is an Italian politician.")은 오답(entity)이다. 문장에서 답을 꺼내도 늘어나는 정답은 0.1% 미만이었다.
- 쓸데없는 글은 정해 둔 목록으로만 잡는다. 특수 토큰과 줄바꿈 뒤의 글은 보지 않는다.
- 지어낸 이름(8번)은 Wikidata 캐시에 기댄다. 실재하지만 Wikidata에서 못 찾은 이름이 broken이 될 수 있다.
- 무작위 200행(모델별 정답·entity·broken·abstain 각 25행)을 Claude가 본 결과 약 98~99%가 위 정의대로 분류됐다(특수 토큰에서 끊기 전 기준). 사람 검증은 아직 하지 않았다.

## 은닉 상태 받기 (git에 없음)
`data/1_raw/dream_hidden_steps/`(65GB)와 `data/1_raw/qwen_hidden/`(0.3GB)는 NAS에 있다.
```
rsync -a --partial --info=progress2 /NAS/hyeonjik/hidden_states/ <repo>/data/1_raw/
```
분석 코드는 repo 루트에서 실행한다. SAE 가중치는 repo의 `sae/` 폴더(`dream_pilot_200M/`, `qwen2.5-7b_L20_batchtopk_16k_k100_20M/`)에 있어야 하고, 불러오는 코드는 `analysis/sae_io.py`다.
