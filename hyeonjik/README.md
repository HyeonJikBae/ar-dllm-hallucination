# 최종 채점과 오답 유형 파이프라인 (Qwen2.5-7B, Dream-v0-Base-7B)

PopQA 사실 11,045개 x 5가지 질문 표현 = 모델당 55,225행. 정답/오답을 먼저 정하고, 오답 행에만 유형을 붙인다. 두 모델에 같은 파이프라인을 쓴다.

```
data/1_raw/             모델 출력(qwen_raw, dream_raw), PopQA 원자료, 은닉 상태
data/2_grading/         run_grading.py, 정답/오답 결과
data/final_report.html  결과 보고서
data/3_classification/  classify.py, 규칙(rules.py, groups.py, wd.py), 정책(classification_policy.md, agent_prompt_en.md), 결과
analysis/1_signal_search/   같은 사실 안에서 맞힘 대 틀림 신호 검색
analysis/2_know_but_wrong/  안다 / 아는데 틀린다 / 모른다 구분
```

## 1. 정답/오답 (`python3 data/2_grading/run_grading.py`)
이분법이다. 모든 답변은 정답 또는 오답 둘 중 하나이고, 애매한 행을 따로 빼지 않는다.

1. 답변 정리. 모델 출력에서 첫 줄/첫 문장만 쓴다. 뒤에 붙은 웹 잔해("Take the 2-minute tour" 등)와 "The film was directed by …" 같은 틀을 걷어 내고, 핵심 개체 구절만 남긴다. 정리 전 문자열과 정리 후 문자열 둘 다 후보로 쓴다.
2. 정규화. 소문자화, 구두점·관사 제거, 공백 정리.
3. 비교 대상. 골드 값과 골드의 모든 별칭(Wikidata alias).
4. 로마 숫자 검사. 군주·속편 등에 붙는 I~XII는 답변과 골드가 같아야 한다. 다르면 문자열이 비슷해도 오답이다 (예: Henry II와 Henry III).
5. 유사도. 정규화한 두 문자열이 같거나, difflib 비율 또는 토큰 정렬 비율 중 큰 값이 90 이상이면 정답.
6. 후보 중 하나라도 조건을 만족하면 정답, 아니면 오답.

결과 열: `correct`, `fact_group`(사실의 5개 표현 중 맞힌 수, 0/5~5/5), `k_correct`.

## 2. 오답 유형 (`python3 data/3_classification/classify.py qwen|dream`)
1. 문자열 규칙. 거절 정규식, 붕괴(반복) 규칙, 질문 단어 포함 여부(베낌)를 먼저 적용한다.
2. Wikidata 규칙. 골드와 답변 이름을 Wikidata 개체로 찾아 SPARQL로 관계를 본다 (@en/@mul 라벨, 캐시 `wd_cache.json`). 나라·지역·직업·작품 참여 같은 속성 연결이 있으면 1(관련), 후보는 있는데 연결이 없으면 2(무관). 큰 나라는 같은 나라 안이면 1이다.
3. 그룹 규칙. 관계별로 정의한 범주 그룹(예: 장르군, 종교군)으로 판정한다.
4. 에이전트 판정. 위 단계에서 확신이 없는 행은 정책 문서(`classification_policy.md`, `agent_prompt_en.md`)를 읽은 에이전트가 판정한다. Wikidata 결과를 힌트 열로 함께 준다. `--agents DIR`로 에이전트 출력을 넣는다.

유형: 1 관련 개체, 2 무관한 개체, 3 질문 베낌, 4 맞은 뒤 틀림, 5 지어낸 이름, 6 붕괴, abstain 거절. 3이 5보다 먼저 배정된다(답의 내용어가 전부 질문에 있으면 3).
`type3`: entity(1, 2, 4), broken(3, 5, 6), abstain.

## 은닉 상태 받기 (git에 없음)
`data/1_raw/dream_hidden_steps/`(65GB)와 `data/1_raw/qwen_hidden/`(0.3GB)는 NAS에 있다.
```
rsync -a --partial --info=progress2 /NAS/hyeonjik/hidden_states/ <repo>/hyeonjik/data/1_raw/
```
받은 뒤 분석은 repo 루트에서 `python3 hyeonjik/analysis/1_signal_search/signal_search.py dream|qwen`, `python3 hyeonjik/analysis/2_know_but_wrong/three_states.py dream|qwen broken|entity`로 실행한다. SAE 불러오는 코드(`sae_analysis/dream_sae_compare.py`, `extract_qwen_features.py`)와 SAE 가중치(`sae/`)는 repo에 있어야 한다.
