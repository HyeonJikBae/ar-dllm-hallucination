# TriviaQA 5-way factual recall 실험

동일한 TriviaQA fact마다 **원본 질문 1개 + 의미가 같은 paraphrase 4개**를 먼저 고정한 뒤, 그 5개 질문을 Dream과 Qwen2.5에 똑같이 입력한다. 모델별 정답 수를 정확히 `0/5`부터 `5/5`까지 fact 단위로 집계한다. 기존 LLaDA backend도 선택적으로 남겨 두었다.

기본 비교쌍은 다음과 같다.

- dLLM: `Dream-org/Dream-v0-Base-7B`
- AR LM: `Qwen/Qwen2.5-7B`

Dream은 Qwen2.5-7B에서 초기화된 모델이어서 크기·계보가 잘 맞는 control이다. 다만 Dream에는 이후 별도의 대규모 diffusion pretraining이 추가됐으므로 결과를 “architecture만의 순수 인과 효과”가 아니라 **paired-lineage end-to-end recall robustness 비교**로 해석한다.

## 핵심 정의

| 정답 수 | fact class | 의미 |
|---:|---|---|
| 0/5 | `NEVER_RECALLED` | 현재 5개 prompt와 decoding에서 한 번도 recall되지 않음 |
| 1/5 | `PROMPT_SENSITIVE_1` | 질문 표현에 따라 1번만 recall |
| 2/5 | `PROMPT_SENSITIVE_2` | 질문 표현에 따라 2번 recall |
| 3/5 | `PROMPT_SENSITIVE_3` | 질문 표현에 따라 3번 recall |
| 4/5 | `PROMPT_SENSITIVE_4` | 질문 표현에 따라 4번 recall |
| 5/5 | `STABLE_RECALL` | 다섯 표현에서 모두 안정적으로 recall |

`1/5`~`4/5`가 현재 실험의 prompt-sensitive hallucination 후보이다. `0/5`는 모델 내부에 지식이 없다고 확정하는 표현이 아니라, **이 조건에서 recall이 관측되지 않았다**는 operational label이다. 이 다섯 질문도 한 paraphrase generator가 만든 한 표본이므로 `k/5`를 모든 질문 표현에 대한 보편적 확률로 해석하지 않는다.

Refusal은 항상 오답으로 채점하면서 `is_refusal`, `refusal_rule`을 별도로 남긴다. 아직 오답의 세부 원인 분류는 하지 않는다.

## 입력 데이터

기본 입력은 이미 로컬에 있는 다음 20개 batch, 9,961개 validation 질문이다. 이 중 정답 alias가 모두 질문 선택지에 노출된 2개는 unaided factual recall을 측정할 수 없어 manifest 생성 시 제외한다. 따라서 전체 실험 가능 문항은 9,959개다.

```text
/NAS/minyeol/sae_entities/dataset/processed/triviaqa/gemma-2b-it/batch_*.json
```

기존 Gemma 생성·label은 사용하지 않고 다음 필드만 가져온다.

- `question_id`
- `question`
- `correct_answer`의 모든 alias
- `split`

`sae_entities` 전처리는 TriviaQA 원본의 대표 정답인 `answer.value`를 버리고 `answer.aliases`만 저장한다. 공식 alias에는 유효한 표기 변형뿐 아니라 동명이인·관련 Wikipedia 제목도 섞인 사례가 확인되므로, 고정 revision의 원본 `rc.nocontext`에서 `answer.value`를 다시 정확히 join한다. 대표 정답 기반 점수를 보수적 primary로, 전체 공식 alias 기반 점수를 sensitivity 분석으로 함께 저장한다. 두 판정이 다르면 자동으로 `alias_only_candidates.jsonl`에 남아 실제 유효 별칭인지 후속 확인할 수 있다.

원본에는 서로 다른 두 질문이 `question_id=bb_956`을 공유한다. 따라서 이 코드는 `question_id + sha256(question)[:12]`로 안정적인 `fact_id`를 만들며 원래 ID도 보존한다.

## 설치와 환경 변수

이 서버에서는 실제 GPU smoke를 통과한 Python 3.10 환경을 사용한다.

```bash
cd /NAS/minyeol/ar-dllm-hallucination
source /opt/anaconda3/2024.10-1/etc/profile.d/conda.sh
conda activate ERArag310
pip install -r TrivaQA_ex/requirements.txt
```

검증한 핵심 버전은 Python 3.10.19, PyTorch 2.6.0+cu124, Transformers 4.49.0, Accelerate 1.4.0이다. 모델은 CPU에 전체 copy를 만든 뒤 옮기지 않고 공식 safetensors shard를 지정 GPU로 직접 적재한다. 이 서버의 strict virtual-memory overcommit 조건에서 연속 shard mmap이 실패하는 문제도 피한다. 다른 서버에서는 CUDA driver에 맞는 PyTorch wheel을 먼저 설치해야 한다. 활성 기본 환경 `minyeol_base`는 현재 torch가 없으므로 모델 실행에 사용하지 않는다.

루트 `.env`에는 API key를 둔다. paraphrase 모델 기본값은 사용자 선택에 따라 `gpt-4o-mini`로 고정했다. 필요할 때만 두 번째 변수나 `--model`로 override한다. `.env`는 Git에서 제외되어 있으며 API key를 결과나 config에 기록하지 않는다.

```dotenv
OPENAI_API_KEY=...
OPENAI_PARAPHRASE_MODEL=gpt-4o-mini  # optional override
```

## 1. TriviaQA fact 고정

기본값은 실수로 전체 API 비용을 발생시키지 않도록 100개만 reproducible sampling한다.

```bash
python TrivaQA_ex/prepare_triviaqa.py \
  --num-facts 100 \
  --seed 42
```

전체 scoreable 9,959개를 고정하려면 `--num-facts 0`을 사용한다. 기존 manifest를 바꿀 때만 `--overwrite`를 명시한다.

이어서 원본 TriviaQA 대표 정답을 복구한다. 기본 dataset revision은 코드에 고정되어 있고, 이 단계에서도 질문과 ID가 모두 정확히 일치하지 않으면 중단한다.

```bash
python TrivaQA_ex/enrich_canonical_answers.py --overwrite
```

이미 받은 Hugging Face validation Arrow가 있으면 네트워크 대신 `--arrow-file <path>`를 지정할 수 있다.

## 2. 원본 + paraphrase 4개 생성

```bash
python TrivaQA_ex/generate_paraphrases.py \
  --resume
```

OpenAI Responses API의 JSON Schema Structured Outputs로 정확히 4개 문자열을 받는다. 현재 고정된 paraphrase protocol은 `triviaqa_same_answer_v2`이며 각 문자열의 끝 물음표까지 schema로 강제한다. 정답 alias는 OpenAI에 보내지 않고 원본 질문만 보낸다. 응답은 `store=False`로 요청하며 fact 하나가 끝날 때마다 append·flush하므로 중단 후 `--resume`할 수 있다.

자동 검사는 개수·중복·형식·새 gold 문자열 노출 warning까지만 담당한다. 생성 직후 각 row는 `review_status="unreviewed"`이다. **동일한 답을 요구하는지 수동으로 확인한 뒤** 승인된 row를 `review_status="approved"`로 바꿔야 전체 실행이 열린다. `--allow-unreviewed`는 1-fact smoke 전용 escape hatch이다.

현재 고정된 100개 세트는 전수 의미 검토했으며, 답 대상 변경·방향 반전·모호성이 있던 8개 fact의 19개 variant를 다음 재현 가능한 review 단계에서 교정했다.

```bash
python TrivaQA_ex/review_paraphrases.py --overwrite
```

## 3. 먼저 1 fact GPU smoke test

기본값은 Dream-v0-Base-7B와 Qwen2.5-7B Base이다. 첫 실행은 checkpoint 다운로드와 native generation 호환성 확인을 위해 1 fact만 돌린다.

```bash
CUDA_VISIBLE_DEVICES=0 python -u TrivaQA_ex/run_experiment.py \
  --input TrivaQA_ex/data/triviaqa_questions_5.jsonl \
  --output-dir TrivaQA_ex/results/smoke_1fact \
  --models dream qwen \
  --limit 1 \
  --max-new-tokens 32 \
  --allow-unreviewed
```

Qwen Instruct checkpoint를 별도 조건으로 실험할 때만 `--qwen-use-chat-template`을 추가한다. 이 옵션은 Dream/LLaDA와 같은 comparison run에서 사용할 수 없도록 막혀 있다. Base-vs-Base 결과와 instruction-tuned 결과를 섞지 않는다.

순차 실행에서는 두 모델이 한 번에 하나씩 GPU에 올라간다. 여러 GPU를 지정하면 모델 worker를 동시에 실행하고, 모델 수보다 GPU가 많으면 추가 GPU를 Dream, Qwen 순서로 round-robin 배정한다. 한 fact의 질문 5개는 반드시 같은 shard에 남는다. 아래 4-GPU 설정은 Dream과 Qwen을 각각 두 개의 fact shard로 나눠 동시에 실행한다.

```bash
python -u TrivaQA_ex/run_experiment.py \
  --input TrivaQA_ex/data/triviaqa_questions_5.jsonl \
  --output-dir TrivaQA_ex/results/dream_qwen2_5_full \
  --models dream qwen \
  --gpu-ids 0 1 2 3 \
  --max-new-tokens 32
```

parent process가 config를 한 번 만든 뒤 shard별 worker가 서로 다른 cache에 기록한다. 이 서버의 strict host-memory commit 한도를 넘지 않도록 checkpoint 적재 구간만 lock으로 한 번에 하나씩 수행하고, 적재가 끝난 모델들의 생성은 GPU별로 병렬 실행한다. 모든 worker가 성공하면 task 중복·누락과 resolved checkpoint revision을 검증하고, 모델별 canonical cache를 원자적으로 만든 뒤 한 번만 채점한다. paraphrase별 generation seed도 `seed + model + fact_id + variant_id`에서 고정하므로 shard 수나 resume 순서가 달라도 seed는 변하지 않는다.

```text
<output-dir>/cache/dream_generations.jsonl
<output-dir>/cache/qwen_generations.jsonl
```

같은 명령에 `--resume`을 추가하면 완료된 `(fact_id, variant_id)`를 건너뛴다. input hash, 선택 fact, 모델 ID·revision, prompt mode, token 수, shard 수, generation seed가 기존 run과 다르면 잘못된 혼합을 막기 위해 중단한다. 물리 GPU 번호 자체는 결과 조건이 아니므로 같은 shard 수를 유지하는 한 다른 번호로 resume할 수 있다.

## 4. GPU 없이 재채점

모델 실행이 끝나면 자동 실행된다. 채점 기준을 수정한 뒤 독립적으로 다시 실행할 수도 있다.

```bash
python TrivaQA_ex/score_results.py \
  --questions TrivaQA_ex/data/triviaqa_questions_5.jsonl \
  --cache-dir TrivaQA_ex/results/smoke_1fact/cache \
  --output-dir TrivaQA_ex/results/smoke_1fact/analysis \
  --models dream qwen \
  --limit 1
```

Primary correctness는 TriviaQA `answer.value` 하나를 기준으로 하며 fuzzy ratio를 쓰지 않는다. 맨 앞의 `Answer:`/`Final answer:`를 제거한 뒤 다음 경우만 인정한다.

1. 대표 정답과 normalized exact match
2. 설명형 응답에서는 질문 자체에 이미 노출되지 않았고 부정문 근처가 아닌 whole-token 대표 정답 span
3. 설명형 응답의 `poker→pokers`, `lily→lilies` 같은 단순 규칙적 영어 복수형 span

동일한 matcher를 전체 공식 alias에도 별도로 적용해 `official_alias_*` 필드와 fact bin을 sensitivity 결과로 남긴다. 대표 정답에는 실패했지만 공식 alias에는 성공한 출력은 자동 정답 확정이 아니라 검토 후보이다. 질문에 이미 whole-token으로 노출된 문자열은 exact match에서도 제외한다. 이는 질문 속 entity/선택지를 그대로 복사해 맞는 것을 차단한다. `strict_is_correct`와 `strict_fact_bin`도 항상 함께 저장하므로 2·3번 규칙이 bin을 얼마나 바꿨는지 확인할 수 있다. 대소문자·Unicode punctuation·영문 관사·연속 공백은 정규화한다. 임의 substring/fuzzy match나 일반 stemming은 쓰지 않는다.

현재 고정 run의 alias-only 후보 감사는 다음처럼 재현한다. 이 스크립트는 현재 확인한 **정확한 normalized output**만 승인하므로, 새로운 후보가 생기면 자동 승인하지 않고 중단한다.

```bash
python TrivaQA_ex/audit_alias_candidates.py
```

Dream/LLaDA의 고정 길이 diffusion 영역에서는 EOS 뒤 토큰이 생길 수 있다. `continuation_token_ids`와 untrimmed `raw_output`은 보존하면서, 모든 모델을 첫 EOS 이전 `scored_output`으로 동일하게 채점한다.

주요 결과:

```text
analysis/
├── variant_results.jsonl       # 모든 모델 × 모든 질문 출력과 판정
├── fact_results.jsonl          # 모델별 정확한 0/5~5/5 집계
├── wrong_outputs.jsonl         # 자동 primary인 대표 정답 기준 오답(refusal 포함)
├── official_alias_wrong_outputs.jsonl # 전체 공식 alias 기준 오답
├── adjudicated_wrong_outputs.jsonl # alias 후보 검토 후 확정한 이번 run의 오답
├── alias_only_candidates.jsonl # 대표 정답 실패/공식 alias 성공 검토 후보
├── summary.json                # bin 및 refusal 통계
├── dream_x_qwen_bins.csv       # 동일 fact의 두 모델 직접 비교
└── missing_generations.jsonl   # 누락 출력; 누락은 절대 0/5로 세지 않음
```

## 테스트

API나 GPU 없이 실행된다. 현재 34개 unit test가 데이터 고정, structured paraphrase 요청, EOS 처리, 3/4-GPU sharding과 merge, canonical/official-alias 이중 판정, 0/5~5/5 채점과 비교 분류를 검사한다.

```bash
python -m unittest discover -s TrivaQA_ex/tests -v
```
