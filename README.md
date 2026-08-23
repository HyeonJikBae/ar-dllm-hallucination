# PopQA AR vs. Diffusion LM 앎/모름 스크리닝

이 프로젝트는 동일한 PopQA 사실에 대해 autoregressive LM과 diffusion LM의 factual recall을 비교한다.

사용 모델:

- AR: `meta-llama/Llama-3.1-8B`
- dLLM: `GSAI-ML/LLaDA-8B-Base`
- dLLM: `Dream-org/Dream-v0-Base-7B`

모두 base checkpoint를 사용한다. 세 모델은 각각의 공식 generation 방식을 유지하며 순차적으로 GPU에 로드된다. 각 생성 결과는 append-only cache에 즉시 저장되므로 중단된 실행을 이어서 진행할 수 있다.

## 실험 조건

두 가지 prompt 형식을 비교한다.

### Prompt X: completion

base model에 자연스러운 평서문 completion 형식이다.

```text
The capital of Hood County is
```

### Prompt O: instructed question

질문 앞에 짧은 답만 요구하는 공통 지시문을 추가한다.

```text
Answer the following question with only the short factual answer. Do not provide any explanation.

Question: What is the capital of Hood County?
Answer:
```

각 PopQA fact에는 의미가 동일한 relation-aware paraphrase 5개를 사용한다.

Fact-level knowledge label:

- `KNOWN`: 5개 중 4개 이상 정답
- `UNKNOWN`: 5개 모두 오답
- `UNCERTAIN`: 1~3개 정답

`UNKNOWN`은 모델 파라미터에 지식이 없다는 뜻이 아니다. 현재 prompt와 decoding 조건에서 정답 recall이 관찰되지 않았다는 operational label이다.

## 파일 구조

```text
model_generation.py          모델 로딩 및 모델별 native generation
popqa_utils.py               공통 PopQA 처리 및 정답 판정 유틸리티
run_prompt_x_experiment.py   Prompt X 평서문 실험
run_prompt_o_experiment.py   Prompt O 지시형 실험
analyze_prompt_x_results.py  Prompt X 결과 재분석
analyze_prompt_o_results.py  Prompt O 결과 재채점 및 분석
requirements.txt             Python 의존성
```

`run_prompt_x_experiment.py`와 `run_prompt_o_experiment.py`는 동일한 sampling, cache 및 모델 backend를 공유한다. 따라서 prompt 형식 외의 조건을 가능한 한 동일하게 유지한다.

## 환경 설정

재현을 위해 당시의 Python 및 패키지 버전을 `requirements.txt`에 고정하였다.

```text
Python 3.10.20
torch 2.9.1+cu130
transformers 4.55.4
datasets 2.14.6
huggingface-hub 0.36.2
accelerate 1.10.1
safetensors 0.8.0
numpy 1.26.4
pyarrow 12.0.1
tokenizers 0.21.4
tqdm 4.68.4
regex 2026.6.28
```

Python 3.10.20 환경을 만든 뒤 설치한다. `torch==2.9.1+cu130`은 실제 실험
환경의 CUDA 빌드까지 포함한 버전이므로, 새 서버에서는 해당 wheel을 제공하는
PyTorch package index가 필요하다. 서버가 CUDA 13.0을 지원하지 않는다면 서버
환경에 맞는 PyTorch wheel을 먼저 설치하고, 나머지 고정 버전을 설치한다.

```bash
pip install -r requirements.txt
```

Llama checkpoint는 gated model이다. Meta의 사용 조건에 동의한 후 Hugging Face 로그인이 필요하다.

```bash
hf auth login
```

GPU 번호는 코드에서 지정하지 않는다. `CUDA_VISIBLE_DEVICES`를 이용해 외부에서 선택한다.

```bash
CUDA_VISIBLE_DEVICES=1 python model_generation.py
```

물리 GPU 1만 노출하면 프로세스 내부에서는 해당 GPU가 논리적 `cuda:0`으로 표시되는 것이 정상이다.

## 1. 모델 로딩 smoke test

전체 실험 전에 세 모델이 정상적으로 로드되고 각 모델의 공식 generation interface가 작동하는지 확인한다.

```bash
CUDA_VISIBLE_DEVICES=1 python -u model_generation.py
```

모델은 한 번에 하나씩 로드된다.

- Llama: greedy autoregressive decoding
- LLaDA: mask-token 기반 denoising sampler
- Dream: native `diffusion_generate`

각 모델 실행 후 객체를 삭제하고 CUDA cache를 비운 뒤 다음 모델을 로드한다.

## 2. 소규모 end-to-end test

먼저 Prompt O 조건을 50개 fact로 확인하는 것을 권장한다.

```bash
CUDA_VISIBLE_DEVICES=1 python -u run_prompt_o_experiment.py \
  --num-samples 50 \
  --seed 42 \
  --max-new-tokens 32 \
  --known-min-correct 4 \
  --unknown-max-correct 0 \
  --output-dir results/prompt_o_question_smoke
```

## 3. 전체 2,000개 실행

### Prompt X

```bash
CUDA_VISIBLE_DEVICES=1 nohup python -u run_prompt_x_experiment.py \
  --num-samples 2000 \
  --seed 42 \
  --max-new-tokens 32 \
  --known-min-correct 4 \
  --unknown-max-correct 0 \
  --output-dir results/prompt_x_completion_2k \
  > prompt_x_experiment_2k.log 2>&1 &
```

### Prompt O

```bash
CUDA_VISIBLE_DEVICES=1 nohup python -u run_prompt_o_experiment.py \
  --num-samples 2000 \
  --seed 42 \
  --max-new-tokens 32 \
  --known-min-correct 4 \
  --unknown-max-correct 0 \
  --output-dir results/prompt_o_question_2k \
  > prompt_o_experiment_2k.log 2>&1 &
```

각 조건은 다음과 같이 모델 출력 30,000개를 생성한다.

```text
2,000 facts × 5 prompts × 3 models = 30,000 outputs
```

다른 완료된 실험과 정확히 동일한 fact를 재사용하려면 다음 옵션을 추가한다.

```bash
--sample-source-results path/to/popqa_knowledge_results.jsonl
```

두 prompt 조건을 직접 비교할 때는 반드시 동일한 source result를 사용해야 한다.

## 중단된 실행 재개

처음 실행한 명령과 동일한 인자를 사용하고 `--resume`만 추가한다.

```bash
CUDA_VISIBLE_DEVICES=1 nohup python -u run_prompt_o_experiment.py \
  --num-samples 2000 \
  --seed 42 \
  --max-new-tokens 32 \
  --known-min-correct 4 \
  --unknown-max-correct 0 \
  --output-dir results/prompt_o_question_2k \
  --resume \
  >> prompt_o_experiment_2k.log 2>&1 &
```

동일한 output directory를 사용하는 프로세스를 동시에 여러 개 실행하면 안 된다. Hugging Face download lock이 충돌하거나 동일 cache 파일에 여러 프로세스가 기록할 수 있다.

## 4. Prompt O 결과 재분석

Prompt X 결과 분석:

```bash
python analyze_prompt_x_results.py \
  --input results/prompt_x_completion_2k/popqa_knowledge_results.jsonl \
  --output-dir results/prompt_x_completion_2k/final_analysis \
  --expected-rows 2000
```

### Prompt O 재분석

원본 pipeline은 raw generation을 그대로 보존한다. Prompt O 결과에는 다음과 같은 generation 잔여물이 나타날 수 있다.

```text
Spain.input
Gunnedahuser
Martin ScorseseQ
```

최종 분석에서는 이러한 잔여물을 분리하고 자연스러운 문장 안에 포함된 정답을 다시 인식한다.

```bash
python analyze_prompt_o_results.py \
  --input results/prompt_o_question_2k/popqa_knowledge_results.jsonl \
  --output-dir results/prompt_o_question_2k/final_analysis \
  --expected-rows 2000
```

재평가기는 다음 규칙을 적용한다.

- 알려진 role 및 prompt-boundary suffix 제거
- 자연스러운 QA 문장 안의 canonical answer 또는 안전한 alias 인식
- 부정된 정답 표현 제외
- 반복된 subject 내부에 target 문자열이 있다는 이유만으로 정답 처리하지 않음
- 긴 무관한 표현 안에 포함된 불안전한 짧은 alias 제외

이전 검토 결과에서 동일한 normalized response의 판정을 선택적으로 재사용하려면 다음 옵션을 지정할 수 있다.

```bash
--completion-classifications path/to/classified_outputs.csv
```

기본값은 비활성화되어 있으므로 Prompt O 분석은 Prompt X 결과 없이 독립적으로 실행할 수 있다.

## 결과 구조

```text
results/prompt_o_question_2k/
├── cache/
│   ├── llama_generations.jsonl
│   ├── llada_generations.jsonl
│   └── dream_generations.jsonl
├── cases/
├── popqa_knowledge_results.jsonl
├── popqa_knowledge_results.csv
├── relation_knowledge_results.csv
├── run_config.json
├── summary.txt
└── final_analysis/
    ├── classified_all_outputs_3way.csv
    ├── classified_all_outputs_3way.jsonl
    ├── corrected_fact_results.csv
    ├── strong_ar_5_0_0_candidates.csv
    ├── strong_reverse_0_4plus_4plus_candidates.csv
    └── FINAL_ANALYSIS_REPORT_KO.md
```

각 cache record는 `(fact_id, template_id)`로 구분된다. `run_config.json`은 서로 다른 설정의 실행이 기존 cache에 잘못 이어지는 것을 방지한다.

## 해석 및 수동 검토

자동 분석은 최종 knowledge claim이 아니라 screening 단계다. 논문용 contrastive case를 확정하기 전에 다음 항목을 직접 검토해야 한다.

1. canonical answer, alias 및 공동 저자와 같은 복수 정답
2. 동명이인 및 동일 지명으로 인한 subject ambiguity
3. 구체적인 factual hallucination과 format failure의 구분
4. 자동 matcher가 문장형 정답을 놓치거나 subject를 정답으로 잘못 센 사례
5. 모든 제외 사유와 raw output 보존
6. 가능하면 선정에 사용하지 않은 held-out paraphrase를 이용한 재검증

Prompt 형식에 덜 민감한 mechanistic analysis를 위해서는 Prompt X와 Prompt O 양쪽에서 동일한 knowledge pattern이 유지되는 사례를 우선적으로 사용하는 것이 좋다.
