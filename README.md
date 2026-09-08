# PopQA Question Augmentation

AR `Qwen2.5-7B`와 dLLM `Dream-7B`의 factual hallucination 비교를 위한 데이터
준비 코드다. PopQA의 fact마다 정답은 유지하고 질문을 5개로 증강한 뒤,
Qwen과 Dream의 원본 출력을 모델별 JSONL로 저장한다.

## 원칙

- PopQA relation별 고정 템플릿으로 재현성을 보장한다.
- 모든 질문은 원래 subject -> object 관계 방향을 유지한다.
- canonical answer와 `possible_answers`를 함께 보존한다.
- 두 모델에서 공통으로 쓸 short-answer prompt도 각 variant에 저장한다.
- 정답 채점과 hallucination 유형 분석은 아직 포함하지 않는다.

## 실행

```bash
pip install -r requirements.txt
python augment_popqa.py --output data/popqa_augmented.jsonl
```

100개만 고정 표본으로 만들려면:

```bash
python augment_popqa.py --num-samples 100 --seed 42 \
  --output data/popqa_augmented_100.jsonl
```

Hub 대신 로컬 원본 JSONL도 사용할 수 있다.

```bash
python augment_popqa.py --input-jsonl data/popqa_raw.jsonl \
  --output data/popqa_augmented.jsonl
```

JSONL 한 줄은 fact 하나이며 `variants` 배열에 `v1`부터 `v5`까지 들어간다.

```json
{
  "schema_version": "popqa-augmentation-v1",
  "sample_id": "123",
  "subject": "Paris",
  "relation": "country",
  "object": "France",
  "answers": ["France", "French Republic"],
  "original_question": "What country is Paris in?",
  "variants": [
    {
      "variant_id": "v1",
      "template_id": "country_v1",
      "question": "In which country is Paris located?",
      "prompt_template_id": "short-answer-v1",
      "prompt": "Answer the following question with only the short factual answer. Do not provide any explanation.\n\nQuestion: In which country is Paris located?\nAnswer:"
    }
  ]
}
```

테스트는 `pytest -q`로 실행한다.

## 모델 실행

먼저 5개 fact로 확인한다.

```bash
CUDA_VISIBLE_DEVICES=0 python generate_answers.py --model qwen \
  --limit-facts 5 --max-new-tokens 12 \
  --output results/qwen_smoke.jsonl
CUDA_VISIBLE_DEVICES=1 python generate_answers.py --model dream \
  --limit-facts 5 --max-new-tokens 12 --dream-steps 12 \
  --output results/dream_smoke.jsonl
```

전체 실행:

```bash
CUDA_VISIBLE_DEVICES=0 python generate_answers.py --model qwen \
  --max-new-tokens 12 --output results/qwen_generation.jsonl
CUDA_VISIBLE_DEVICES=1 python generate_answers.py --model dream \
  --max-new-tokens 12 --dream-steps 12 \
  --output results/dream_generation.jsonl
```

중단된 실행은 같은 명령에 `--resume`을 추가한다.
