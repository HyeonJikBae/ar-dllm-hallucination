# 문헌 정리와 우리 연구에의 적용 (2026-11-17)

출처: (A) 사용자가 직접 찾아 준 10 편의 요약 (사용자 제공 표를 압축 정리; 각 논문 원문·수치는 이 세션에서 독립 재검증하지 않았다), (B) 이 세션에서 웹 검색으로 찾은 논문 (검색 결과 요약 수준, 원문 미확인 표시).
인용할 때는 반드시 원문을 확인할 것.

## A. 사용자 제공 10 편

### A1. DLM decoding, commitment, collapse (6 편)
| 논문 | 핵심 | 한계 (저자 또는 평가) |
|---|---|---|
| **The Flexibility Trap** (ICML 2026) | arbitrary-order (LCR) 디코딩은 쉬운 토큰을 먼저 정하고 고엔트로피 분기 토큰 ("Therefore", "Since") 을 우회해 entropy 가 일찍 붕괴, Pass@k 가 줄어든다.  block 이 클수록 Pass@k 단조 감소, HumanEval AR-only 21.3% vs AO-only 0.6%.  JustGRPO: 학습 때 dLLM 을 AR policy 로 취급해 표준 GRPO 적용, 추론은 병렬 디코딩 유지 (GSM8K 89.1%) | arbitrary order 쪽이 사실상 LCR 하나, Pass@k 를 RL 상한의 proxy 로 가정, 학습 AR / 평가 LCR 불일치 미분석 |
| **Low-Confidence Remasking Traps Flexibility** (ICLR 2027 under review) | 위 논문의 원인은 order 가 아니라 LCR 의 proposal filtering.  LCR = 모든 mask 위치에서 샘플한 뒤 확률 가장 높은 proposal 만 commit, TPP = top-1 확률이 가장 높은 위치를 먼저 고르고 그 분포에서 샘플.  LCR 은 non-top 토큰을 (1−c_T)^m 으로 지수 억제 (LLaDA T=1.2 에서 non-top commit 0.7–1.1% 대 TPP 36–38%).  LCR→TPP 로 바꾸면 Pass@k 가 AR 수준 회복, EGI (첫 step 만 entropy 최대 위치에서 샘플) 는 AR 초과 | toy model 단순화, 주 분석 one-token-per-step, EGI 첫 step 개입 heuristic |
| **ReMDM** (NeurIPS 2025) | unmask 된 토큰을 확률 σ_t 로 다시 mask 하는 backward process.  재학습 없이 사전학습 MDLM/LLaDA 에 sampler 로 붙임.  OWT MAUVE 0.656 (AR 0.760), LLaDA Countdown·TruthfulQA 개선 | σ_t 하이퍼파라미터 많음, LLaDA 이득 작음 (45.2→46.1), remask 대상이 confidence heuristic |
| **Toward Safer DLMs: Priming Vulnerability** (ICLR 2026) | 중간 step 에 harmful 응답의 affirmative 토큰이 나타나면 aligned 모델도 끌려감 (anchoring attack 1 step 개입 ASR 2%→21%, t=16 이면 80%+).  원인: 학습이 fully-masked 시작만 다루고 오염된 중간 상태 복구를 안 가르침.  Recovery Alignment (오염 상태에서 시작하는 GRPO) 로 ASR 88.7→8.3% | 늦은 개입 (t=32) 에는 여전히 43–51%, RLHF 형만 구현 |
| **The Answer Is Not the State** (under review) | 응답을 증거 E / 근거 R / 결정 D 로 나누고 증거 도착 시점 τ 와 reset 시점 γ 를 통제.  **Answer reset (D 만 재mask) 은 부족하고 Path reset (R+D) 이 결정을 현재 증거에 따르게 함** (correct 증거 정확도 96.7% 대 answer reset 55.6%, 오염 증거에서 wrong 추종 90%).  coherent rationale 이식이 shuffle 보다 결정을 +28.3pt 더 조종.  attribution 으로 고른 partial reset 70% 는 full path reset 99% 에 못 미침 (attribution–repair gap) | Dream, LLaDA, iLLaDA 모두 bidirectional MDLM, 산술·HotpotQA 위주, free-form 에서 Dream 효과 거의 없음 (+1pt), 오염 증거도 똑같이 따름 |
| **DLMs Know the Answer Before Decoding (Prophet)** (ICLR 2026) | LLaDA GSM8K 에서 random remasking 시 97.2% 가 step 절반에 이미 top-1 정답 (LCR 24.2%, "Answer:" suffix 75.8%).  answer region 의 top-1/top-2 logit gap 으로 조기 종료, step 최대 3.4× 감소 | **정답을 맞힌 표본만 분석 (selection bias)**, answer region 식별되는 과제 한정, 임계값 heuristic |

### A2. Activation steering 과 표현 이론 (4 편)
| 논문 | 핵심 | 한계 |
|---|---|---|
| **CASAL** (ICLR 2026, Meta) | known/unknown 정의: 질문당 10 샘플 중 7 개 이상 정답이면 known, 7 개 이상 오답이면 unknown (중간 버림).  질문 마지막 토큰 residual 에서 difference-in-means steering vector 를 만들고 한 layer sub-module 을 target activation 과의 MSE 로 학습 (steering 을 가중치로 증류).  hallucination 30–40% 감소, LoRA SFT/DPO/GRPO 대비 compute ~30×, data ~20× 효율, OOD·VLM·MoE 에도 | 라벨링에 10 샘플 필요, 한 위치·한 layer, short-form QA 만, PopQA 에서 known 정확도 하락 (91.1→85.1) |
| **AlphaSteer** (ICLR 2026) | steering vector 를 s = Δh 로 학습, Δ 를 benign activation 의 null space 로 projection → benign 이면 Δh≈0, malicious 는 refusal vector 로 매핑 (regularized least squares 닫힌 해).  safety–utility 충돌을 원리적으로 해결.  adaptive GCG 에도 DSR 95.5% | null space 가 근사라 benign 데이터 coverage 의존 (14k→3k 면 utility 67.3→52.9), Cipher 공격에 약함, GPT-4o 판정 |
| **CAST** (ICLR 2025, IBM) | condition vector (PCA 첫 성분) 와 프롬프트 첫 forward 의 cosine similarity 가 θ 를 넘으면 이후 모든 생성 step 에 behavior vector 를 더함.  harmful 거부율 크게 상승, harmless 거의 그대로 | layer·θ·강도에 민감, **조건 검사가 프롬프트 첫 pass 만이라 생성 중 드러나는 내용은 못 잡음**, 수학 문제를 harmful 로 오판하는 사례 (AlphaSteer 재현) |
| **A Theory of Platonic Representations in LMs** (ICLR 2027 under review) | multilingual Random Hierarchy Model (상위 규칙 공유, 하위 언어별) 에서 Bayes-optimal next-token predictor 가 belief propagation.  중간 layer 가 공유된 상위 latent 를 encode 해 유사도 최대, 유사성과 정렬은 별개 (정렬은 code-switch 데이터 필요).  layerwise novelty (이전 layer 로 선형 예측되는 성분을 뺌) 로 언어 간 유사도 상승 예측, Mistral·OLMo 에서 확인 | 합성 CFG 와 소형 transformer 중심, BP 선형 쌓임 등 모델링 가정 많음, 실제 LLM 검증은 일부 |

### A3. 사용자 정리의 핵심 논점 (보존)
- **두 논문의 쟁점**: Pass@k 격차의 원인이 order 자체 (Flexibility Trap) 인지 LCR 의 filtering (LCR 논문) 인지.  붕괴 비교 실험은 **decoding rule (LCR / TPP / AR order) 을 분리해 보고**해야 공격받지 않는다.
- **잘못된 토큰을 수정할 능력과 무엇이 잘못됐는지 판단하는 능력은 다르다** (ReMDM 은 전자만).
- **빠른 수렴 (Prophet) 은 옳은 결론 때문인가, 다른 결론을 고려하기 어려운 상태 때문인가?** 논문들만으로는 구별 불가.
- **세 가지 실패 분리**: Knowledge failure (모름) / Evidence-use failure (있는 정보를 못 씀) / Decision-revision failure (결정을 못 고침).
- **제안된 중심 질문**: "When Does a Diffusion Language Model Become Committed to an Incorrect Answer?"  가장 약한 고리는 "priming·lock-in 이 사실 질문의 환각에서도 일어나는가".  분석 논문이 방법 논문보다 문제 정의가 명확하며 방법은 원인 분석 뒤에 도출하는 편이 설득력 있다.
- **사용자 정리의 실험 3 개**: ① 증거 교체 시점 조절 후 답 변경 가능성 ② LCR/TPP/random/remasking 을 같은 compute budget 에서 비교 ③ 오류 전후 activation 추적과 causal patching/steering.
- **DLM-hal 메모**: CASAL 의 known/unknown 정의로 "모르는 질문에서 더 지어내는가" 와 "아는 질문에서도 무너지는가" 분리 가능.  세 steering 논문의 방향·개입을 SAE 대신 인과 검증 도구로 쓸 수 있으나 모두 AR 프롬프트 마지막 토큰 기준이라 DLM 에서는 위치·denoising step 을 새로 정해야 한다.  Platonic 의 RHM+BP 는 AR 대 DLM 붕괴 차이의 장난감 모델 후보.

## B. 이 세션에서 찾은 문헌 (원문 미확인, 검색 요약 수준)
- DLM 환각 감지: TraceDet (ICLR 2026, 디코딩 trace 의 정보병목), TDGNet (arXiv 2602.08048, 시간 그래프), DynHD (2603.16459), HIVE (2604.26139, Dream-7B-Instruct 평가), **TRE (2607.22661, 학습 없는 단일 궤적 엔트로피, 늦은 commit 가중)**, DeMTS (2608.14632).  → 우리 감지 신호 (궤적) 와 직접 비교 가능.
- DLM 시간 동역학: Time Is a Feature (2508.09138, 중간 정답 소실 "temporal oscillation", 시간 자기일관성), DLMs Know the Answer Before Decoding (2508.19982), Remask Don't Replace (2604.18738, T2M), Frequency-Aware Remasking (2611.10486), Corrective Diffusion LMs (2512.15596: 저신뢰 위치가 실제 오류와 잘 맞지 않아 remasking 이 맞는 토큰을 덮어쓴다).
- 아는데 틀림 / 환각 메커니즘: Simhi et al. Distinguishing Ignorance from Error (2410.22071, HK+/HK−, WACK), Ferrando et al. Do I Know This Entity? (ICLR 2025, SAE 로 개체 인식 방향, 속성 이동 attention head 를 교란), Yuksekgonul et al. Attention Satisfies (ICLR 2024, 제약 토큰 attention 으로 사실 오류 예측, SAT Probe).  우리 E13 에서 주어 attention 감지는 0.50 이었다.
- 조건부·희소 조향: ACT (2406.00034, 표본별 세기·클러스터별 벡터), GAPS (2609.01878, 정적 AUROC 게이트 |AUROC−0.5| > 7σ0 + 동적 가우시안 사후 게이트, Gemma-3 4B·Qwen-3 1.7B 독성 과제, 층 15), Sparse Representation Steering (2503.16851), Sparse Mediation (EACL 2026, 차원 10–30% 만으로 조향), AxBench (ICML 2025: 평균 차이 DiffMean 이 SAE 보다 개념 감지·조향 모두 우수), FLARE (2305.06983, 불확실하면 다시 생성: 두 번 생성의 선행).
- 표현 정렬 비판: Platonic Representation Hypothesis (Huh et al. ICML 2024), Aristotelian View (2602.14486: 정렬 점수가 모델 폭·깊이로 부풀려짐), vec2vec (2505.12540).
- GAPS 한계 (논문 식에서 도출한 우리 판단): 좌표별 독립 가우시안 가정 (우리 신호는 1,022 개 좌표에 중복·분산), 사전확률 동일 가정, 사실 단위 혼입 (아는가 축) 미처리, 방향·세기는 여전히 평균 차이와 전역 α, AR 의 매 토큰 마지막 위치 설계, 평가 과제가 분리 쉬운 개념 (독성).

## C. 우리 연구에의 적용 (근거와 연결)
| 실험 | 가져오는 논문 | 합당한 이유 |
|---|---|---|
| P1 확정 시점 분석 | Prophet, Answer Is Not the State | Prophet 은 맞힌 표본만 분석 (selection bias) → 우리는 틀린 표본을 추가.  오답 유형별 확정 step 이 이미 다르게 나옴 (붕괴 4, entity 22) |
| P2 첫 토큰 고정 진단 | Priming Vulnerability (anchoring) | "첫 결정만 틀린가" 를 가름.  반대 방향 (맞힌 질문에 틀린 첫 토큰 주입) 은 묶임의 세기를 잼.  정답 라벨을 쓰므로 진단용 |
| P3 되돌리기 범위 | Answer Is Not the State, ReMDM | 우리 두 번 생성은 전체 reset + 조향.  answer 만 / path 전체 reset 비교로 사실 질문에서도 "답만 열어서는 부족" 이 성립하는지 확인.  E14 다시 가리기가 실패한 이유 설명 |
| P4 확정 순서 통제 | Flexibility Trap, LCR 논문 | Dream 이 Qwen 보다 2.6 배 붕괴하는 이유가 모델인지 확정 순서인지.  AR 순서에서 붕괴가 사라지면 후자 |
| P5 CAST 식 기준선 | CAST | CAST 는 첫 forward 에서만 조건 검사.  우리 궤적 감지가 그보다 나은지 전체 규모로 |
| P6 null-space 조향 | AlphaSteer | 깨짐을 원리적으로 줄일 후보.  다만 같은 사실 안 정답/오답 분리가 약해 (0.62~0.66) 기대 낮음, 5/5 보호엔 유효할 수 있음 |
| P7 샘플링 라벨 | CASAL | "문구 민감성인가 단순 불안정인가" 방어 |
| P8 플라토닉 정렬 | Platonic (이론), PRH 비판 | "표현은 같고 디코딩이 다르다" 를 받침.  Dream 초기화가 Qwen2.5 인지 확인 필요, 무작위 대응 대조 필수 |
- 쓰지 않는 것: Recovery Alignment, CASAL 증류 (학습 필요).  ReMDM 은 인용만 (E14 에서 다시 가리기 실패).  LCR 대 TPP 는 온도 0 에서 동일하므로 한 줄 명시.
