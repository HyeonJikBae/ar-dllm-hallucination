# 지식 bin 확장 감사: 3/5, 2/5, 1/5, 0/5

이 문서는 `official_alias_is_correct`로 만든 자동 bin을 그대로 덮어쓰지 않고, 질문 동치성·의미 정답·생성 완결성·상대 모델 결과를 추가로 감사한 결과다.

## 어디까지 이미 했고, 무엇을 새로 했는가

앞선 `PRIORITY_CASE_ANALYSIS.md`에서는 자동 `4/5` 11건을 전수 분석했고, 모델 비교에 유용한 12 facts만 골랐기 때문에 그 안에 자동 `3/5` 6건과 `2/5` 3건이 일부 포함돼 있었다. 따라서 나머지 bin 분석은 **부분적으로만** 된 상태였다.

이번 확장에서 다음을 전수 확인했다.

| 자동 bin | model-fact | 응답 | 앞서 분석됨 | 이번 최종 범위 |
|---:|---:|---:|---:|---:|
| 3/5 | 9 | 45 | 6/9 | 9/9 |
| 2/5 | 5 | 25 | 3/5 | 5/5 |
| 1/5 | 13 | 65 | 0/13 | 13/13 |
| 0/5 | 77 | 385 | 사례 감사 전 | 77/77 및 고유 48 facts inventory |

자동 분포는 다음과 같다.

| 모델 | 3/5 | 2/5 | 1/5 | 0/5 |
|---|---:|---:|---:|---:|
| Dream | 7 | 1 | 5 | 46 |
| Qwen | 2 | 4 | 8 | 31 |
| 합계 | 9 | 5 | 13 | 77 |

## 해석 원칙

`k/5`는 내부 지식량의 직접 측정치가 아니라 **다섯 질문에서 gold answer가 출력된 횟수**다. 다섯 질문은 같은 생성기가 만든 상관된 paraphrase이고 독립 Bernoulli trial이 아니므로, `1/5=20% 확률`, `0/5=지식 없음`처럼 확률적으로 읽을 수 없다.

권장 operational label은 다음과 같다.

| 자동 bin | 권장 label | 허용되는 해석 |
|---:|---|---|
| 3/5 | `RECALL_OBSERVED_3_OF_5` | 여러 표현에서 recall됐지만 질문 형태에 따라 실패가 관측됨 |
| 2/5 | `RECALL_OBSERVED_2_OF_5` | 두 번 재현된 recall이나 질문 품질·채점 검토 필요 |
| 1/5 | `SINGLE_OBSERVED_RECALL` | 한 번 recall이 관측됨; chance인지 cue-dependent access인지 미결정 |
| 0/5 | `NO_CORRECT_RECALL_OBSERVED` | 현재 다섯 출력에서 gold recall을 관측하지 못함 |

## 1. 자동 3/5 전수 감사

| 모델 | Fact | 자동→수동 | 등급 | 핵심 판정 |
|---|---|---|---|---|
| Dream | Aurochs | `11010→11010` | CORE | v2·v4 모두 `Mammoth`; 안정적 same-domain competitor |
| Dream | Ewan McGregor | `11001→11001` | CORE | v2·v3 모두 `Guy Pearce` |
| Dream | James Garner | `11010→11010` | CORE | v2·v4 모두 actor 대신 `Jim Rockford` 복사 |
| Qwen | Tyne | `11100→11100` | CORE | title completion에서 `Lindisfarne`, `Shannon`으로 실패 |
| Dream | McKinley | `00111→10111` | QUALIFIED | v0 피격 서술은 의미 정답; 실제 4/5로 이동 |
| Dream | Clumber Spaniel | target `10110`, literal `11111` | QUALIFIED | Sporting dog는 넓어진 질문에 맞음 |
| Dream | Pinocchio | 유효 v0–v3 `1011` | QUALIFIED | v1 Tinker Bell은 실제 실패; v4 `created`는 관계 drift |
| Dream | Daughter | `11001` | QUALIFIED | relation 방향 단서 약화 + `Joey/Joely` source typo |
| Qwen | AFC Wimbledon | benchmark `10011` | QUALIFIED | Reading FC 반복이나 현재형 Kingsmeadow 관계가 시간적으로 낡음 |

### 가장 강한 3/5 사례

1. **Aurochs:** Dream이 `common/widely known/familiar`에서는 Aurochs, `commonly call/everyday name`에서는 Mammoth를 출력한다. 모든 질문에 `Bos primigenius`, large cattle, Europe, 1627 단서가 남아 있어 정보 부족이 아니다.
2. **Ewan McGregor:** Dream이 두 동치 질문에서 같은 배우 `Guy Pearce`를 반복한다. Qwen은 다섯 질문 모두 Ewan McGregor다.
3. **James Garner:** Dream이 긴 actor-role 구조에서 답 slot에 질문 속 character `Jim Rockford`를 넣는다. 동일 오류가 두 번 재현된다.
4. **Tyne:** Qwen은 세 번 Tyne을 회수한 뒤 v3에서 band entity `Lindisfarne`를 복사하고 v4에서 다른 지명 `Shannon`을 만든다.

### 새로 발견된 3/5 사례

- `Daughter`: 정답 질문은 `X to Y`, `familial tie`처럼 관계 방향/가족 유형이 분명하다. 오답 질문은 `relationship between`, `connected`처럼 관계를 대칭적·포괄적으로 바꾼다. 단, 원 entity `Joey Richardson`은 실제 `Joely Richardson`의 오타이므로 수정 후 재실험해야 한다.
- `AFC Wimbledon`: benchmark 내부에서는 Qwen이 `Reading FC`를 두 번 반복하는 좋은 competitor 사례다. 그러나 AFC Wimbledon은 Kingsmeadow에서 Plough Lane으로 옮겼으므로 현재형 질문은 시점이 없는 temporal fact가 된다.
- `Tyne`: 기존 cross-model 12 facts에 포함되지 않았던 깨끗한 Qwen 3/5다.

결론적으로 자동 3/5 9건 중 바로 사용할 CORE는 4건이다. 나머지 5건에도 지식/실패 신호가 있지만 재채점, valid-subset 사용, source 수정 또는 시점 고정이 필요하다.

## 2. 자동 2/5 전수 감사

| 모델 | Fact | pattern | 등급 | 핵심 판정 |
|---|---|---|---|---|
| Qwen | Henry III–56 years | `10100` | CORE | `56` 2회 대 동일 경쟁 수치 `59` 3회 |
| Dream | Paris | `11000` | QUALIFIED | 올림픽 개최 도시는 알지만 `first + twice` 관계가 불안정; v4 제외 |
| Qwen | Paris | `10100` | QUALIFIED | St. Louis/Los Angeles; v4 제외 |
| Qwen | Sliced bread | `10001` | QUALIFIED | 정답 2, 실제 오답 2, 답 직전 token cutoff 1 |
| Qwen | Clumber Spaniel | target `10100`, literal `11101` | EXCLUDE | Sporting 2개는 broadened 질문에 유효 |

### 가장 중요한 2/5 사례: 56 대 59

다섯 질문은 모두 Henry III의 영국 왕 재위 기간을 묻는다.

| v | Qwen 출력 |
|---|---|
| v0 | `56 years` ✓ |
| v1 | `59 years` ✗ |
| v2 | `56 years` ✓ |
| v3 | `59 years` ✗ |
| v4 | `59 years` ✗ |

이는 정답 지식이 두 번 재현됐고 동일 오답이 세 번 반복된 사례다. `2/5`를 “우연 정답에 가깝다”고 단순화하기보다 **두 숫자 기억/연상의 경쟁**으로 분석하는 편이 타당하다. Royal Household 자료도 Henry III의 재위를 56년으로 정리한다: https://www.royal.uk/50-facts-about-queens-reign

### Paris

유효한 v0–v3에서 다음 네 상태가 모두 나온다.

| v | Dream | Qwen |
|---|---|---|
| v0 | Paris ✓ | Paris ✓ |
| v1 | Paris ✓ | St. Louis ✗ |
| v2 | Athens ✗ | Paris ✓ |
| v3 | London ✗ | Los Angeles ✗ |

두 모델 모두 도시/올림픽 domain은 유지하지만 `역사상 가장 먼저 두 번째 개최에 도달`한 ordinal-temporal relation에서 흔들린다. v4는 비문이므로 제외한다.

### Sliced bread

- v0·v4: `sliced bread` 정답
- v2: fabricated `Chillicothe Cracker`
- v3: 관련 브랜드 `Wonder Bread`
- v1: 답을 쓰기 직전 32-token budget에서 종료된 non-answer

따라서 자동 2/5의 세 오답을 모두 동일한 hallucination으로 세면 안 된다. 최초 상업적 sliced bread가 1928년 Chillicothe Baking Company에서 판매됐다는 source fact 자체는 확인된다: https://americanhistory.si.edu/explore/stories/best-thing-sliced-bread

## 3. 자동 1/5 전수 감사

| 모델 | Fact | pattern | 수동 분류 | 실제 관찰 |
|---|---|---|---|---|
| Dream | Isle of Wight | `00001` | RECLASSIFY | `Wight` 4회도 의미 정답; 실제 5/5 |
| Dream | Raj Quartet | `10000` | NEAR_RECALL | v1 `Raj Quartels`; exact 1/5지만 entity recall은 최소 2/5 |
| Dream | The Tenth Planet | `00001` | WEAK | `Tomb of the Cybermen` 3회 |
| Dream | Tyne | `00010` | WEAK | place-name cue에서만 Tyne; Sea/Hill 각 2회 |
| Dream | 1972 | `10000` | WEAK | 원문만 정답; `1968` 3회 |
| Qwen | The Tenth Planet | `00100` | WEAK | `The Daleks' Master Plan` 4회 |
| Qwen | Alaska | `01000` | WEAK | canonical `longest coastline`에서만 정답; Maine 4회 |
| Qwen | Montmorency | `10000` | WEAK | 원문에서 희귀 정확명, paraphrase에서 붕괴 |
| Qwen | Cigarettes | `01000` | WEAK | `consumer products` type cue에서만 정답 |
| Qwen | 1972 | `10000` | WEAK | 원문만 정답; `1971` 4회 |
| Qwen | Dublin | `10000` | WEAK | 원문만 정답; Lisbon 3회 |
| Qwen | Great Smoo–cave | `10000` | CHANCE CANDIDATE | Smoo→Smoky/smooth lexical guessing이 지배 |
| Qwen | Daughter | `00001` | INVALID | `Joey/Joely` entity typo; 수정 후 재실험 필요 |

### 핵심 결론

13건을 일괄적으로 “우연히 한 번 맞힘”이라고 부를 수 없다.

- 자동평가 false negative로 실제 5/5: 1건
- near-recall: 1건
- source entity 오류: 1건
- 반복 경쟁자/cue 의존/원문 의존의 weak knowledge: 9건
- 현재 결과만 보면 chance-candidate에 가장 가까움: 1건

특히 13건 중 6건은 정답이 v0 원문에서만 나왔다. 기대 위치가 균등하다고 가정하면 v0에 과도하게 몰려 있으며, Dream과 Qwen이 **같은 1972 fact에서 모두 원문만 맞힌다**. 이는 architecture 차이보다 원본 QA 표면형 친숙도 또는 memorization 가능성을 먼저 확인해야 한다는 신호다.

### 1/5 추가 실험 설계

각 fact에 단순히 비슷한 질문 4개를 더 만드는 것으로는 부족하다. 최소 다음 구조를 권장한다.

1. 중립적이고 사람 검수된 동치 질문 10개
2. 현재 유일한 정답 질문의 핵심 cue를 유지한 질문 5개
3. 그 cue만 제거한 matched 질문 5개

예:

- Tyne: `place name` cue 5개 대 cue 없는 title completion 5개
- Cigarettes: `consumer product/item` cue 5개 대 generic `what` 5개
- Alaska: `longest coastline`, `greatest total coastline length`, `among all 50 states`를 교차
- 1972: 세 사건의 순서와 `appointment/elected/assumed office`를 factorial하게 변경하고 각 sub-fact도 따로 질문

현재 Qwen은 greedy decoding이고 Dream도 temperature/algorithm temperature가 0이므로, 이번 single-hit는 sampling temperature로 생긴 변동이라기보다 질문 표현·생성·평가의 영향으로 보는 것이 우선이다.

## 4. 자동 0/5 전수 감사

### 총괄

| 항목 | 결과 |
|---|---:|
| 공식 0/5 model-fact | 77 |
| 고유 fact | 48 |
| 양 모델 모두 0/5 | 29 |
| 한 모델만 0/5 | 19 |
| 출력에서 실제 정답 지식이 직접 확인된 공식 0/5 model-fact | 최소 7 |
| 32-token 상한에 EOS 없이 도달한 출력 | 23/385 |
| token cutoff 영향을 받은 model-fact | 11 |
| 자동 refusal | 0 |
| 명시적 no-information 응답 | 2 |
| 빈 출력 | 2 |
| 0/5 모델의 상대 모델이 5/5 | 9 facts |
| 상대 모델이 4/5 이상인 강한 격차 | 12 facts |

### 0/5인데 실제 정답 흔적이 직접 나온 최소 7 model-facts

| Fact | 모델 | 자동 | 수동/증거 | 원인 |
|---|---|---:|---|---|
| James Parkinson | Dream | 0/5 | 5/5 | alias가 문자 그대로 `Dr. (James) Parkinson` 하나뿐 |
| James Parkinson | Qwen | 0/5 | 4/5 | Parkinson/James Parkinson을 matcher가 모두 놓침 |
| Olfactory bulb | Dream | 0/5 | brain 5/5 | gold `Nose`가 해부학적으로 부정확 |
| Olfactory bulb | Qwen | 0/5 | brain 5/5 | gold `Nose`가 해부학적으로 부정확 |
| Senet | Dream | 0/5 | board 계열 최소 2/5 | `On a board`만 alias라 상세 board 답을 놓침 |
| Senet | Qwen | 0/5 | board 계열 최소 2/5, 지식 흔적 4회 | bare `board`도 놓침 |
| Henry III | Dream | 0/5 | `1216 to 1272` 1회 | 56년과 동치인 정확한 연도 범위를 놓침 |

Olfactory bulb는 뇌의 아래쪽, 각 nasal cavity 위에 있는 구조로 설명된다. 따라서 두 모델의 `brain` 답을 무지로 처리하면 안 된다: https://www.cancer.gov/publications/dictionaries/cancer-terms/def/olfactory-bulb

James Parkinson의 1817 저작과 저자명도 직접 확인된다: https://wellcomecollection.org/works/e6gjgeqw

Senet은 실제 board game이며 3×10 squares의 board에서 진행된다: https://www.metmuseum.org/ja/perspectives/ancient-egypt-board-games

### 양 모델 모두 공식 0/5인 29 facts

| Gold | Dream의 대표 출력 | Qwen의 대표 출력 | 주요 해석 |
|---|---|---|---|
| Carnaby Street | Oxford Street×4 | Oxford Street×5 | shared shopping-street competitor |
| Morgan Grenfell | Barclays×5 | HSBC×3 | 시점·기준 없는 오래된 `biggest` 질문 |
| Nose/olfactory bulb | brain×5 | brain×5 | reference 오류; 모델 답이 더 타당 |
| The Richardsons | Kray×3 | Kray 계열×5 | related London gang substitution |
| Devon Loch | Red Rum×4 | cutoff×3 등 | salient horse + 생성 상한 |
| David Frost | David Paradine×5 | David Paradine×5 | 회사명 entity를 창립자로 복사 |
| Louise Woodward | 여러 유명 재판 | 여러 유명 재판 | v2 answer type drift도 존재 |
| Byron | Allan/Alan×4 | Francis×3 | 안정적인 middle-name 경쟁자 |
| Long jump | hurdles/200m | 100m×4 | decathlon event order 혼동 |
| Cat Stevens | 여러 음악인 | 여러 음악인 | diffuse artist uncertainty |
| Green | Brown/Davies | Al/Hughes | surname completion/slot copy |
| Cleveland Brown | Peter Griffin×5 | Peter Griffin×5 | shared prominence bias |
| Mike Atherton | Andrew Flintoff×5 | Michael Vaughan×5 | 모델별 안정적 captain competitor |
| On a board | board 설명×2 | board 계열×4 | semantic alias/reference failure |
| Tony Blair | Obama/빈 출력 | no-information 등 | 답 미관측 + 질문 `published` 모호성 |
| 960 | 1600/1920 | 240/480 | 화폐 단위·산술 오류 |
| None/0 runs | 100×4 | 12×3 | 수치 오답; gold `None`도 부자연스러움 |
| Michelle Ryan | 여러 배우 | 여러 배우 | diffuse person hallucination |
| Bolton | Southport/Liverpool | Cheltenham/Stratford | Octagon Theatre 명칭 비고유성 |
| Robbie Coltrane | Simon Bradley×4 | 여러 배우 | actor competitor |
| Strictly Come Dancing | 여러 TV show | 여러 TV show | programme-domain 분산 |
| Netherlands | Ireland×3 | Australia×4 | cricket country association |
| Sue Barker | Fred Perry×3 | 여러 선수 | question entity copy/tennis prominence |
| Appendicitis | tuberculosis×2 | tuberculosis×3 | shared royal-illness misconception |
| James Hogg | Robert Burns×5 | Robert Burns×5 | shared Scottish-poet prominence |
| José Antonio Reyes | Wright 계열×4 | Ian Wright×4 | related Arsenal player substitution |
| Quiver | Family 계열 | invented band names | 원문 control character/source corruption |
| Stephen Potter | Robert Greene×5 | Ian Fleming×4 | 모델별 안정적 author competitor |
| James Parkinson | Parkinson×5 | Parkinson×4 | 명백한 scorer false negative |

이 중 `David Paradine`, `Peter Griffin`, `Robert Burns`는 두 모델이 다섯 질문 모두 같은 오답으로 수렴했다. 이는 “아무 정보도 없는 random guess”와 다른 현상이며, 공통 학습분포의 salient competitor 또는 slot-binding bias를 시사한다.

### 한 모델만 0/5인 19 facts

상대 모델 5/5인 가장 강한 9개 Dream-zero/Qwen-stable facts:

| Gold | Dream 대표 오류 | Qwen | 판정 |
|---|---|---:|---|
| Henkel | Unilever/BIC | 5/5 | 강한 Qwen 우위 |
| Dennis Potter | Alan Bennett×5 | 5/5 | 가장 깨끗한 stable knowledge gap |
| Great British Bake Off | One Show/Doctor Who | 5/5 | Qwen 우위 |
| Breakfast at Tiffany's | 다른 영화 | 5/5 | film-song gap |
| Crimson Tide | The Siege×3 등 | 5/5 | same-domain movie confusion |
| The Merry Widow | 다른/invented operetta | 5/5 | work-character relation gap |
| Clematis | Moneywort×3 등 | 5/5 | plant-name gap |
| Red Hot Poker | Devil's fingers×3 등 | 5/5 | common-name gap |
| America yacht | 질문을 따라 쓰다 5회 cutoff | 5/5 | Dream 지식 부재 판정 불가; 생성 실패 |

추가 10개:

| 0/5 모델 | Gold | 상대 모델 | 해석 |
|---|---|---:|---|
| Qwen | Colonel | Dream 4/5 | 깨끗한 Dream 우위; Qwen은 장성 계급으로 수렴 |
| Qwen | New Orleans | Dream 4/5 | Dream 우위이나 축약된 river-route premise 존재 |
| Dream | Jacob Epstein | Qwen 4/5 | Dream은 Henry Moore×4; 강한 Qwen 우위 |
| Dream | Montmorency | Qwen 1/5 | Qwen 원문 single recall; chance 확정 불가 |
| Dream | Cigarettes | Qwen 1/5 | Qwen type cue 조건부 recall |
| Dream | 56 years | Qwen 2/5 | Dream도 정확한 연도 범위 1회 출력 |
| Dream | A cave | Qwen 1/5 | Qwen original-only recall |
| Dream | Sliced bread | Qwen 2/5 | Qwen 두 변형에서 recall 재현 |
| Dream | AFC Wimbledon | Qwen 3/5 | temporal relation 주의 |
| Dream | Dublin | Qwen 1/5 | Qwen original-only recall; Dream 빈 출력 1회 |

### 0/5의 생성·refusal 문제

자동 `is_refusal`은 385개 모두 false지만 실제로는 다음이 존재한다.

- 명시적 no-information 2개: `answer is not provided...`
- 빈 출력 2개
- tautological non-answer
- EOS 없이 32-token 상한에 걸린 출력 23개, 11 model-facts
- `America` yacht fact의 Dream은 다섯 번 모두 질문을 따라 쓰다가 cutoff되어 지식을 출력할 기회가 없었음

따라서 이전의 “refusal은 어차피 모르는 것이므로 오답” 원칙은 유지할 수 있지만, 현재 detector가 refusal/no-answer/generation failure를 모두 포착하지는 않는다. `WRONG_ANSWER`, `NO_INFORMATION`, `EMPTY`, `TRUNCATED_BEFORE_ANSWER`를 최소한 별도 상태로 기록해야 한다.

## 최종 권장 분석 순서

1. **3/5 CORE 4건**: Aurochs, Ewan McGregor, James Garner, Tyne
2. **2/5 CORE/비교 가능 subset**: Henry III 56↔59, Paris v0–v3, Sliced bread의 genuine wrong만
3. **1/5 재실험**: scorer/source 문제를 먼저 제거하고 fact당 15–20개의 controlled question 추가
4. **0/5 정제**: scorer/reference 오류와 truncation을 제거한 뒤에만 `NO_CORRECT_RECALL_OBSERVED` control로 사용

표현 분석에는 같은 fact에서 반복되는 competitor를 우선한다.

- Aurochs↔Mammoth
- Ewan McGregor↔Guy Pearce
- James Garner↔Jim Rockford
- 56↔59
- Alaska↔Maine
- Tenth Planet↔각 모델 고유 Doctor Who serial
- 양 모델 shared wrong: David Paradine, Peter Griffin, Robert Burns

이 사례들은 단발성 무관 오답보다 hidden-state direction, answer-token logit, SAE feature 차이를 해석하기 쉽다.

