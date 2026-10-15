# 우선순위 실제 사례 수동 감사: Part 1–2

대상은 `official_alias_is_correct` 기준 결과다. 자동 점수는 보존하고, 질문 동치성·답 target·응답 전체 품질을 별도 수동 필드로 기록했다.

## 핵심 결론

- Part 1 자동 4/5 11건: 핵심 5건, 보조 2건, 제외/재생성 4건.
- Part 2 비교 12 facts: 핵심 6건, 조건부 4건, 현재 causal set 제외 2건.
- Part 2의 60쌍은 자동으로 양쪽 정답/한쪽 정답/양쪽 오답이 38/17/5였지만, 질문을 문자 그대로 읽은 수동 의미 판정에서는 42/14/4로 바뀐다.
- 의미·target이 비교 가능한 53쌍만 남기면 양쪽 정답/한쪽 정답/양쪽 오답은 40/12/1이다. 나머지 7쌍은 paraphrase drift/모호성으로 비교에서 제외했다.
- 가장 반복적인 실제 실패는 (a) 질문 속 entity 복사/역할 결속 실패, (b) 같은 의미권 경쟁 entity로 치환, (c) 특정 lexical cue에 대한 민감성이다.
- 자동 0/1 판정만으로는 scorer 오류와 paraphrase drift를 구별하지 못하므로 hidden-state 분석 전에 이 감사 필터가 필요하다.

## Part 1. 자동 4/5 사례 11건 전수 감사

| 모델 | 사례 | 자동→수동 | 실패 v | 등급 | 판정 | 핵심 해석 |
|---|---|---:|---|---|---|---|
| dream | Eve–Adam's rib | 11110→11111 | v4 | EXCLUDE | SCORER_FALSE_NEGATIVE | 'The rib of Adam'은 Adam's rib과 동치다. 지식 실패가 아니라 matcher 누락이다. |
| dream | Oberst–Colonel | 11110→11110 | v4 | CORE | CLEAN_PROMPT_SENSITIVE_FAILURE | 동치 질문에서 Colonel 대신 같은 군 계급 범주의 General을 냈다. |
| dream | Commandaria–Cyprus | 11011→11011 | v2 | CORE | CLEAN_PROMPT_SENSITIVE_FAILURE | 섬을 묻는데 질문 속 와인명 Commandaria를 복사했다. |
| dream | Missouri–New Orleans | 01111→01111 | v0 | SECONDARY | SOURCE_PREMISE_AMBIGUITY | 원문에서 San Francisco를 냈지만 Missouri가 직접 바다에 닿지 않는 축약 전제가 있다. |
| dream | Christmas Island–Australia | 11101→11101 | v3 | EXCLUDE | PARAPHRASE_ANSWER_TYPE_DRIFT | v3만 국가가 아니라 사람·직책·기관도 가능한 질문으로 바뀌었다. |
| qwen | Vectis–Isle of Wight | 10111→10111 | v1 | CORE | PROMPT_SENSITIVE_ASSOCIATIVE_COMPETITOR | 거의 동치인 문장에서 crown dependency의 강한 연상 경쟁자 Isle of Man으로 이동했다. |
| qwen | ECTO-1–Ghostbusters | 11011→11011 | v2 | EXCLUDE | PARAPHRASE_ANSWER_SCOPE_DRIFT | v2의 generic 'Who'가 집단 정답을 개인 정답으로 바꿀 수 있게 했다. |
| qwen | Solent–strait | 10111→10111 | v1 | CORE | PROMPT_SENSITIVE_ENTITY_ECHO | 분류명 strait 대신 질문의 고유명 Solent만 반복한 non-answer다. |
| qwen | Blue Fairy–Pinocchio | 11110→11110 | v4 | EXCLUDE | PARAPHRASE_RELATION_DRIFT | v4는 Blue Fairy가 생명을 준 관계를 'created'로 바꿔 원 fact를 훼손했다. |
| qwen | Jacob Epstein | 11011→11011 | v2 | SECONDARY | ANSWER_TYPE_CUE_ABLATION | 작품 관계는 유지되지만 구체 직업 단서가 사라지자 William Etty로 이동했다. |
| qwen | Paul Scott–Raj Quartet | 11011→11011 | v2 | CORE | CLEAN_PROMPT_SENSITIVE_FAILURE | 핵심 단서와 답 유형이 유지된 동치 질문에서 The Four Guineas를 생성했다. |

### 1-1. dream · Eve–Adam's rib

- Fact ID: `odql_1396__8dddda2c19d8`; gold: `ADAM'S RIB`
- 판정: `SCORER_FALSE_NEGATIVE` / `EXCLUDE`
- 메커니즘: possessive ↔ of-phrase alias gap
- 활용: 4/5 사례에서 제거하고 의미 기준 5/5로 교정

| v | 질문 | 출력 | 자동 | 수동 target | 질문 감사/응답 감사 |
|---|---|---|---:|---:|---|
| v0 | According to the Bible, what was Eve made from? | Adam's rib | ✓ | ✓ | VALID_EQUIVALENT; CORRECT |
| v1 | What material was used to create Eve, as described in the Bible? | Adam's rib | ✓ | ✓ | VALID_EQUIVALENT; CORRECT |
| v2 | In the biblical account, from what substance was Eve formed? | From Adam's rib. | ✓ | ✓ | VALID_EQUIVALENT; CORRECT |
| v3 | What was the origin of Eve's creation according to the Bible? | Eve was created from Adam's rib. | ✓ | ✓ | VALID_EQUIVALENT; CORRECT |
| v4 | According to the scriptures, what was Eve created from? | The rib of Adam | ✗ | ✓ | VALID_EQUIVALENT; SCORER_FALSE_NEGATIVE_CORRECT; The rib of Adam = Adam's rib; 소유격/of-구문 동치 누락 |

### 1-2. dream · Oberst–Colonel

- Fact ID: `odql_4288__9fe6137b224b`; gold: `COLONEL`
- 판정: `CLEAN_PROMPT_SENSITIVE_FAILURE` / `CORE`
- 메커니즘: same-type competitor substitution
- 활용: 핵심 내부표현 분석 사례

| v | 질문 | 출력 | 자동 | 수동 target | 질문 감사/응답 감사 |
|---|---|---|---:|---:|---|
| v0 | What is the equivalent rank in the British Army of the German rank Oberst? | Colonel | ✓ | ✓ | VALID_EQUIVALENT; CORRECT |
| v1 | What British Army rank corresponds to the German rank of Oberst? | Colonel | ✓ | ✓ | VALID_EQUIVALENT; CORRECT |
| v2 | Which rank in the British Army is the same as the German Oberst? | Colonel | ✓ | ✓ | VALID_EQUIVALENT; CORRECT |
| v3 | In the British Army, what rank is equal to the German Oberst? | Colonel | ✓ | ✓ | VALID_EQUIVALENT; CORRECT |
| v4 | What rank in the British Army equates to the Oberst rank in Germany? | General | ✗ | ✗ | VALID_EQUIVALENT; INCORRECT |

### 1-3. dream · Commandaria–Cyprus

- Fact ID: `odql_11052__8cf11607ec01`; gold: `CYPRUS`
- 판정: `CLEAN_PROMPT_SENSITIVE_FAILURE` / `CORE`
- 메커니즘: question-entity copy / role-binding failure
- 활용: 핵심 내부표현 분석 사례; v4의 부가 허위 주장도 별도 표시

| v | 질문 | 출력 | 자동 | 수동 target | 질문 감사/응답 감사 |
|---|---|---|---:|---:|---|
| v0 | The sweet red wine 'Commandaria' is produced on which island? | Cyprus. | ✓ | ✓ | VALID_EQUIVALENT; CORRECT |
| v1 | On which island is the sweet red wine known as 'Commandaria' made? | Cyprus. | ✓ | ✓ | VALID_EQUIVALENT; CORRECT |
| v2 | What is the island where 'Commandaria', the sweet red wine, is produced? | Commandaria. | ✗ | ✗ | VALID_EQUIVALENT; INCORRECT |
| v3 | Which island produces the sweet red wine named 'Commandaria'? | Cyprus. | ✓ | ✓ | VALID_EQUIVALENT; CORRECT |
| v4 | In which location is the sweet red wine 'Commandaria' crafted? | Commandaria is crafted in the town of Commandaria, located on the island of Cyprus. | ✓ | ✓ | VALID_BUT_BROADER_LOCATION_WORDING; island 대신 location이지만 원 target Cyprus로 답할 수 있음; TARGET_CORRECT_FALSE_EXTRA_CLAIM; Cyprus는 맞지만 'town of Commandaria'라는 부가 주장은 근거가 없고 공식 자료의 여러 생산 마을 설명과 충돌 |

### 1-4. dream · Missouri–New Orleans

- Fact ID: `qw_5761__e1270874009d`; gold: `New Orleans`
- 판정: `SOURCE_PREMISE_AMBIGUITY` / `SECONDARY`
- 메커니즘: route-relation retrieval failure
- 활용: 보조 사례로만 사용

| v | 질문 | 출력 | 자동 | 수동 target | 질문 감사/응답 감사 |
|---|---|---|---:|---:|---|
| v0 | If you travelled down the Missouri River from Bismarck, North Dakota, which city would you be in when you reached the ocean? | San Francisco, California | ✗ | ✗ | SOURCE_PREMISE_COMPRESSED; Missouri가 Mississippi에 합류한 뒤 Gulf로 간다는 경로를 생략; INCORRECT |
| v1 | Which city would you arrive at when traveling downstream on the Missouri River from Bismarck, North Dakota, until you reach the ocean? | New Orleans, Louisiana | ✓ | ✓ | VALID_EQUIVALENT; CORRECT |
| v2 | If you journeyed along the Missouri River starting in Bismarck, North Dakota, what city would you reach by the time you got to the oce… | New Orleans, Louisiana | ✓ | ✓ | VALID_EQUIVALENT; CORRECT |
| v3 | When following the Missouri River downstream from Bismarck, North Dakota, which city would be your destination by the time you reached… | New Orleans, Louisiana | ✓ | ✓ | VALID_EQUIVALENT; CORRECT |
| v4 | What city would you find yourself in while moving downstream on the Missouri River from Bismarck, North Dakota, upon reaching the ocea… | New Orleans, Louisiana | ✓ | ✓ | VALID_EQUIVALENT; CORRECT |

### 1-5. dream · Christmas Island–Australia

- Fact ID: `qw_11772__6fca5e284e4d`; gold: `Australia`
- 판정: `PARAPHRASE_ANSWER_TYPE_DRIFT` / `EXCLUDE`
- 메커니즘: country/nation cue removed by generic 'Who'
- 활용: v3 재생성 후 재실험

| v | 질문 | 출력 | 자동 | 수동 target | 질문 감사/응답 감사 |
|---|---|---|---:|---:|---|
| v0 | Christmas Island is administered by which country? | Australia | ✓ | ✓ | VALID_EQUIVALENT; CORRECT |
| v1 | Which country governs Christmas Island? | Australia | ✓ | ✓ | VALID_EQUIVALENT; CORRECT |
| v2 | Christmas Island falls under the administration of which nation? | Australia | ✓ | ✓ | VALID_EQUIVALENT; CORRECT |
| v3 | Who is responsible for the administration of Christmas Island? | The Administrator of the Northern Territory | ✗ | ✗ | INVALID_ANSWER_TYPE_DRIFT; country/nation 제약을 generic Who로 변경; INCORRECT |
| v4 | In which country is Christmas Island administratively managed? | Australia | ✓ | ✓ | VALID_EQUIVALENT; CORRECT |

### 1-6. qwen · Vectis–Isle of Wight

- Fact ID: `odql_12511__72ba0f388cf4`; gold: `Isle of Wight`
- 판정: `PROMPT_SENSITIVE_ASSOCIATIVE_COMPETITOR` / `CORE`
- 메커니즘: crown-dependency association → Isle of Man
- 활용: 핵심 사례; v1의 어색한 genitive는 작은 제한으로 명시

| v | 질문 | 출력 | 자동 | 수동 target | 질문 감사/응답 감사 |
|---|---|---|---:|---:|---|
| v0 | The Vectis National Party was a political party formed in 1967 which campaigned for crown dependency for which island? | Isle of Wight | ✓ | ✓ | VALID_EQUIVALENT; CORRECT |
| v1 | In 1967, the Vectis National Party was established to advocate for crown dependency of which island? | Isle of Man | ✗ | ✗ | VALID_EQUIVALENT; INCORRECT |
| v2 | Which island was the focus of the Vectis National Party's campaign for crown dependency, established in 1967? | Isle of Wight | ✓ | ✓ | VALID_EQUIVALENT; CORRECT |
| v3 | What island did the Vectis National Party, formed in 1967, seek to have recognized as a crown dependency? | Isle of Wight | ✓ | ✓ | VALID_EQUIVALENT; CORRECT |
| v4 | The Vectis National Party, created in 1967, aimed to campaign for crown dependency for which specific island? | Isle of Wight | ✓ | ✓ | VALID_EQUIVALENT; CORRECT |

### 1-7. qwen · ECTO-1–Ghostbusters

- Fact ID: `jp_1696__b0f3caa4d5fb`; gold: `The Ghostbusters`
- 판정: `PARAPHRASE_ANSWER_SCOPE_DRIFT` / `EXCLUDE`
- 메커니즘: group/team cue removed → individual answer
- 활용: 동치 paraphrase 분석에서 제외하고 answer-type ambiguity로 분리

| v | 질문 | 출력 | 자동 | 수동 target | 질문 감사/응답 감사 |
|---|---|---|---:|---:|---|
| v0 | What group of people drove a 1959 Chevrolet hearse with license plate ECTO-1? | Ghostbusters | ✓ | ✓ | VALID_EQUIVALENT; CORRECT |
| v1 | Which group was known for operating a 1959 Chevrolet hearse that had the license plate ECTO-1? | Ghostbusters | ✓ | ✓ | VALID_EQUIVALENT; CORRECT |
| v2 | Who drove the 1959 Chevrolet hearse featuring the license plate ECTO-1? | Dr. Peter Venkman | ✗ | ✗ | INVALID_ANSWER_SCOPE_DRIFT; group/team 제약이 사라져 개인과 집단을 모두 허용; INCORRECT |
| v3 | Can you identify the group that utilized a 1959 Chevrolet hearse with the license plate ECTO-1? | Ghostbusters | ✓ | ✓ | VALID_EQUIVALENT; CORRECT |
| v4 | What team drove a 1959 Chevrolet hearse sporting the license plate ECTO-1? | Ghostbusters | ✓ | ✓ | VALID_EQUIVALENT; CORRECT |

### 1-8. qwen · Solent–strait

- Fact ID: `qw_8009__df818dda9063`; gold: `A strait`
- 판정: `PROMPT_SENSITIVE_ENTITY_ECHO` / `CORE`
- 메커니즘: awkward metalinguistic construction → subject echo
- 활용: 사용 가능하되 v1 문장의 어색함을 명시

| v | 질문 | 출력 | 자동 | 수동 target | 질문 감사/응답 감사 |
|---|---|---|---:|---:|---|
| v0 | "In the United Kingdom, what is ""The Solent""?" | The Solent is a strait in the United Kingdom, located between the Isle of Wight and the mainland. It is a signific… | ✓ | ✓ | VALID_EQUIVALENT; CORRECT |
| v1 | What is referred to as 'The Solent' in the United Kingdom? | Solent | ✗ | ✗ | VALID_BUT_AWKWARD_METALINGUISTIC; 의도는 분류 질문이나 수동태 표현이 어색하고 약간 underspecified; INCORRECT |
| v2 | In the UK, what does the term 'The Solent' denote? | The Solent is a strait in the United Kingdom that separates the Isle of Wight from the mainland. It is a significa… | ✓ | ✓ | VALID_EQUIVALENT; CORRECT |
| v3 | What is the designation of 'The Solent' in the United Kingdom? | The Solent is a strait in the United Kingdom. | ✓ | ✓ | VALID_EQUIVALENT; CORRECT |
| v4 | How is 'The Solent' defined in the context of the United Kingdom? | The Solent is a strait in the United Kingdom that separates the Isle of Wight from the mainland. | ✓ | ✓ | VALID_EQUIVALENT; CORRECT |

### 1-9. qwen · Blue Fairy–Pinocchio

- Fact ID: `wh_2691__fc85150cefc4`; gold: `PINOCCHIO`
- 판정: `PARAPHRASE_RELATION_DRIFT` / `EXCLUDE`
- 메커니즘: brought to life → created
- 활용: v4 재생성 후 재실험

| v | 질문 | 출력 | 자동 | 수동 target | 질문 감사/응답 감사 |
|---|---|---|---:|---:|---|
| v0 | Which Disney character is brought to life by the Blue Fairy ? | Pinocchio | ✓ | ✓ | VALID_EQUIVALENT; CORRECT |
| v1 | Who is the Disney character who comes to life thanks to the Blue Fairy? | Pinocchio | ✓ | ✓ | VALID_EQUIVALENT; CORRECT |
| v2 | Which character from Disney is animated by the Blue Fairy's magic? | Pinocchio | ✓ | ✓ | VALID_EQUIVALENT; CORRECT |
| v3 | What Disney figure is given life through the Blue Fairy's intervention? | Pinocchio | ✓ | ✓ | VALID_EQUIVALENT; CORRECT |
| v4 | Which character in Disney lore is created by the Blue Fairy? | John Henry | ✗ | ✗ | INVALID_RELATION_DRIFT_FALSE_PREMISE; brought/given to life를 created로 바꿈; Pinocchio를 만든 이는 Geppetto; INCORRECT |

### 1-10. qwen · Jacob Epstein

- Fact ID: `sfq_18839__d81f22a291c7`; gold: `JACOB EPSTEIN`
- 판정: `ANSWER_TYPE_CUE_ABLATION` / `SECONDARY`
- 메커니즘: sculptor cue broadened to artist
- 활용: 순수 표면 paraphrase가 아닌 cue-ablation 사례로 분석

| v | 질문 | 출력 | 자동 | 수동 target | 질문 감사/응답 감사 |
|---|---|---|---:|---:|---|
| v0 | Which sculptor's works include St Michael's Victory over the Devil at Coventry Cathedral and Jacob and the Angel? | Sir Jacob Epstein | ✓ | ✓ | VALID_EQUIVALENT; CORRECT |
| v1 | Who is the sculptor known for creating St Michael's Victory over the Devil at Coventry Cathedral and Jacob and the Angel? | Sir Jacob Epstein | ✓ | ✓ | VALID_EQUIVALENT; CORRECT |
| v2 | Which artist created both St Michael's Victory over the Devil at Coventry Cathedral and Jacob and the Angel? | William Etty | ✗ | ✗ | VALID_FACT_BUT_TYPE_CUE_BROADENED; sculptor를 상위 범주 artist로 바꿔 retrieval cue 강도가 달라짐; INCORRECT |
| v3 | Identify the sculptor whose creations include St Michael's Victory over the Devil at Coventry Cathedral and Jacob and the Angel? | Sir Jacob Epstein | ✓ | ✓ | VALID_EQUIVALENT; CORRECT |
| v4 | Name the sculptor responsible for the works St Michael's Victory over the Devil at Coventry Cathedral and Jacob and the Angel? | Sir Jacob Epstein | ✓ | ✓ | VALID_EQUIVALENT; CORRECT |

### 1-11. qwen · Paul Scott–Raj Quartet

- Fact ID: `wh_4381__e2162a7442d0`; gold: `RAJ QUARTET`
- 판정: `CLEAN_PROMPT_SENSITIVE_FAILURE` / `CORE`
- 메커니즘: unsupported title substitution
- 활용: 가장 강한 Qwen 핵심 사례

| v | 질문 | 출력 | 자동 | 수동 target | 질문 감사/응답 감사 |
|---|---|---|---:|---:|---|
| v0 | By whatname are the four novels of Paul Scott best known, the last of which is entitled A Division of Spoils ? | The four novels of Paul Scott, including the last one titled "A Division of Spoils," are collectively known as the… | ✓ | ✓ | VALID_EQUIVALENT; CORRECT |
| v1 | What is the commonly used title for the four novels written by Paul Scott, with the final one called A Division of Spoils? | The commonly used title for the four novels written by Paul Scott, with the final one called A Division of Spoils,… | ✓ | ✓ | VALID_EQUIVALENT; CORRECT |
| v2 | Under what title are Paul Scott's four novels collectively recognized, the last of these being A Division of Spoils? | The Four Guineas | ✗ | ✗ | VALID_EQUIVALENT; INCORRECT |
| v3 | What name refers to the quartet of novels authored by Paul Scott, with the last one named A Division of Spoils? | The name that refers to the quartet of novels authored by Paul Scott, with the last one named A Division of Spoils… | ✓ | ✓ | VALID_EQUIVALENT; CORRECT |
| v4 | Which title is primarily associated with Paul Scott's four novels, the final entry being A Division of Spoils? | Raj Quartet | ✓ | ✓ | VALID_EQUIVALENT; CORRECT |

## Part 2. 지식 통제 모델 비교 12 facts · 60 질문쌍

여기서 CORE는 현재 질문 전체 또는 명시된 유효 subset을 핵심 비교에 쓸 수 있는 군, QUALIFIED는 재채점·부분 사용·재생성이 필요한 군, EXCLUDE는 관측된 모델 차이가 평가/질문 오류로 사라져 현재 contrast에서 빼야 하는 군이다.

| 사례 | Dream | Qwen | 등급 | 유효/제외 v | 수동 결론 |
|---|---:|---:|---|---|---|
| ECTO-1–Ghostbusters | 5/5 `11111` | 4/5 `11011` | EXCLUDE | 유효 v0,v1,v3,v4; 제외 v2 | 자동 Dream 5/5 대 Qwen 4/5 차이는 v2의 target drift와 Dream scorer false positive가 만든 인공 차이다. 유효 4개에서는 둘 다 안정적이다. |
| Commandaria–Cyprus | 4/5 `11011` | 5/5 `11111` | CORE | 유효 v0,v1,v2,v3,v4 | v2에서만 Dream이 질문 속 와인명을 복사하고 Qwen은 Cyprus를 회수한다. v4 Dream은 target은 맞지만 허위 부가 설명을 생성한다. |
| Bos primigenius–Aurochs | 3/5 `11010` | 5/5 `11111` | CORE | 유효 v0,v1,v2,v3,v4 | Dream은 v2·v4에서 Mammoth로 이동하고 Qwen은 5개 모두 Aurochs다. 핵심 단서가 모두 유지된 강한 모델 차이다. |
| Eve–Adam's rib | 4/5 `11110` | 5/5 `11111` | EXCLUDE | 유효 v0,v1,v2,v3,v4 | Dream v4도 의미상 정답이므로 두 모델 모두 실제 5/5다. 자동 모델 차이가 사라진다. |
| Moulin Rouge–Ewan McGregor | 3/5 `11001` | 5/5 `11111` | CORE | 유효 v0,v1,v2,v3,v4 | Dream은 v2·v3에서 Guy Pearce, Qwen은 전부 Ewan McGregor다. 질문 의미와 actor target이 유지된다. |
| McKinley–assassination | 3/5 `00111` | 5/5 `11111` | QUALIFIED | 유효 v0,v1,v2,v3,v4 | Dream v0은 사건 서술형 정답이라 false negative다. 수동으로는 Dream 4/5, Qwen 5/5이며 실제 차이는 v1 pneumonia다. |
| Christmas Island–Australia | 4/5 `11101` | 5/5 `11111` | QUALIFIED | 유효 v0,v1,v2,v4; 제외 v3 | 유일한 모델 차이 v3가 country/nation 제약을 제거했다. 유효한 4개에서는 둘 다 Australia다. |
| Solent–strait | 5/5 `11111` | 4/5 `10111` | CORE | 유효 v0,v1,v2,v3,v4 | v1에서 Qwen만 Solent를 되풀이한다. 실제 non-answer지만 v1 문장이 어색해 causal evidence의 강도는 낮춘다. |
| Clumber–Spaniel | 3/5 `10110` | 2/5 `10100` | QUALIFIED | 유효 v0,v2; 제외 v1,v3,v4 | v1·v3·v4가 breed family에서 더 넓은 dog group으로 target을 바꾼다. Sporting은 literal하게 유효해 자동 3/5·2/5를 지식 차이로 읽을 수 없다. |
| Olympics twice–Paris | 2/5 `11000` | 2/5 `10100` | CORE | 유효 v0,v1,v2,v3; 제외 v4 | 유효한 v0–v3에서 v1은 Dream만, v2는 Qwen만 맞고 v3은 서로 다른 오답이다. v4만 비문이라 제외하면 좋은 양방향 비교다. |
| Jim Rockford–James Garner | 3/5 `11010` | 5/5 `11111` | CORE | 유효 v0,v1,v2,v3,v4 | Dream은 v2·v4에서 배우를 묻는데 배역 Jim Rockford를 복사한다. Qwen은 5개 모두 James Garner다. |
| Blue Fairy–Pinocchio | 3/5 `10110` | 4/5 `11110` | QUALIFIED | 유효 v0,v1,v2,v3; 제외 v4 | 유효한 v0–v3에서 Dream만 v1을 Tinker Bell로 틀린다. v4는 created라는 잘못된 관계라 두 모델 오답을 비교하면 안 된다. |

### 2-1. ECTO-1–Ghostbusters

- Fact ID: `jp_1696__b0f3caa4d5fb`; gold: `The Ghostbusters`
- 판정: `TARGET_DRIFT_AND_DREAM_FALSE_POSITIVE` / `EXCLUDE`
- 결론: 자동 Dream 5/5 대 Qwen 4/5 차이는 v2의 target drift와 Dream scorer false positive가 만든 인공 차이다. 유효 4개에서는 둘 다 안정적이다.

| v | 질문 | Dream (자동/target/literal) | Qwen (자동/target/literal) | 비교 가능성 | 분석 |
|---|---|---|---|---|---|
| v0 | What group of people drove a 1959 Chevrolet hearse with license plate ECTO-1? | The Ghostbusters (✓/✓/✓) | Ghostbusters (✓/✓/✓) | VALID_EQUIVALENT | 양쪽 모두 target을 회수 |
| v1 | Which group was known for operating a 1959 Chevrolet hearse that had the license plate ECTO-1? | The Ghostbusters (✓/✓/✓) | Ghostbusters (✓/✓/✓) | VALID_EQUIVALENT | 양쪽 모두 target을 회수 |
| v2 | Who drove the 1959 Chevrolet hearse featuring the license plate ECTO-1? | Ecto-1 is the license plate of the 1959 Chevrolet hearse driven by Egon… (✓/✗/✗) | Dr. Peter Venkman (✗/✗/✗) | INVALID_ANSWER_SCOPE_DRIFT | 둘 다 집단 대신 개인을 답함. Dream은 문맥 속 영화명 때문에 자동 false positive. |
| v3 | Can you identify the group that utilized a 1959 Chevrolet hearse with the license plate ECTO-1? | The Ghostbusters (✓/✓/✓) | Ghostbusters (✓/✓/✓) | VALID_EQUIVALENT | 양쪽 모두 target을 회수 |
| v4 | What team drove a 1959 Chevrolet hearse sporting the license plate ECTO-1? | The Ghostbusters (✓/✓/✓) | Ghostbusters (✓/✓/✓) | VALID_EQUIVALENT | 양쪽 모두 target을 회수 |

### 2-2. Commandaria–Cyprus

- Fact ID: `odql_11052__8cf11607ec01`; gold: `CYPRUS`
- 판정: `ROBUST_DREAM_SPECIFIC_ROLE_BINDING_FAILURE` / `CORE`
- 결론: v2에서만 Dream이 질문 속 와인명을 복사하고 Qwen은 Cyprus를 회수한다. v4 Dream은 target은 맞지만 허위 부가 설명을 생성한다.

| v | 질문 | Dream (자동/target/literal) | Qwen (자동/target/literal) | 비교 가능성 | 분석 |
|---|---|---|---|---|---|
| v0 | The sweet red wine 'Commandaria' is produced on which island? | Cyprus. (✓/✓/✓) | Cyprus (✓/✓/✓) | VALID_EQUIVALENT | 양쪽 모두 target을 회수 |
| v1 | On which island is the sweet red wine known as 'Commandaria' made? | Cyprus. (✓/✓/✓) | Cyprus (✓/✓/✓) | VALID_EQUIVALENT | 양쪽 모두 target을 회수 |
| v2 | What is the island where 'Commandaria', the sweet red wine, is produced? | Commandaria. (✗/✗/✗) | Cyprus (✓/✓/✓) | VALID_EQUIVALENT | Dream만 질문의 와인 entity를 복사; Qwen은 위치 entity 회수. |
| v3 | Which island produces the sweet red wine named 'Commandaria'? | Cyprus. (✓/✓/✓) | Cyprus (✓/✓/✓) | VALID_EQUIVALENT | 양쪽 모두 target을 회수 |
| v4 | In which location is the sweet red wine 'Commandaria' crafted? | Commandaria is crafted in the town of Commandaria, located on the islan… (✓/✓/✓) | Cyprus (✓/✓/✓) | VALID_BUT_BROADER_LOCATION_WORDING | 둘 다 target 정답. Dream은 존재하지 않는 것으로 보이는 'town of Commandaria'를 덧붙임. |

### 2-3. Bos primigenius–Aurochs

- Fact ID: `odql_13017__735a489e6be8`; gold: `AUROCHS`
- 판정: `ROBUST_DREAM_SPECIFIC_ASSOCIATIVE_FAILURE` / `CORE`
- 결론: Dream은 v2·v4에서 Mammoth로 이동하고 Qwen은 5개 모두 Aurochs다. 핵심 단서가 모두 유지된 강한 모델 차이다.

| v | 질문 | Dream (자동/target/literal) | Qwen (자동/target/literal) | 비교 가능성 | 분석 |
|---|---|---|---|---|---|
| v0 | What is the common name for the species of large cattle, Bos primigenius, once prevalent in Europe, that was hunted to extincti… | Aurochs (✓/✓/✓) | Aurochs (✓/✓/✓) | VALID_EQUIVALENT | 양쪽 모두 target을 회수 |
| v1 | What is the widely known name for the large cattle species, Bos primigenius, that became extinct in Europe in 1627 after being … | aurochs (✓/✓/✓) | Aurochs (✓/✓/✓) | VALID_EQUIVALENT | 양쪽 모두 target을 회수 |
| v2 | What do we commonly call the species of large cattle, known scientifically as Bos primigenius, that was hunted to extinction in… | Mammoth (✗/✗/✗) | Aurochs (✓/✓/✓) | VALID_EQUIVALENT | Dream Mammoth 대 Qwen Aurochs. |
| v3 | What is the familiar name for the extinct large cattle species, Bos primigenius, which was prevalent in Europe and hunted to ex… | Aurochs (✓/✓/✓) | Aurochs (✓/✓/✓) | VALID_EQUIVALENT | 양쪽 모두 target을 회수 |
| v4 | What is the everyday name for the species of large cattle called Bos primigenius, which was prevalent in Europe and became exti… | Mammoth (✗/✗/✗) | Aurochs (✓/✓/✓) | VALID_EQUIVALENT | Dream Mammoth 대 Qwen Aurochs; v2와 동일 경쟁자 재현. |

### 2-4. Eve–Adam's rib

- Fact ID: `odql_1396__8dddda2c19d8`; gold: `ADAM'S RIB`
- 판정: `DREAM_SCORER_FALSE_NEGATIVE` / `EXCLUDE`
- 결론: Dream v4도 의미상 정답이므로 두 모델 모두 실제 5/5다. 자동 모델 차이가 사라진다.

| v | 질문 | Dream (자동/target/literal) | Qwen (자동/target/literal) | 비교 가능성 | 분석 |
|---|---|---|---|---|---|
| v0 | According to the Bible, what was Eve made from? | Adam's rib (✓/✓/✓) | Adam's rib (✓/✓/✓) | VALID_EQUIVALENT | 양쪽 모두 target을 회수 |
| v1 | What material was used to create Eve, as described in the Bible? | Adam's rib (✓/✓/✓) | Adam's rib (✓/✓/✓) | VALID_EQUIVALENT | 양쪽 모두 target을 회수 |
| v2 | In the biblical account, from what substance was Eve formed? | From Adam's rib. (✓/✓/✓) | Adam's rib (✓/✓/✓) | VALID_EQUIVALENT | 양쪽 모두 target을 회수 |
| v3 | What was the origin of Eve's creation according to the Bible? | Eve was created from Adam's rib. (✓/✓/✓) | Adam's rib (✓/✓/✓) | VALID_EQUIVALENT | 양쪽 모두 target을 회수 |
| v4 | According to the scriptures, what was Eve created from? | The rib of Adam (✗/✓/✓) | Adam's rib (✓/✓/✓) | VALID_EQUIVALENT | Dream 'The rib of Adam'도 의미상 정답; 자동 차이는 사라짐. |

### 2-5. Moulin Rouge–Ewan McGregor

- Fact ID: `qf_2121__a630bba15b58`; gold: `Ewan McGregor`
- 판정: `ROBUST_DREAM_SPECIFIC_COMPETITOR_FAILURE` / `CORE`
- 결론: Dream은 v2·v3에서 Guy Pearce, Qwen은 전부 Ewan McGregor다. 질문 의미와 actor target이 유지된다.

| v | 질문 | Dream (자동/target/literal) | Qwen (자동/target/literal) | 비교 가능성 | 분석 |
|---|---|---|---|---|---|
| v0 | The male lead roll was played by whom in the 2001 film Moulin Rouge? | Ewan McGregor (✓/✓/✓) | Ewan McGregor (✓/✓/✓) | VALID_EQUIVALENT | 양쪽 모두 target을 회수 |
| v1 | Who portrayed the male lead in the film Moulin Rouge released in 2001? | Ewan McGregor (✓/✓/✓) | Ewan McGregor (✓/✓/✓) | VALID_EQUIVALENT | 양쪽 모두 target을 회수 |
| v2 | In the 2001 movie Moulin Rouge, who took on the role of the male lead? | Guy Pearce (✗/✗/✗) | Ewan McGregor (✓/✓/✓) | VALID_EQUIVALENT | Dream Guy Pearce 대 Qwen Ewan McGregor. |
| v3 | Which actor played the main male character in the 2001 film Moulin Rouge? | Guy Pearce (✗/✗/✗) | Ewan McGregor (✓/✓/✓) | VALID_EQUIVALENT | Dream이 같은 경쟁자 Guy Pearce를 반복; Qwen은 정답. |
| v4 | Who was the performer of the male lead role in Moulin Rouge, a film from 2001? | Ewan McGregor (✓/✓/✓) | Ewan McGregor (✓/✓/✓) | VALID_EQUIVALENT | 양쪽 모두 target을 회수 |

### 2-6. McKinley–assassination

- Fact ID: `qw_10823__e181c7f9b649`; gold: `Assassination`
- 판정: `USABLE_AFTER_MANUAL_RESCORING` / `QUALIFIED`
- 결론: Dream v0은 사건 서술형 정답이라 false negative다. 수동으로는 Dream 4/5, Qwen 5/5이며 실제 차이는 v1 pneumonia다.

| v | 질문 | Dream (자동/target/literal) | Qwen (자동/target/literal) | 비교 가능성 | 분석 |
|---|---|---|---|---|---|
| v0 | How did US president William McKinley die in 1901? | Shot in the stomach by a man. (✗/✓/✓) | Assassination (✓/✓/✓) | VALID_EQUIVALENT | Dream은 assassination 라벨 대신 피격을 서술했으므로 의미상 정답으로 교정. |
| v1 | What was the cause of death for US president William McKinley in 1901? | The cause of death for US president William McKinley in 1901 was pneumo… (✗/✗/✗) | Assassination (✓/✓/✓) | VALID_EQUIVALENT | Dream pneumonia 대 Qwen assassination; 실제 모델 차이. |
| v2 | In what manner did US president William McKinley pass away in 1901? | Shot and killed by an assassin (✓/✓/✓) | Assassination (✓/✓/✓) | VALID_EQUIVALENT | 양쪽 모두 target을 회수 |
| v3 | What circumstances led to the death of US president William McKinley in 1901? | William McKinley was assassinated by Leonolgosz on September 6, 1901, i… (✓/✓/✓) | Assassination (✓/✓/✓) | VALID_EQUIVALENT | 양쪽 모두 assassination target은 맞지만 Dream이 Leon Czolgosz를 'Leonolgosz'로 훼손. |
| v4 | How did William McKinley, the US president, meet his end in 1901? | William McKinley was assassinated by Leonolgosz in Buffalo, New York on… (✓/✓/✓) | Assassination (✓/✓/✓) | VALID_EQUIVALENT | v3과 같은 target 정답 + 잘못 생성된 범인 이름이 반복됨. |

### 2-7. Christmas Island–Australia

- Fact ID: `qw_11772__6fca5e284e4d`; gold: `Australia`
- 판정: `ANSWER_TYPE_DRIFT` / `QUALIFIED`
- 결론: 유일한 모델 차이 v3가 country/nation 제약을 제거했다. 유효한 4개에서는 둘 다 Australia다.

| v | 질문 | Dream (자동/target/literal) | Qwen (자동/target/literal) | 비교 가능성 | 분석 |
|---|---|---|---|---|---|
| v0 | Christmas Island is administered by which country? | Australia (✓/✓/✓) | Australia (✓/✓/✓) | VALID_EQUIVALENT | 양쪽 모두 target을 회수 |
| v1 | Which country governs Christmas Island? | Australia (✓/✓/✓) | Australia (✓/✓/✓) | VALID_EQUIVALENT | 양쪽 모두 target을 회수 |
| v2 | Christmas Island falls under the administration of which nation? | Australia (✓/✓/✓) | Australia (✓/✓/✓) | VALID_EQUIVALENT | 양쪽 모두 target을 회수 |
| v3 | Who is responsible for the administration of Christmas Island? | The Administrator of the Northern Territory (✗/✗/✗) | Australia (✓/✓/✓) | INVALID_ANSWER_TYPE_DRIFT | 질문의 answer type이 변한 유일한 차이이므로 causal 비교 불가. |
| v4 | In which country is Christmas Island administratively managed? | Australia (✓/✓/✓) | Australia (✓/✓/✓) | VALID_EQUIVALENT | 양쪽 모두 target을 회수 |

### 2-8. Solent–strait

- Fact ID: `qw_8009__df818dda9063`; gold: `A strait`
- 판정: `QWEN_ENTITY_ECHO_ON_AWKWARD_PROMPT` / `CORE`
- 결론: v1에서 Qwen만 Solent를 되풀이한다. 실제 non-answer지만 v1 문장이 어색해 causal evidence의 강도는 낮춘다.

| v | 질문 | Dream (자동/target/literal) | Qwen (자동/target/literal) | 비교 가능성 | 분석 |
|---|---|---|---|---|---|
| v0 | "In the United Kingdom, what is ""The Solent""?" | The Solent is a strait in the United Kingdom that separates the Isle of… (✓/✓/✓) | The Solent is a strait in the United Kingdom, located between the Isle … (✓/✓/✓) | VALID_EQUIVALENT | 양쪽 모두 target을 회수 |
| v1 | What is referred to as 'The Solent' in the United Kingdom? | The Solent is a strait in the United Kingdom that separates the Isle of… (✓/✓/✓) | Solent (✗/✗/✗) | VALID_BUT_AWKWARD_METALINGUISTIC | Qwen은 질문의 subject만 되풀이한 non-answer; Dream은 strait 회수. |
| v2 | In the UK, what does the term 'The Solent' denote? | The Solent is a strait in the English Channel, separating the Isle of W… (✓/✓/✓) | The Solent is a strait in the United Kingdom that separates the Isle of… (✓/✓/✓) | VALID_EQUIVALENT | 양쪽 모두 target을 회수 |
| v3 | What is the designation of 'The Solent' in the United Kingdom? | The Solent is a strait in the United Kingdom that separates the Isle of… (✓/✓/✓) | The Solent is a strait in the United Kingdom. (✓/✓/✓) | VALID_EQUIVALENT | 양쪽 모두 target을 회수 |
| v4 | How is 'The Solent' defined in the context of the United Kingdom? | The Solent is a strait in the United Kingdom that separates the Isle of… (✓/✓/✓) | The Solent is a strait in the United Kingdom that separates the Isle of… (✓/✓/✓) | VALID_EQUIVALENT | 양쪽 모두 target을 회수 |

### 2-9. Clumber–Spaniel

- Fact ID: `sfq_19011__697b024818fc`; gold: `Spaniel`
- 판정: `MULTI_VARIANT_TARGET_LEVEL_DRIFT` / `QUALIFIED`
- 결론: v1·v3·v4가 breed family에서 더 넓은 dog group으로 target을 바꾼다. Sporting은 literal하게 유효해 자동 3/5·2/5를 지식 차이로 읽을 수 없다.

| v | 질문 | Dream (자동/target/literal) | Qwen (자동/target/literal) | 비교 가능성 | 분석 |
|---|---|---|---|---|---|
| v0 | A 'Clumber' is a breed of which dog? | Spaniel (✓/✓/✓) | A 'Clumber' is a breed of a dog called a Clumber Spaniel. It's a big, f… (✓/✓/✓) | VALID_EQUIVALENT | 양쪽 모두 target을 회수 |
| v1 | A Clumber belongs to which general type of dog? | Sporting dog (✗/✗/✓) | Sporting (✗/✗/✓) | INVALID_TARGET_LEVEL_DRIFT | 양쪽 Sporting 답은 broader category 질문에는 유효; 원 gold와 target level이 다름. |
| v2 | Complete the dog type: Clumber what? | Spaniel (✓/✓/✓) | Spaniel (✓/✓/✓) | VALID_EQUIVALENT | 양쪽 모두 target을 회수 |
| v3 | Clumber is a breed in which broader dog category? | Spaniels (✓/✓/✓) | Terrier (✗/✗/✗) | INVALID_AMBIGUOUS_CATEGORY_LEVEL | Dream Spaniels, Qwen Terrier지만 category level 자체가 모호. |
| v4 | Which general dog type includes the Clumber breed? | Sporting dog (✗/✗/✓) | Sporting (✗/✗/✓) | INVALID_TARGET_LEVEL_DRIFT | 양쪽 Sporting 답은 literal하게 유효해 자동 both-wrong이 오해를 낳음. |

### 2-10. Olympics twice–Paris

- Fact ID: `sfq_3251__eb8293631a4b`; gold: `PARIS`
- 판정: `BOTH_SENSITIVE_EXCLUDE_BAD_V4` / `CORE`
- 결론: 유효한 v0–v3에서 v1은 Dream만, v2는 Qwen만 맞고 v3은 서로 다른 오답이다. v4만 비문이라 제외하면 좋은 양방향 비교다.

| v | 질문 | Dream (자동/target/literal) | Qwen (자동/target/literal) | 비교 가능성 | 분석 |
|---|---|---|---|---|---|
| v0 | Which was the first city to host the Summer Olympics twice? | Paris (✓/✓/✓) | Paris (✓/✓/✓) | VALID_EQUIVALENT | 양쪽 모두 target을 회수 |
| v1 | What was the first city to hold the Summer Olympics on two occasions? | Paris (✓/✓/✓) | St. Louis (✗/✗/✗) | VALID_EQUIVALENT | Dream Paris 정답, Qwen St. Louis 오답. |
| v2 | Which city was the inaugural host of the Summer Olympics for a second time? | Athens (✗/✗/✗) | Paris (✓/✓/✓) | VALID_EQUIVALENT | Dream Athens 오답, Qwen Paris 정답. |
| v3 | What city became the first to welcome the Summer Olympics twice? | London, England (✗/✗/✗) | Los Angeles (✗/✗/✗) | VALID_EQUIVALENT | Dream London, Qwen Los Angeles로 서로 다른 오답. |
| v4 | Which city hosted the Summer Olympics for the first time on two different instances? | Athens, Greece (✗/✗/✗) | Los Angeles (✗/✗/✗) | INVALID_UNGRAMMATICAL_AMBIGUITY | 질문이 비문/모호하며 둘 다 오답; 제외. |

### 2-11. Jim Rockford–James Garner

- Fact ID: `sfq_7552__2766dae6d17b`; gold: `James Garner`
- 판정: `ROBUST_DREAM_SPECIFIC_ROLE_BINDING_FAILURE` / `CORE`
- 결론: Dream은 v2·v4에서 배우를 묻는데 배역 Jim Rockford를 복사한다. Qwen은 5개 모두 James Garner다.

| v | 질문 | Dream (자동/target/literal) | Qwen (자동/target/literal) | 비교 가능성 | 분석 |
|---|---|---|---|---|---|
| v0 | On TV who played the private eye Jim Rockford? | James Garner. (✓/✓/✓) | James Garner (✓/✓/✓) | VALID_EQUIVALENT | 양쪽 모두 target을 회수 |
| v1 | Who portrayed the detective Jim Rockford on television? | Jim Rockford was portrayed by James Garner. (✓/✓/✓) | James Garner (✓/✓/✓) | VALID_EQUIVALENT | 양쪽 모두 target을 회수 |
| v2 | Which actor played the role of private investigator Jim Rockford in the TV series? | Jim Rockford (✗/✗/✗) | James Garner (✓/✓/✓) | VALID_EQUIVALENT | Dream은 요구 actor 대신 질문 속 role entity Jim Rockford 복사. |
| v3 | In the television show, who was cast as the private eye Jim Rockford? | James Garner (✓/✓/✓) | James Garner (✓/✓/✓) | VALID_EQUIVALENT | 양쪽 모두 target을 회수 |
| v4 | Who was the actor that took on the role of Jim Rockford, the private investigator, in the TV program? | Jim Rockford (✗/✗/✗) | James Garner (✓/✓/✓) | VALID_EQUIVALENT | Dream이 동일 role-copy 오류를 재현; Qwen은 정답. |

### 2-12. Blue Fairy–Pinocchio

- Fact ID: `wh_2691__fc85150cefc4`; gold: `PINOCCHIO`
- 판정: `DREAM_SENSITIVE_AFTER_EXCLUDING_FALSE_PREMISE` / `QUALIFIED`
- 결론: 유효한 v0–v3에서 Dream만 v1을 Tinker Bell로 틀린다. v4는 created라는 잘못된 관계라 두 모델 오답을 비교하면 안 된다.

| v | 질문 | Dream (자동/target/literal) | Qwen (자동/target/literal) | 비교 가능성 | 분석 |
|---|---|---|---|---|---|
| v0 | Which Disney character is brought to life by the Blue Fairy ? | Pinocchio (✓/✓/✓) | Pinocchio (✓/✓/✓) | VALID_EQUIVALENT | 양쪽 모두 target을 회수 |
| v1 | Who is the Disney character who comes to life thanks to the Blue Fairy? | Tinker Bell (✗/✗/✗) | Pinocchio (✓/✓/✓) | VALID_EQUIVALENT | Dream Tinker Bell 대 Qwen Pinocchio; 유효 질문에서 실제 차이. |
| v2 | Which character from Disney is animated by the Blue Fairy's magic? | Pinocchio (✓/✓/✓) | Pinocchio (✓/✓/✓) | VALID_EQUIVALENT | 양쪽 모두 target을 회수 |
| v3 | What Disney figure is given life through the Blue Fairy's intervention? | Pinocchio (✓/✓/✓) | Pinocchio (✓/✓/✓) | VALID_EQUIVALENT | 양쪽 모두 target을 회수 |
| v4 | Which character in Disney lore is created by the Blue Fairy? | Tiana (✗/✗/✗) | John Henry (✗/✗/✗) | INVALID_RELATION_DRIFT_FALSE_PREMISE | brought to life가 created로 바뀐 false premise; 두 오답을 지식 실패로 세면 안 됨. |

## 오류 메커니즘별 실제 사례

- **role/entity copy:** Dream `Commandaria→Commandaria`, `actor→Jim Rockford`; Qwen `definition→Solent`.
- **same-domain competitor:** Dream `Aurochs→Mammoth`, `Ewan McGregor→Guy Pearce`, `Colonel→General`; Qwen `Isle of Wight→Isle of Man`.
- **unsupported title/entity:** Qwen `Raj Quartet→The Four Guineas`, `Jacob Epstein→William Etty`.
- **데이터/평가 confound:** `The rib of Adam` false negative, Ghostbusters 문맥 문자열 false positive, Christmas Island·Clumber·Pinocchio의 target/relation drift.

## 사실 확인 메모

- UNESCO는 Commandaria를 Troodos 남사면의 여러 마을에서 만드는 와인으로 설명한다. 따라서 Dream의 `town of Commandaria`는 target은 맞아도 부가 사실은 신뢰할 수 없다: https://ich.unesco.org/en/RL/commandaria-wine-02288
- 호주 정부는 Christmas Island를 Australian Government가 관할하며 Indian Ocean Territories Administrator가 대표한다고 설명한다. `Administrator of the Northern Territory`는 부정확하다: https://www.infrastructure.gov.au/territories-regions-cities/territories/indian-ocean-territories/christmas-island/governance-administration
- AKC에서 Clumber Spaniel은 Sporting Group에 속한다. 따라서 broader-category 질문에서 `Sporting`은 단순 오답이 아니다: https://www.akc.org/breeder-programs/breed-health-testing-requirements/sporting-group-health-testing-requirements/
- Disney 자료는 Blue Fairy가 Pinocchio를 *brought to life*했다고 설명한다. `created by`는 동일 관계가 아니다: https://d23.com/a-to-z/pinocchio-film/
- 미국 NPS는 McKinley가 암살자의 총탄에 맞고 8일 뒤 상처로 사망했다고 설명한다. Dream v0은 표현이 거칠지만 assassination 사건을 서술한다: https://www.nps.gov/places/mckinley-monument.htm

## 산출물 해석 주의

`manual_target_is_correct`는 원래 TriviaQA gold target을 회수했는지, `manual_literal_is_correct`는 drift된 질문 자체에 문자 그대로 맞는지를 구분한다. 예를 들어 Clumber의 `Sporting`은 후자에는 맞지만 전자에는 맞지 않는다. `manual_response_quality`는 target이 포함돼도 허위 부가 설명이 있는 경우를 보존한다.
