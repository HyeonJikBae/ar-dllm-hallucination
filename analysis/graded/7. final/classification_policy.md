# PopQA 오답 유형 분류 — 최종 정책

Qwen2.5-7B가 PopQA 11,045개 사실(사실마다 질문 5개, 합계 55,225행)에 낸 답을 분류한 최종 결과와, 그 분류에 쓴 정책이다.
최종 파일은 `qwen2.5-7b_popqa_final.jsonl`이고, 이 문서의 코드는 `code/`, 에이전트에게 준 프롬프트 원문은 `agent_prompt_en.md`에 있다.

## 1. 최종 파일의 열

| 열 | 뜻 |
|---|---|
| `sample_id`, `version` | 사실 ID와 질문 변형(`question`, `v1`~`v4`) |
| `relation` | PopQA 관계(director, capital 등) |
| `question`, `model_answer`, `gold_value`, `gold_aliases` | 질문, 모델 답, 정답, 정답 별칭 |
| `verdict_original` | 처음 채점한 값(`correct`, `incorrect`, `uncertain`). 어느 단계에서도 바꾸지 않았다 |
| **`final_label`** | **최종 분류.** 아래 값 중 하나 |
| `final_conf` | 라벨이 확실하면 `true`. 규칙이 확신했거나 에이전트가 확신한 경우다. `uncertain` 행은 `null` |
| `label_source` | 라벨을 정한 방법(5절) |
| `k_correct`, `fact_group` | 그 사실의 5개 질문 중 정답으로 센 개수와 `0/5`~`5/5`. `uncertain`이거나 정답이 의심스러운 행은 세지 않는다 |
| **`analysis_use`** | SAE 분석에 실제로 쓴 행이면 `true` |
| `gold_issue` | 정답 자체가 의심스러운 행 |
| `audit_flag` | 정답으로 처리됐지만 점검에서 정답이 아니라고 나온 행이면 `not_correct`. 값이 없는 행이 대부분이다 |
| `note` | 라벨을 바꾼 근거(80자 이하) |

### `final_label` 값

| 값 | 의미 | 행 수 |
|---|---|---|
| `correct` | 원래 정답 | 11,073 |
| `correct_lenient` | 별칭, 철자, 정답이 함의하는 상위어 등으로 정답에 가깝다고 보아 정답으로 제안한 행. 원본 `verdict`는 `incorrect`다 | 1,817 |
| `1` | 관련 엔티티: 질문이 요구하는 역할·영역이 정답과 공통인 실존 엔티티 | 28,635 |
| `2` | 무관 엔티티: 실존하지만 그 공통점이 없는 엔티티 | 11,114 |
| `3` | 질문 속 제목·표현을 그대로 답으로 복사 | 1,239 |
| `4` | 정답 뒤에 별개의 거짓 주장이 붙음 | 2 |
| `5` | 질문 조각과 일반어를 이어 붙인 비실재 이름 | 98 |
| `6` | 붕괴: 빈 출력, 같은 구의 반복, 단어 중간 절단, 프롬프트 누출, 의미 없는 문자열 | 922 |
| `abstain` | 기권: "모른다", "정보 없음" | 242 |
| `uncertain` | 원래 채점이 불확실해서 분류하지 않음 | 83 |

오답으로 분류한 행은 42,252행(1~6과 기권)이고, 비율은 유형 1 67.8%, 유형 2 26.3%, 유형 3 2.9%, 유형 5 0.2%, 유형 6 2.2%, 기권 0.6%이다. 유형 4는 2행뿐이라 분석에 쓸 수 없다.

## 2. 판정 순서

한 행의 답은 아래 순서로 첫 번째로 해당하는 것으로 정한다.

1. 정답 인정(`correct`, `correct_lenient`)
2. 6 붕괴
3. 기권
4. 4 정답 뒤 거짓
5. 5 재조합
6. 3 복사
7. 1 관련 / 2 무관

정답 인정 기준: 같은 엔티티의 별칭·철자·대소문자·경칭·접미사 차이, 정답이 참이면 반드시 참인 상위 범주, 장르·종교·스포츠·직업에서 정답보다 한 단계 아래의 같은 계열. 형제 범주(rock vs pop, 다른 종교)나 다른 엔티티, 인물·장소에서 정답보다 좁은 답은 정답이 아니다. **애매하면 정답으로 처리한다.** 아는 사실에 애매한 행이 섞이면 분석이 흐려지기 때문이다.

## 3. 어떤 방법으로 정했나

| 유형 | 방법 |
|---|---|
| 정답 인정(별칭·짧은 이름) | 문자열 규칙(`code/string_rules_prep.py`의 `C-alias`, `C-short`) |
| 6 붕괴(빈 출력, 프롬프트 누출, 반복) | 문자열 규칙(`6-empty`, `6-leak`, `6-loop`) |
| 6 붕괴(절단, 의미 없는 문자열) | 쌍 단위로 에이전트가 판정(`6-trunc`) |
| 기권 | 문자열 규칙: 14단어 이하의 답에 "not specified", "no information", "I don't know" 등이 있으면 기권 |
| 3 복사 | 문자열 규칙: 답에서 경칭·관사 등을 뺀 단어열이 질문 속에 그대로 들어 있으면 복사(`3-copy`). 일부는 에이전트가 판정 |
| 4 정답 뒤 거짓, 5 재조합 | 에이전트가 쌍 단위로 판정(`4-extra`, `5-splice`) |
| **1 / 2** | **Wikidata 규칙 + 직접 정한 그룹 + 에이전트**(4절). 이 정책에서 가장 공을 들인 부분 |

판정 단위는 행이 아니라 **(관계, 정답, 답) 쌍**이다. 같은 쌍은 모든 행에서 같은 라벨을 받는다.

## 4. 유형 1과 2의 기준

공통점은 "사람이다"처럼 모든 답에 해당하는 것이 아니라 **질문이 요구하는 역할이나 영역**이다.

| 관계 | 유형 1 (공통점 있음) | 유형 2 |
|---|---|---|
| 감독·제작·각본·작곡·저자 | 그 역할로 알려진 사람 | 다른 직업으로만 알려진 사람, 인물이 아닌 것 |
| 아버지·어머니 | 같은 성이거나 알려진 친족 | 무관한 사람 |
| 장르 | 같은 영역 | 다른 영역 |
| 스포츠·종교·직업 | 같은 큰 범주 | 다른 범주 |
| 출생지·수도·국가 | 같은 나라(큰 나라는 같은 주·지방) | 다른 나라 |

"애매하니까 1", "애매하니까 2"로 정하지 않는다. 판정할 수 없으면 불확실로 두고 에이전트에게 넘긴다.

### 4.1 Wikidata 규칙 (`code/rules.py`)

모델의 답 문자열로 Wikidata에서 후보 엔티티를 찾고(이름 변형·별칭 포함, 위키 문서 수(sitelinks)가 많은 순 8개), 후보의 직업·국가·가족·장르 영역을 가져와 판정한다. 정답 엔티티는 PopQA 원본의 Wikidata ID(`o_uri`)를 쓴다.

- **감독·제작·각본·작곡·저자:** 답의 후보 중 사람이 있고 그 직업에 역할이 있으면 유형 1. 직업 이름은 단어 단위로 맞춘다(저자는 writer, author, novelist, poet, playwright, essayist, biographer. `songwriter`나 `screenwriter`의 글자만 겹치는 것은 세지 않는다). 역할을 가진 후보가 가장 유명한 사람이 아니거나, 설명 문구가 창작 직업이 아니면 불확실이다. 이름이 한 단어뿐인 답은 항상 불확실이다. **제작자는 영역을 구분한다.** 질문이 묻는 작품이 영화·TV인지 음반인지(Wikidata의 작품 종류)와 답인 제작자의 영역(영화·TV 제작자인지 음반 프로듀서인지)이 같으면 유형 1, 다르면 유형 2다. 한쪽 영역을 알 수 없으면 불확실이다.
- **장르:** 정답과 답의 영역(음악, 영상, 문학, 게임, 무대)이 겹치면 유형 1. 영역은 Wikidata의 종류(`genre`, `style`) 이름과 설명 문구에서 읽는다. 후보가 여러 뜻을 가지고 덜 유명한 뜻으로만 겹치면 불확실이다.
- **출생지·수도·국가:** 정답과 답의 소속 국가가 같으면 유형 1. 미국·러시아·중국·인도·캐나다·브라질·호주는 행정구역 사슬(최대 5단계)에서 같은 주·지방이 있어야 한다. 관계가 `country`나 `capital of`이면 인접국도 유형 1이다. 옛 국가(동독, 소련 등)나, 이름이 같은 지역이 비슷한 비중으로 유명하면 불확실이다.
- **아버지·어머니:** 같은 성이거나(칭호 `of`, `the`가 있는 이름은 제외), 두 단계 이내 친족이거나, 같은 귀족 가문이면 유형 1. 정답이 사람이 아닌 동물이면 유형 2. 같은 이름의 사람이 여럿이면 불확실이다.

### 4.2 직접 정한 그룹 (`code/groups.py`)

스포츠·종교·직업은 값의 종류가 적어서(고유 쌍 약 600개) 그룹을 직접 정했다. 같은 그룹이면 유형 1이다. **이 그룹은 판단이 들어간 정의이므로 바꿀 수 있다.** 항목은 긴 구절부터 맞추고 맞춘 부분은 지운다.

**스포츠**

| 그룹 | 포함 항목 |
|---|---|
| `aquatic` | diving, dragon boat racing, paralympic swimming, rowing, sailing, swimming, triathlon, water polo |
| `athletics` | artistic gymnastics, athletics, climbing, gripping, gymnastics, rhythmic gymnastics, rock climbing, track and field, triathlon, weightlifting |
| `ball_games` | american football, association football, australian rules football, baseball, basketball, beach volleyball, cricket, field hockey, football, footballer, gaelic football, golf, handball, lacrosse, polo, rugby, rugby league, rugby union, sepak takraw, snooker, soccer, softball, table tennis, tennis, volleyball, water polo |
| `combat` | boxing, brazilian jiu-jitsu, fencing, judo, karate, mixed martial artist, mixed martial arts, professional wrestling, sumo, taekwondo, wrestling |
| `cycling` | bmx, cycling, road bicycle racing, track cycling |
| `dance` | dancing |
| `disc` | ultimate frisbee |
| `equestrian` | equestrian, horse riding, polo |
| `kabaddi` | kabaddi |
| `mind` | chess |
| `motor` | auto racing, motorcycle speedway |
| `paralympic` | paralympic |
| `precision` | archery, paralympic archery |
| `puck_stick` | hockey, ice hockey, roller hockey |
| `racket` | badminton |
| `video_game` | pong, super mario bros., super smash bros., tetris, the legend of zelda: ocarina of time, world of warcraft |
| `winter` | alpine skiing, bobsleigh, cross-country skiing, curling, figure skating, ice hockey, nordic combined, ski jumping, ski mountaineering, skiing, snowboarding, speed skating |

**종교**

| 그룹 | 포함 항목 |
|---|---|
| `christian` | american baptist churches usa, anglican, anglican communion, anglicanism, baptist, baptists, catholic, catholic church, catholicism, celtic christianity, christadelphians, christian, christian methodist episcopal church, christianity, church of scotland, congregational church, eastern orthodox church, episcopal church, evangelical lutheran church in america, evangelicalism, free methodist church, greek orthodox church of antioch, international pentecostal holiness church, lutheran, lutheran church--missouri synod, lutheran church–missouri synod, lutheranism, mennonite brethren church, methodism, methodist, mormon, mormonism, non-denominational, orthodox, orthodox christianity, presbyterian, presbyterian church (usa), protestant, protestantism, prussian union of churches, quaker, roman catholic, roman catholicism, romanian orthodox church, serbian orthodox church, seventh-day adventist, southern baptist convention, unitarian, unitarian universalism, unitarian universalist, unitarianism, united methodist church |
| `deism` | deism |
| `east_asian` | confucianism, shinto, taoism |
| `folk` | aymara, traditional african religion |
| `freemasonry` | freemasonry |
| `indian_religions` | buddhism, hinduism, jainism, sikhism, zen buddhism |
| `irreligion` | agnosticism, atheism, secular humanism |
| `islam` | islam, muslim, sunni islam |
| `judaism` | jewish, judaism |
| `occult` | thelema |
| `paganism` | anglo-saxon paganism |
| `rastafari` | rastafarianism |
| `zoroastrianism` | zoroastrianism |

**직업**

| 그룹 | 포함 항목 |
|---|---|
| `business` | art collector, art dealer, boston merchant, businessman, businessperson, chief executive officer, entrepreneur, executive director, executive vice president, financier, industrialist, president and ceo |
| `culinary` | chef |
| `medicine` | doctor, physician |
| `military_security` | british army, british army officer, chief inspector, french resistance, general, military officer, soldier, test pilot, united states army |
| `music` | composer, disc jockey, musician, singer, songwriter |
| `performing` | actor, actress, comedian, dancer, model, presenter, seiyū, singer |
| `politics_law` | attorney, chief justice, chief minister, diplomat, director of the office of international affairs, director of the office of management and budget, duke of suffolk, head of the department, king of poland, lawyer, lord mayor, lord of the admiralty, lord of the manor, magistrate, mayor, minister of foreign affairs, political science, politician, president of poland, president of the french republic, president of the united states, prime minister, prince |
| `religion` | bishop, chief rabbi, chief rabbi of israel, monk, orthodox priest, priest, rabbi, vestal virgin |
| `science_academia` | art historian, assistant professor, astrologer, astronomer, biochemist, bioethics, biologist, economist, geologist, historian, mathematician, meteorologist, philosopher, physicist, professor, statistics, teacher, vice president of the chinese academy of sciences |
| `screen_stage` | actor, actress, comedian, dancer, director, director of photography, documentary filmmaker, executive producer, film, film director, film industry, japanese animation, playwright, presenter, screenwriter, seiyū |
| `sports` | athlete, chess grandmaster, cricketer, footballer, golfer, hockey player, horse trainer, jockey, poker player, professional cyclist, wrestler |
| `visual_arts` | architect, art collector, art dealer, art historian, artist, fashion, fashion designer, model, painter, photographer |
| `writing_media` | author, critic, documentary filmmaker, editor, journalist, novelist, playwright, poet, printer, publisher, screenwriter, writer |

색(`color`)은 쌍이 9개뿐이라 에이전트가 판정했다.

### 4.3 확신도와 에이전트

모든 규칙 판정에는 확신(hi)과 불확실(lo)이 붙는다. 불확실이면 에이전트가 질문, 정답, Wikidata 설명(힌트)을 보고 판정한다. 에이전트가 판정한 쌍은 `label_source`가 `agent+...`이다. 에이전트가 확신하지 못한 쌍은 규칙의 기울기를 따르거나 기존 라벨을 유지하고 `final_conf`를 `false`로 둔다.

## 5. `label_source` 값

| 값 | 행 수 | 뜻 |
|---|---|---|
| `original-correct` | 11,073 | 원본 정답 |
| `lenient-correct-rule` / `lenient-correct-grounded` | 1,813 / 4 | 정답에 가깝다고 보아 정답으로 제안 |
| `string-rule` | 1,773 | 문자열 규칙(빈 출력, 반복, 누출, 기권, 복사) |
| `wikidata-rule` | 31,964 | Wikidata 규칙으로 1/2 판정 |
| `group-rule` | 1,959 | 직접 정한 그룹으로 1/2 판정 |
| `agent+wikidata-hint`, `agent+hint` | 5,835, 26 | 규칙이 확신하지 못해 에이전트가 판정(힌트를 봄) |
| `agent-pair-review` | 695 | 이전 라운드에서 에이전트가 쌍 단위로 판정한 3~6 유형 등 |
| `none` | 83 | 분류하지 않음 |

## 6. 에이전트 프롬프트 원문

에이전트에게는 아래 규칙 파일(`agent_prompt_en.md`)을 읽게 했다. 독립 판정에서는 이전 라벨을 보여 주지 않았고, 근거가 약한 쌍에서는 `hint` 열(Wikidata 후보와 증거)을 붙였다. 입력 한 줄은 `id<TAB>관계<TAB>gold=<정답><TAB>ans=<모델 답><TAB>q=<질문 예시><TAB>hint=...`이고, 출력은 한 줄에 `id<TAB>코드`다.

````markdown
# PopQA wrong-answer labeling (agent instructions, English)

Do the work yourself in this turn. Do not spawn sub-agents. Judge EVERY line by reading it. Do not use bulk rules (name lists, character counts).

## Input and output
Each input line is a pair that a base LLM answered wrongly on PopQA:
`id<TAB>relation<TAB>gold=<correct answer><TAB>ans=<model answer><TAB>q=<question example>`
Write one line per id: `id<TAB>CODE`. Use exactly one TAB. Every id exactly once, no other text. Append `?` to CODE if unsure.

## Codes (decision order: C -> 6 -> N -> 4 -> 5 -> 3 -> 1/2)
- `C`  Acceptable as correct. Alias, spelling, case, honorific, suffix; a broader category that the gold implies (gold "punk rock" -> "rock"); for genre/religion/sport/occupation only, a sub-type exactly one level below the gold in the same family. Siblings (rock vs pop, different religions, different occupations) and different entities are NOT acceptable. For people and places, a narrower answer is NOT acceptable. If it is borderline, answer `C`.
- `6`  Collapsed output: empty, a phrase repeated 3+ times, cut mid-word, prompt template leakage, gibberish. Junk after a correct answer is also 6.
- `N`  Abstention: "I don't know", "not specified", "no information".
- `4`  The gold (or an equivalent) is present AND a separate false claim is added. A plain wrong answer is never 4.
- `5`  A non-existent name built by splicing question tokens (subject, title) with generic words. An unfamiliar real-looking name is NOT 5.
- `3`  The answer is the subject, title or phrase copied from the question.
- `1`  A real entity that shares with the gold the ROLE or DOMAIN the question asks for.
- `2`  A real entity with no such shared role or domain.

## 1 vs 2: shared role or domain
"Both are people" is not a commonality. The commonality is what the question asks for.
| relation | 1 (shared) | 2 (not shared) |
|---|---|---|
| director, author, screenwriter, producer, composer | a person known for that role (any work) | known only for other work (actor, singer, politician), or not a person |
| father, mother | same family name, or a known relative / contemporary of the family | an unrelated person |
| genre | same domain (music with music, film/TV with film/TV, literature with literature, video games with video games) | a different domain (music vs film; a video-game genre such as role-playing game, platform game or first-person shooter vs a film, music or literature genre) |
| sport, religion, occupation | same broad group (ball games; same religious tradition; same field) | a different group |
| place of birth, capital | same country (USA, Russia, China, India, Canada, Brazil, Australia: same state or province) | a different country |
| country, capital of | neighboring or same sub-region | a different region |
If you cannot tell whether a name holds the role, label `1?`. Never decide by "ambiguous, so 1" or "ambiguous, so 2".

## Independent pass vs adjudication pass
- Independent pass: you see no earlier label. Label from the definitions only.
- Adjudication pass: each line also shows label A (current) and label B (independent reviewer). Output `id<TAB>FINAL<TAB>reason in at most 12 words (only when you change A)`. If you are not sure, keep A and add `?`.

## Worked examples for 1 vs 2 (gold -> model answer)
Use these to calibrate the boundary. The role or domain the question asks for decides, not the kind of entity.
| relation | label 1 (shares the role/domain) | label 2 (does not) |
|---|---|---|
| director | Liz Friedlander -> Michael Bay (both film directors) | Roy Del Ruth -> John F. Kennedy (a politician) |
| producer | Rakesh Roshan -> Rajkumar Santoshi (both Indian film makers) | Kunle Afolayan -> The Beatles (a band, not a film producer) |
| screenwriter | Paul Feig -> John Logan (both known as screenwriters) | Hirokazu Koreeda -> John Lennon (musician) |
| author | Roald Dahl -> Stanislaw Lem (both fiction authors) | William Goldman -> David Bowie (musician) |
| composer | Sergei Prokofiev -> Gustav Mahler (both composers) | Henry Jackman -> Seth Rogen (actor) |
| genre | romantic comedy -> Drama (film/TV); trip hop -> psychedelic (music) | telenovela -> Latin pop (TV vs music); real-time tactics -> science fiction (video game vs film) |
| father | Giovanni delle Bande Nere -> Lorenzo de' Medici (same family) | Charles Rothschild -> Duke Ellington (unrelated person) |
| mother | Natalia Ginzburg -> Anna Maria Ginzburg (same family name) | unrelated woman of another family |
| place of birth | Lankaran -> Baku (both Azerbaijan); Belo Horizonte -> Rio de Janeiro (both Brazil); Saint Petersburg -> Moscow (both Russia) | a city in a different country |
| country | Serbia -> Bosnia and Herzegovina (neighbors) | India -> Norway |
| sport | diving -> swimming (aquatic) | ice hockey -> football |
| occupation | actor -> film director (film field) | chef -> actor |
| religion | Anglicanism -> Presbyterian (Christian tradition) | Catholic Church -> Buddhism |
A different city in the same country is still 1 for place of birth. A person known only for an unrelated career is 2 even if famous.

## Grounded pass (hint column)
Each line may end with `hint=<label>(<hi|lo>) <evidence>`. The hint is automated evidence from Wikidata (candidate entities, their occupations, countries, genre domains). Use it as evidence, not as the answer: decide from the definitions. When the answer is a full sentence, first find the entity it names (the answer to the question), then judge that entity. A candidate that is clearly not the person or place the answer means (a same-name entity) does not count.

````

## 7. 사실별 분포

사실의 5개 질문 중 맞힌 개수별 사실 수: 0/5 7,106개, 1/5 723개, 2/5 361개, 3/5 303개, 4/5 404개, 5/5 1,772개 (맞힌 개수를 셀 수 없는 사실 376개 제외).

| fact | 오답 행 | 유형 1 | 유형 2 | 유형 3 | 유형 5 | 유형 6 | 기권 |
|---|---|---|---|---|---|---|---|
| 0/5 | 35,509 | 24,873 | 8,774 | 896 | 74 | 707 | 184 |
| 1/5 | 2,867 | 1,390 | 1,088 | 211 | 18 | 121 | 39 |
| 2/5 | 1,044 | 569 | 376 | 49 | 4 | 39 | 7 |
| 3/5 | 595 | 338 | 189 | 36 | 1 | 28 | 3 |
| 4/5 | 399 | 223 | 134 | 23 | 0 | 14 | 4 |

`analysis_use`가 `true`인 행은 48,570개다. SAE 분석에서 이 행들로 안다(5/5), 아는데 틀림(1~4/5의 유형 1·2), 모른다(0/5)를 나눴다(안다 1,726 fact, 유형 1 883, 유형 2 627, 모른다 7,087).

## 8. 품질 검증

| 점검 | 결과 |
|---|---|
| 에이전트만으로 세 라운드 반복 | 독립 판정과 불일치율 11.9%, 22.3%, 33.7%. 수렴 기준(3% 미만)에 닿지 못해 Wikidata 방식으로 전환 |
| 시험 300쌍(관계 12개 × 25) | 규칙이 확신한 판정의 95%가 이전 라벨과 같음 |
| 처음 보는 560쌍을 힌트 없이 새 에이전트가 판정(수정 전) | 규칙 판정 88.9%, 에이전트 판정 87.0% 일치. 갈린 40쌍을 직접 판정하니 규칙의 명백한 오류 3.6%, 정의에 따라 갈림 5.3% |
| 오류 수정(저자의 `songwriter`, 한 단어 답, 같은 이름의 지역, 음악인 제작자) | 수정 후 다시 돌림 |
| 처음 보는 720개를 힌트 없이 새 에이전트가 판정(수정 후) | 아래 표 |
| 제작자 관계의 기준 통일 | 위 점검에서 같은 상황의 라벨이 쌍마다 다르게 나온 것이 확인됨. 질문이 묻는 작품(영화·TV 또는 음반)과 답의 영역이 같으면 1, 다르면 2로 통일하고 다시 돌림(423행 변경, 모두 제작자 관계). 통일 후의 정확도는 다시 재지 않았다 |

수정 후 점검(720개, 갈린 116개는 근거를 보며 직접 판정):

| 대상 | 표본 | 블라인드와 일치 | 직접 판정 후 맞음 | 틀림 | 정의에 따라 갈림 |
|---|---|---|---|---|---|
| 1/2 판정 (Wikidata 규칙) | 300쌍 | 91.0% | 92.7% | 1.0% | 6.3% |
| 1/2 판정 (에이전트) | 200쌍 | 77.5% | 79.5% | 3.5% | 17.0% |
| 정답 제안(`correct_lenient`) | 60행 | 78.3% | 83.3% | 5.0% | 11.7% |
| 3 복사 | 40행 | 85.0% | 97.5% | 0% | 2.5% |
| 5 재조합 | 40행 | 67.5% | 80.0% | 2.5% | 17.5% |
| 6 붕괴 | 40행 | 80.0% | 80.0% | 5.0% | 15.0% |
| 기권 | 40행 | 90.0% | 90.0% | 0% | 10.0% |

위 검증은 에이전트와 같은 모델 계열이 했으므로 공통의 편향은 걸러내지 못한다. 에이전트가 판정한 쌍은 같은 상황에서도 판정이 흔들릴 수 있다(제작자 관계에서 확인되어 통일했다).

## 9. 알려진 한계

- 유형 1과 2의 경계는 정의에 따라 갈리는 쌍이 약 5% 있다(예: Lennon을 저자로 볼지, 아이스하키를 구기로 볼지, 작곡가와 배우를 같은 분야로 볼지).
- 스포츠·종교·직업 그룹은 직접 정한 것이다. 인도 종교와 기독교 안의 교파 묶음 등은 바꿀 수 있다.
- 이름이 같은 사람이나 지역은 가장 널리 알려진 후보로 정하고, 불확실하면 에이전트가 판단했다. 오류가 남을 수 있다.
- 3촌 이상의 먼 친족, 옛 국가의 소속은 규칙이 처리하지 못한다.
- 정답으로 제안한 행(`correct_lenient`)에는 정답이 아닌 것이 약 5% 섞여 있을 수 있다(예: 아론의 어머니를 "Mary"로 답한 행).
- 유형 4(2행)와 유형 5(98행)는 표본이 너무 적다. 유형 3, 5, 6, 4의 라벨은 1과 2만큼 Wikidata로 검증하지 않았고, 대부분 문자열 규칙과 이전 에이전트 판정이다.
- `correct`와 `correct_lenient`의 정답 판정은 엔티티 일치로 다시 하지 않았다. 정답으로 처리한 행의 약 17%가 점검에서 정답이 아니라고 나왔고(`audit_flag`), 그 사실은 5/5 그룹에서 뺐다.
- 원본 `verdict`는 바꾸지 않았다. `uncertain` 83행은 분류하지 않았다.

## 10. 이전 버전과 재현

| 폴더 | 내용 |
|---|---|
| `1~5. reviewed` | 1,000개 표본의 반복 검토 |
| `6. excl 0of5` | 1,000개 표본에서 0/5를 뺀 것 |
| 이전의 `7~12` | **삭제함(번호는 정리하면서 바뀌어 지금 7번이 최종 폴더다).** 11,045개 전체로 확장하며 기준을 정하던 초안(7~9), 에이전트 독립 판정 3라운드(10~11, 수렴하지 못함), Wikidata 기반 판정의 원본 형태(12). 필요한 내용은 모두 최종 폴더에 들어 있다 |
| **`7. final`** | **이 폴더. 최종 파일과 정책** |

재현 순서(모든 코드는 `code/`, 입력은 이 폴더의 최종 파일과 `grounded/`의 `agent_v3.json`, `merged_v3.json`):

1. `full.py [출력경로]`: 최종 파일에서 1/2로 분류된 (관계, 정답, 답) 쌍을 뽑아 Wikidata 규칙(`wd.py`, `rules.py`)으로 판정한다.
2. `merge_groups.py`: 위 결과에 직접 정한 그룹(`groups.py`) 판정을 합쳐 `merged_v3.json`을 만든다.
3. 확신하지 못한 쌍은 에이전트에게 `agent_prompt_en.md`로 판정시킨다. 지금까지의 판정은 `agent_v3.json`에 모아 두었다.
4. `apply_final.py [출력경로]`: 규칙과 에이전트 판정을 최종 파일의 1/2 행에 다시 적용한다. 최종 파일은 덮어쓰지 않는다.

검증: 이 순서를 최종 파일에서 다시 돌렸을 때 합치기는 기존 결과와 같았고, 라벨 재적용은 55,225행 중 바뀌는 행이 없었다. 규칙 전체 재실행은 22,052쌍에서 라벨과 확신도가 같았다(이후 1/2에서 빠진 31쌍은 입력에서 제외됨). 문자열 규칙 단계(`string_rules_prep.py`)는 처음 채점 결과(`1. reviewed 1`)에서 시작하는 초기 코드라 이 순서에는 들어 있지 않다.

SAE feature 추출과 분석 스크립트는 `sae_analysis/`에 있고 모두 이 폴더의 최종 파일을 읽는다(`build_meta_from_final.py`로 메타를 만들고, `extract_features_refined.py`가 같은 파일로 feature를 추출한다). 이 경로로 만든 메타로 분석을 다시 돌렸을 때 기존 로그와 줄 단위로 같았다. Wikidata 질의는 공개 SPARQL 엔드포인트의 읽기 전용 조회이고, 요청 간격을 1초 이상 둔다.
