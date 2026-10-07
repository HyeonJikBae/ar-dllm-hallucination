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
