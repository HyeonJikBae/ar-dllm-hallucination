"""E56 공용 (TriviaQA 다섯 문구).  repo 루트에서 실행.  common.py 는 고치지 않는다 (채점·분류 규칙은 data/2_grading, data/3_classification 에서 가져옴).
채점 (PopQA run_grading.is_correct 와 같은 규칙을 TriviaQA 별칭에 적용):
  - 별칭 목록 = TriviaQA answer.value + answer.aliases, 그중 원문 질문에 (정규화 후 단어 단위로) 이미 나오는 별칭은 뺌 (질문 노출 별칭; 예 'Kniphofia').
    여기에 각 별칭의 앞 관사 (the/a/an) 를 지운 형태를 더함 (TriviaQA 공식 정규화가 관사를 지우므로).  괄호 지운 형태는 넣지 않음
    (TriviaQA 별칭은 위키 넘겨주기라 'Magda (James Bond)' 같은 다른 개체가 섞여 있어, 괄호를 지우면 엉뚱한 이름이 정답이 됨).
  - 정답 = is_correct(답, 별칭) 또는 is_correct(앞 관사를 지운 답, 별칭).  (is_correct: junk 이면 오답, 정규화 후 같거나 difflib 유사도 >= 90, 로마 숫자 일치)
  - 보조 (민감도): 포함 규칙 = 주 규칙 정답 이거나, junk 아님 + 별칭 (정규화 4 글자 이상) 이 답 안에 단어 단위로 들어 있고 그 별칭이 해당 문구 질문에는 없음.
분류 (오답만): classify_rules.classify(답, 문구 질문, relation="", has_candidate=None) -> entity / broken / abstain.
  (PopQA 의 8 번 규칙 'broken-fabricated' 는 Wikidata 후보 조회가 필요한데 TriviaQA 에는 관계·캐시가 없어 has_candidate=None -> entity 로 둠.)
금 첫 토큰: tok(" " + answer.value)[0] (E44/E50 규칙).  '어느 별칭이든': value 와, 괄호가 없고 value 와 내용어 (3 글자 이상) 를 하나 이상 공유하는 별칭에서
  'List of / History of ...' 같은 넘겨주기 문서명이 아니고 value 보다 2 단어 이상 길지 않은 것에서 앞 관사·호칭 (the/a/an/sir/dr/...) 을 지운 형태들의 tok(" " + 형태)[0] 집합 (넘겨주기 잡음 'Sports in Utah', 'Kamal Khan' 과 ' Sir' 같은 일반 토큰을 피하려고).
"""
import collections, json, re, sys
sys.path.insert(0, "data/2_grading"); sys.path.insert(0, "data/3_classification")
from run_grading import is_correct, norm, junk
from classify_rules import classify

D56 = "followup_mk/e56_data"
VERS = ["question", "v1", "v2", "v3", "v4"]
ART = re.compile(r"^(the|a|an)\s+", re.I)


def strip_art(s): return ART.sub("", (s or "").strip())


def word_in(needle, hay):
    """정규화된 needle 이 정규화된 hay 안에 단어 경계로 들어 있나."""
    n, h = norm(needle), norm(hay)
    return bool(n) and f" {n} " in f" {h} "


def alias_forms(value, aliases, question):
    """(채점 별칭 목록, 질문 노출로 뺀 별칭 목록).  value 가 질문에 노출되면 None 을 돌려줌 (그 질문은 쓰지 않음)."""
    if word_in(value, question): return None, None
    base, exposed = [], []
    for a in [value] + list(aliases):
        a = (a or "").strip()
        if not norm(a): continue
        if word_in(a, question): exposed.append(a); continue
        base.append(a)
    out = []
    for a in base:
        for f in (a, strip_art(a)):
            if norm(f) and f not in out: out.append(f)
    return out, exposed


def grade_correct(ans, value, aliases):
    r = {"model_answer": ans, "gold_value": value, "gold_aliases": aliases}
    if is_correct(r): return True
    a2 = strip_art(ans)
    return a2 != (ans or "").strip() and is_correct({**r, "model_answer": a2})


def grade_contain(ans, value, aliases, qtext):
    if grade_correct(ans, value, aliases): return True
    if not ans or junk(ans): return False
    return any(len(norm(a)) >= 4 and word_in(a, ans) and not word_in(a, qtext) for a in aliases)


def grade(ans, value, aliases, qtext):
    """-> correct / entity / broken / abstain (주 규칙)."""
    if grade_correct(ans, value, aliases): return "correct"
    return classify(ans, qtext, "", lambda n: None)[0]


TITLE = re.compile(r"^((the|a|an|sir|dame|lord|lady|dr|dr\.|mr|mr\.|mrs|mrs\.|ms|prof|professor|saint|st|st\.|king|queen|president|general)\s+)+", re.I)
NOISE = re.compile(r"^(list|lists|history|geography|politics|economy|culture|sports?|religion|education|climate|transport|transportation|demographics|music|government|outline|index)\s+(of|in)\b", re.I)
CW = lambda s: {w for w in norm(s).split() if len(w) >= 3 and w not in ("the", "and", "of")}


def first_tokens(tok, value, aliases):
    """(금 첫 토큰, 별칭 첫 토큰 집합).  Qwen2 계열 토크나이저 (Dream, Qwen2.5) 는 BOS 를 붙이지 않는다."""
    g = tok(" " + value.strip(), add_special_tokens=False)["input_ids"][0]
    s = {g}; vw = CW(value)
    for a in [value] + list(aliases):
        a = a.strip()
        if "(" in a or not (CW(a) & vw) or NOISE.match(a) or len(a.split()) > len(value.split()) + 1: continue
        f = TITLE.sub("", a).strip()
        if norm(f): s.add(tok(" " + f, add_special_tokens=False)["input_ids"][0])
    return g, sorted(s)


def load_items(path=f"{D56}/items.jsonl"):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def groups(items, idf, bs):
    g = collections.defaultdict(list)
    for it in items: g[len(idf(it))].append(it)
    for L, xs in sorted(g.items()):
        for b0 in range(0, len(xs), bs): yield L, xs[b0:b0 + bs]
