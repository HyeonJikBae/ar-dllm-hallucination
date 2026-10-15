"""String-only grader shared by the Dream sampler comparison (same rules as the stage-1 string rules of the Qwen pipeline).

string_rule(...) -> code or None:
  6-empty / 6-leak / 6-loop  collapse (empty output, prompt-template leak, repeated phrase)
  N-abstain                  abstention
  C-alias / C-short          answer equals a gold alias (after normalisation) or the short form of a person name
  3-copy                     answer copied from the question
  None                       anything else (wrong entity, or a longer answer containing the gold)
contains_gold(...)           a normalised gold/alias occurs as a whole-word span in a normalised answer of <= 12 words
"""
import collections
import re
import unicodedata

TYPE = {**{r: "A" for r in ["director", "producer", "screenwriter", "composer", "author"]}, **{r: "B" for r in ["father", "mother"]},
        **{r: "C" for r in ["genre", "religion", "sport", "occupation", "color"]}, **{r: "D" for r in ["place of birth", "country", "capital", "capital of"]}}
TITLES = {"sir", "dr", "mr", "mrs", "ms", "prof", "professor", "lord", "lady", "dame", "the", "a", "an", "saint", "st", "jr", "sr", "ii", "iii", "iv"}
ABST = re.compile(r"\b(not specified|no information|not provided|not mentioned|don t know|do not know|unknown|cannot|can t|unclear|no answer|not available|not given|not known|not sure|there is no|n a)\b")


def norm(s):
    s = unicodedata.normalize("NFKD", (s or "").lower()); s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s)).strip()


def loopy(t):
    w = t.split()
    if len(w) >= 4:
        c = collections.Counter(w)
        if c.most_common(1)[0][1] >= 3 and len(w) >= 5:
            return True
        bg = collections.Counter(zip(w, w[1:]))
        if bg.most_common(1)[0][1] >= 3:
            return True
    return False


def string_rule(answer, question, gold, aliases, rel):
    a = answer or ""; na = norm(a); ng = norm(gold)
    nal = {norm(x) for x in aliases} | {ng}; nq = norm(question)
    if not na:
        return "6-empty"
    if "question" in na.split() and re.search(r"\b(question|answer)\b ?:?", a.lower()) and re.search(r"(?i)question:|answer:|direct answer", a):
        return "6-leak"
    if loopy(na):
        return "6-loop"
    if len(na.split()) <= 14 and ABST.search(na):
        return "N-abstain"
    if na in nal:
        return "C-alias"
    wa = [w for w in na.split() if w not in TITLES]; wg = [w for w in ng.split() if w not in TITLES]
    if wa and wg and set(wa) == set(wg):
        return "C-alias"
    if wa and wg and TYPE.get(rel) in ("A", "B") and len(wa) >= 2 and set(wa) < set(wg):
        return "C-short"
    if wa and wg and len(wg) >= 2 and wa == [wg[-1]] and TYPE.get(rel) in ("A", "B"):
        return "C-short"
    if wa and wg and set(wg) < set(wa) and len(wa) - len(wg) <= 2 and len(wa) <= 6:
        return "C-alias"
    if wa and len(" ".join(wa)) >= 3 and re.search(r"(?<![a-z0-9])" + re.escape(" ".join(wa)) + r"(?![a-z0-9])", nq):
        return "3-copy"
    return None


def contains_gold(answer, gold, aliases):
    na = norm(answer)
    if not na or len(na.split()) > 12:
        return False
    return any(g and re.search(r"(?<![a-z0-9])" + re.escape(g) + r"(?![a-z0-9])", na) for g in {norm(gold)} | {norm(x) for x in aliases})
