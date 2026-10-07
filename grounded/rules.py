"""Wikidata-grounded 1 (shares the role/domain) vs 2 (does not) for a wrong answer.
label_pair(rel, gold_qid, answer) -> (label, confidence, evidence); label '?' means send to an agent with the evidence."""
import re
import wd

BAD_DIR = ("art director", "casting", "artistic", "music director", "creative director", "managing", "executive director", "technical director")
AUTHOR_RE = re.compile(r"(?<![\w])(writer|author|novelist|poet|playwright|essayist|biographer)(?![\w])")
ROLE = {
    "director": lambda o: "director" in o and not any(x in o for x in BAD_DIR),
    "producer": lambda o: "producer" in o,
    "screenwriter": lambda o: any(x in o for x in ("screenwriter", "television writer", "scriptwriter", "screen writer")),
    "composer": lambda o: "composer" in o,
    "author": lambda o: bool(AUTHOR_RE.search(o)) and not any(x in o for x in ("television writer", "screen writer")),
}
WORK = re.compile(r"\((?:[^)]*\b(?:film|novel|album|song|series|book|play|band|game|tv|miniseries|show|magazine)\b[^)]*)\)", re.I)
DOMAINS = [("music", ("music",)), ("screen", ("film", "television", "tv ")), ("literary", ("literary", "literature", "fiction", "book")), ("game", ("video game", "game genre")), ("stage", ("theatre", "theater", "drama genre", "performing"))]
PLACE_REL = ("place of birth", "capital", "country", "capital of")
NEIGHBOR_REL = ("country", "capital of")


MUSIC_GOLD = ("record producer", "music producer", "musician", "singer", "songwriter", "composer", "bandleader", "disc jockey")
FILM_GOLD = ("film producer", "film director", "television producer", "screenwriter", "film actor", "executive producer")
CREATIVE = ("writer", "author", "novelist", "poet", "director", "producer", "screenwriter", "filmmaker", "composer", "musician", "playwright", "dramatist",
            "journalist", "actor", "actress", "singer", "songwriter", "artist", "showrunner", "animator", "editor", "critic", "publisher", "conductor", "lyricist")


def gold_is_music(gold_qid):
    """True/False when the gold entity clearly sits in music / screen, None when unknown."""
    g = wd.entities([gold_qid])[gold_qid] if gold_qid else None
    if not g:
        return None
    labs = [x.lower() for x in wd.labels(g["claims"]["P106"] + g["claims"]["P31"]).values()]
    m = any(any(k in l for k in MUSIC_GOLD) or "band" in l or "musical group" in l or "record label" in l for l in labs)
    f = any(any(k in l for k in FILM_GOLD) or "company" in l or "channel" in l for l in labs)
    if m and not f:
        return True
    if f and not m:
        return False
    return None


def role_label(rel, ans, gold_qid=None):
    if WORK.search(ans):
        return "2", "hi", "title with qualifier (work, not a person)"
    cs = wd.candidates(ans)
    if not cs:
        return "?", "lo", "no wikidata candidate"
    humans = [c for c in cs if c["human"]]
    holders = [h for h in humans if any(ROLE[rel](o) for o in h["occ"])]
    ev = lambda c: f"{c['id']} {c['label']} [{c['desc'][:50]}] occ={c['occ'][:4]} sl={c['sl']}"
    if holders:
        best = max(holders, key=lambda c: c["sl"]); top = max(humans, key=lambda c: c["sl"])
        if rel == "producer":
            ans_film = any(any(k in o for k in ("film producer", "television producer", "executive producer", "movie producer")) for o in best["occ"])
            ans_music = any(("record producer" in o or "music producer" in o) for o in best["occ"])
            gm = gold_is_music(gold_qid)
            d = best["desc"].lower()
            music_desc = any(k in d for k in ("musician", "singer", "songwriter", "record producer", "composer", "guitarist", "rapper", "drummer")) and not any(k in d for k in ("film", "television", "tv ", "movie"))
            if gm is not True and music_desc:
                return "1", "lo", "answer is described as a musician/record producer, gold is not music: " + ev(best)
            if gm is True and not ans_music:
                return "1", "lo", "gold is music, answer is a film producer: " + ev(best)
            if gm is False and not ans_film:
                return "1", "lo", "gold is film/TV, answer is only a record/music producer: " + ev(best)
            if gm is None and not ans_film:
                return "1", "lo", "only a record/music producer: " + ev(best)
        if not any(k in best["desc"].lower() for k in CREATIVE):
            return "1", "lo", "holder's description is not a creative profession (possible wrong entity): " + ev(best)
        if best is top or best["sl"] >= 0.5 * top["sl"]:
            return "1", "hi", ev(best)
        return "1", "lo", "role holder is not the most prominent: " + ev(best) + " | top: " + ev(top)
    single = len(ans.split()) == 1
    if humans:
        top = max(humans, key=lambda c: c["sl"])
        conf = "hi" if top["sl"] >= 5 and top["occ"] else "lo"
        if single:
            conf = "lo"
        return "2", conf, "no holder; top human " + ev(top) + (" (single-word answer)" if single else "")
    top = max(cs, key=lambda c: c["sl"])
    conf = "hi" if top["sl"] >= 5 else "lo"
    if single:
        conf = "lo"
    return "2", conf, f"non-person {top['id']} {top['label']} [{top['desc'][:50]}]" + (" (single-word answer)" if single else "")


def domains(inst):
    out = set()
    for i in inst:
        if "genre" not in i and "style" not in i:
            continue
        for d, kws in DOMAINS:
            if any(k in i for k in kws):
                out.add(d)
    return out


def domains_desc(desc):
    d = (desc or "").lower(); out = set()
    if "film" in d or "television" in d or " tv " in d:
        out.add("screen")
    if "music" in d:
        out.add("music")
    if "video game" in d:
        out.add("game")
    return out


def domains_loose(labels):
    out = set()
    for l in labels:
        for d, kws in DOMAINS:
            if any(k in l for k in kws):
                out.add(d)
    return out


def genre_label(gold_qid, ans):
    g = wd.entities([gold_qid])[gold_qid]
    if not g:
        return "?", "lo", "gold entity missing"
    wd.labels(g["claims"]["P31"] + g["claims"]["P279"])
    gd = domains([wd._cache["lab"][q][0].lower() for q in g["claims"]["P31"]]) | domains_desc(g["desc"])
    cs = [c for c in wd.candidates(ans) if any(("genre" in i or "style" in i) for i in c["inst"])]
    if not cs:
        return "?", "lo", f"no genre candidate; gold domains={sorted(gd)}"
    top = max(cs, key=lambda c: c["sl"])
    sup = lambda c: domains_desc(c["desc"])
    td = domains(top["inst"]) | sup(top); ad = set()
    for c in sorted(cs, key=lambda c: -c["sl"])[:5]:
        ad |= domains(c["inst"]) | sup(c)
    if not gd or not ad:
        return "?", "lo", f"domains unknown gold={sorted(gd)} ans={sorted(ad)} ({top['label']})"
    if gd & td:
        return "1", "hi", f"gold {sorted(gd)} vs ans {top['label']} {sorted(td)}"
    if gd & ad:
        return "1", "lo", f"match only via a less prominent sense: gold {sorted(gd)}, top {top['label']} {sorted(td)}, all {sorted(ad)}"
    return "2", "hi", f"gold {sorted(gd)} vs ans {top['label']} {sorted(ad)}"


def country_set(c):
    return set(c["country"]) | ({c["id"]} if any(("country" in i or "sovereign state" in i) for i in c["inst"]) else set())


BIG = {"Q30", "Q159", "Q148", "Q668", "Q16", "Q155", "Q408"}  # USA Russia China India Canada Brazil Australia: same state/province needed


def anc(q, depth=5):
    seen, frontier = {q}, [q]
    for _ in range(depth):
        nxt = []
        for x in frontier:
            e = wd.entities([x])[x]
            for y in (e["claims"]["P131"] if e else []):
                if y not in seen:
                    seen.add(y); nxt.append(y)
        frontier = nxt
    return seen


def _place_core(rel, gold_qid, ans, force_id=None):
    g = wd.entities([gold_qid])[gold_qid]
    if not g:
        return "?", "lo", "gold entity missing", None
    wd.labels(g["claims"]["P31"] + g["claims"]["P17"])
    gl = [wd._cache["lab"][q][0].lower() for q in g["claims"]["P31"]]
    gset = set(g["claims"]["P17"]) | ({gold_qid} if any(("country" in i or "sovereign state" in i) for i in gl) else set())
    cs = [c for c in wd.candidates(ans) if c["country"] or any(("country" in i or "sovereign state" in i) for i in c["inst"])]
    if not gset or not cs:
        return "?", "lo", f"country unknown gold={sorted(gset)} cands={len(cs)}", None
    names = lambda s: [wd.labels([q])[q] for q in s]
    top = max(cs, key=lambda c: c["sl"]); lowconf = False
    if force_id:
        top = next((c for c in cs if c["id"] == force_id), top)
    aset = country_set(top)
    if not (gset & aset) and not force_id:
        alt = [c for c in cs if country_set(c) & gset]
        if alt:
            top = max(alt, key=lambda c: c["sl"]); aset = country_set(top); lowconf = True
    if gset & aset:
        big = (gset & aset) & BIG
        if big and rel != "country" and gold_qid not in gset:
            ga, aa = anc(gold_qid) | gset, anc(top["id"]) | aset
            common = (ga & aa) - big - {gold_qid} if gold_qid != top["id"] else ga
            common = {c for c in common if c not in gset and c not in aset}
            if not common:
                return "2", ("lo" if lowconf else "hi"), f"same big country {names(big)} but different state/region ({top['label']})", top["id"]
            return "1", ("lo" if lowconf else "hi"), f"same state/region {names(common)[:2]} ({top['label']})", top["id"]
        return "1", ("lo" if lowconf else "hi"), f"same country {names(gset & aset)} ({top['label']})", top["id"]
    if rel in NEIGHBOR_REL:
        nb = set()
        for q in gset:
            e = wd.entities([q])[q]
            nb |= set(e["claims"]["P47"]) if e else set()
        if nb & aset:
            return "1", "hi", f"neighbor country {names(nb & aset)}", top["id"]
    hist = ("east germany", "west germany", "soviet union", "yugoslavia", "czechoslovakia", "austria", "british empire", "ottoman", "kingdom of", "empire")
    allnames = " ".join(n.lower() for n in names(gset) + names(aset))
    conf = "lo" if any(h in allnames for h in hist) else "hi"
    return "2", conf, f"gold country {names(gset)} vs ans {top['label']} {names(aset)}" + (" (historical state)" if conf == "lo" else ""), top["id"]

def place_label(rel, gold_qid, ans):
    lab, conf, ev, tid = _place_core(rel, gold_qid, ans)
    if tid is None or conf == "lo":
        return lab, conf, ev
    cs = [c for c in wd.candidates(ans) if c["country"] or any(("country" in i or "sovereign state" in i) for i in c["inst"])]
    tl = next((c["label"].lower() for c in cs if c["id"] == tid), "")
    top_sl0 = next((c["sl"] for c in cs if c["id"] == tid), 0)
    if sum(1 for c in cs if c["label"].lower() == tl and c["sl"] >= max(2, 0.15 * top_sl0)) >= 3:
        return lab, "lo", ev + " | generic place name shared by several places"
    top_sl = next((c["sl"] for c in cs if c["id"] == tid), 0)
    for c in cs:
        if c["id"] != tid and c["label"].lower() == tl and c["sl"] >= max(3, 0.3 * top_sl):
            l2, _, _, _ = _place_core(rel, gold_qid, ans, force_id=c["id"])
            if l2 != lab:
                return lab, "lo", ev + f" | same name resolves differently for {c['id']}"
    return lab, conf, ev


def parent_label(gold_qid, gold_name, ans):
    g = wd.entities([gold_qid])[gold_qid]
    if not g:
        return "?", "lo", "gold entity missing"
    cs = [c for c in wd.candidates(ans) if c["human"]]
    if "Q5" not in g["claims"]["P31"]:
        gl = " ".join(wd.labels(g["claims"]["P31"]).values()).lower()
        animal = any(k in gl for k in ("horse", "dog", "cat", "animal", "breed", "cattle", "mammal"))
        if cs and animal:
            return "2", "hi", f"gold is an animal ({gl[:40]}); answer is a person"
        return "?", "lo", f"gold is not a person ({gl[:40]})"
    def sur(n):
        t = re.sub(r"[^\w ]", "", n).split()
        return "" if (len(t) < 2 or "of" in [x.lower() for x in t] or "the" in [x.lower() for x in t]) else t[-1].lower()
    if sur(gold_name) and sur(gold_name) == sur(ans):
        return "1", "hi", "same family name"
    R = lambda e: set(sum((e["claims"][p] for p in ("P22", "P25", "P40", "P26", "P3373", "P1038")), []))
    rg = R(g)
    if not cs:
        return "?", "lo", "no human candidate"
    top = sorted(cs, key=lambda c: -c["sl"])
    for c in top[:4]:
        e = wd._cache["ent"][c["id"]]
        if c["id"] in rg or gold_qid in R(e) or (rg & R(e)):
            return "1", "hi", f"relative/in-law {c['id']} {c['label']}"
        if set(c["claims"]["P53"]) & set(g["claims"]["P53"]):
            return "1", "hi", f"same noble family {c['label']}"
    same = sum(1 for c in top if c["sl"] >= 3 and c["label"].lower() == top[0]["label"].lower())
    many = sum(1 for c in top if c["sl"] >= 10) > 1 or same > 1
    return "2", ("lo" if many or top[0]["sl"] < 5 else "hi"), f"no family link; top {top[0]['label']} [{top[0]['desc'][:40]}]" + (" (several same-name people)" if many else "")


def label_pair(rel, gold_qid, gold_name, ans):
    if rel in ROLE:
        return role_label(rel, ans, gold_qid)
    if rel == "genre":
        return genre_label(gold_qid, ans)
    if rel in PLACE_REL:
        return place_label(rel, gold_qid, ans)
    if rel in ("father", "mother"):
        return parent_label(gold_qid, gold_name, ans)
    return "?", "lo", "relation not implemented (agent with description)"
