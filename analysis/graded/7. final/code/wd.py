"""Cached Wikidata lookups through batched SPARQL (the REST API throttles hard).
candidates(name) -> candidate dicts; entities(qids) -> {qid: {label, desc, sitelinks, claims{P: [qid]}}}; labels(qids)."""
import json, os, re, time, urllib.error, urllib.parse, urllib.request

EP = "https://query.wikidata.org/sparql"
UA = "hallucination-label-research/0.1 (academic research; read-only lookups)"
CACHE_PATH = os.environ.get("WD_CACHE", "grounded/wd_cache.json")
PROPS = ["P31", "P279", "P106", "P17", "P27", "P495", "P131", "P136", "P734", "P22", "P25", "P40", "P26", "P3373", "P1038", "P53", "P47"]
_cache = json.load(open(CACHE_PATH)) if os.path.exists(CACHE_PATH) else {}
for k in ("names", "ent", "lab"):
    _cache.setdefault(k, {})
_last = [0.0]
GAP = float(os.environ.get("WD_GAP", "1.0"))


def save():
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    json.dump(_cache, open(CACHE_PATH + ".tmp", "w"), ensure_ascii=False)
    os.replace(CACHE_PATH + ".tmp", CACHE_PATH)


def sparql(q):
    for attempt in range(10):
        w = GAP - (time.time() - _last[0])
        if w > 0:
            time.sleep(w)
        _last[0] = time.time()
        try:
            req = urllib.request.Request(EP + "?format=json&query=" + urllib.parse.quote(q), headers={"User-Agent": UA})
            return json.load(urllib.request.urlopen(req, timeout=120))["results"]["bindings"]
        except urllib.error.HTTPError as e:
            ra = e.headers.get("Retry-After")
            time.sleep(min(int(ra) + 2, 120) if ra and ra.isdigit() else 8 * (attempt + 1))
        except Exception:
            time.sleep(8 * (attempt + 1))
    raise RuntimeError("sparql failed")


def _esc(s):
    return s.replace("\\", "\\\\").replace('"', '\\"')


def variants(n):
    return list(dict.fromkeys([n, n.lower(), n[:1].upper() + n[1:].lower()]))


def prefetch_names(names, batch=50, top=8):
    names = [n for n in dict.fromkeys(names) if n not in _cache["names"]]
    if not names:
        return
    for b in range(0, len(names), batch):
        chunk = names[b:b + batch]
        vals = " ".join('"%s"@en' % _esc(v) for n in chunk for v in variants(n))
        rows = sparql(f"SELECT ?name ?item ?sl WHERE {{ VALUES ?name {{ {vals} }} ?item rdfs:label|skos:altLabel ?name . ?item wikibase:sitelinks ?sl . FILTER(?sl >= 1) }}")
        by = {}
        for r in rows:
            by.setdefault(r["name"]["value"], set()).add((int(r["sl"]["value"]), r["item"]["value"].rsplit("/", 1)[1]))
        for n in chunk:
            lst = set().union(*[by.get(v, set()) for v in variants(n)])
            _cache["names"][n] = [q for _, q in sorted(lst, reverse=True)[:top]]
        if (b // batch) % 10 == 0:
            save()
    save()


def prefetch_ents(qids, batch=120):
    qids = [q for q in dict.fromkeys(qids) if q and q not in _cache["ent"]]
    if not qids:
        return
    pv = " ".join("wdt:" + p for p in PROPS)
    for b in range(0, len(qids), batch):
        chunk = qids[b:b + batch]
        vals = " ".join("wd:" + q for q in chunk)
        rows = sparql(f"""SELECT ?item ?il ?d ?sl ?p ?v ?vl WHERE {{ VALUES ?item {{ {vals} }}
 OPTIONAL {{ ?item wikibase:sitelinks ?sl }}
 OPTIONAL {{ ?item rdfs:label ?il FILTER(LANG(?il)="en") }}
 OPTIONAL {{ ?item schema:description ?d FILTER(LANG(?d)="en") }}
 OPTIONAL {{ VALUES ?p {{ {pv} }} ?item ?p ?v . OPTIONAL {{ ?v rdfs:label ?vl FILTER(LANG(?vl)="en") }} }} }}""")
        ents = {q: {"label": "", "desc": "", "sitelinks": 0, "claims": {p: [] for p in PROPS}} for q in chunk}
        for r in rows:
            q = r["item"]["value"].rsplit("/", 1)[1]
            e = ents[q]
            e["label"] = r.get("il", {}).get("value", e["label"]); e["desc"] = r.get("d", {}).get("value", e["desc"])
            e["sitelinks"] = int(r["sl"]["value"]) if "sl" in r else e["sitelinks"]
            if "p" in r and "v" in r and r["v"]["type"] == "uri":
                p = r["p"]["value"].rsplit("/", 1)[1]; v = r["v"]["value"].rsplit("/", 1)[1]
                if v not in e["claims"][p]:
                    e["claims"][p].append(v)
                if v not in _cache["lab"] or not _cache["lab"][v][0]:
                    _cache["lab"][v] = [r.get("vl", {}).get("value", ""), ""]
        for q, e in ents.items():
            _cache["ent"][q] = e
            _cache["lab"].setdefault(q, [e["label"], e["desc"]])
        if (b // batch) % 10 == 0:
            save()
    save()


def entities(ids):
    prefetch_ents(ids)
    return {i: _cache["ent"].get(i) for i in ids}


def labels(ids):
    return {i: _cache["lab"].get(i, ["", ""])[0] for i in ids}


def candidates(name, limit=8):
    prefetch_names([name])
    ids = _cache["names"].get(name, [])[:limit]
    ents = entities(ids)
    out = []
    for i in ids:
        e = ents[i]
        if not e:
            continue
        c = e["claims"]
        L = lambda p: [_cache["lab"].get(q, ["", ""])[0].lower() for q in c[p]]
        out.append({"id": i, "label": e["label"], "desc": e["desc"], "sl": e["sitelinks"], "human": "Q5" in c["P31"],
                    "inst": L("P31"), "occ": L("P106"), "country": c["P17"] + c["P27"] + c["P495"], "genre": L("P136"), "claims": c})
    return out
