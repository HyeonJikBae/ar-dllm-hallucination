"""Does the name a type-5 (recombination) answer gives exist on Wikidata?
Usage: python3 grounded/check_type5.py   -> prints a table and writes /tmp/type5_check.json"""
import collections, json, sys
sys.path.insert(0, "grounded")
import wd

rows = [json.loads(l) for l in open("analysis/graded/7. final/qwen2.5-7b_popqa_final.jsonl")]
r5 = [r for r in rows if r["final_label"] == "5"]
uniq = collections.OrderedDict()
for r in r5:
    uniq.setdefault((r["relation"], r["gold_value"], (r["model_answer"] or "").strip()), []).append(r)
# sentence-form answers: the name the answer actually gives
OVERRIDE = {
 'Sune Sik\'s father was Søren Sik.': 'Søren Sik', 'Vanlandi\'s father is known as "The Great Vanlandi."': 'The Great Vanlandi',
 "Corbin Bleu's father is James Bleu.": 'James Bleu', 'Ruby Stewart was sired by the stallion "Ruby\'s Secret" and her': "Ruby's Secret",
 'Dixie Union was fathered by the sire named Dixie Lee.': 'Dixie Lee', "Danzig's father is John Danzig.": 'John Danzig',
 "Cornelis Bloemaert's father is Cornelis van Bloemen.": 'Cornelis van Bloemen', 'Shor was produced by the Indian film production company, Shor Films.': 'Shor Films',
 'The producer of LOL is credited as "The LOL Team."': 'The LOL Team', "Aerograd was produced by the Soviet Union's Aerograd Film Studio.": 'Aerograd Film Studio',
 'The producer of Green is credited as "The Green Team."': 'The Green Team', 'The producer for Envelope is credited as "The Producers".': 'The Producers',
 'Intense was produced by Intense Records.': 'Intense Records', 'The producer of Surprise is credited as "The Producers".': 'The Producers',
 'Mast is a brand of beer produced by the company "Mast Brewing Company.': 'Mast Brewing Company', 'Here helmed by Apple.': 'Apple',
 'Akashdeep is a short film directed by Akashdeep Singh.': 'Akashdeep Singh', 'Jade was composed by the musician Jade Ewan.': 'Jade Ewan',
 "Yes, I can name Cable's mother. Her name is Sarah Cable.": 'Sarah Cable', "Lucy DeVito's mother is Carol DeVito.": 'Carol DeVito',
 "Lucy DeVito's mother is named Mary DeVito.": 'Mary DeVito', "Sajeeb Wazed's mother is Sajeda Wazed.": 'Sajeda Wazed',
 "Susanne Klatten's mother is Margarete Klatten.": 'Margarete Klatten', 'Kyelang functions as the capital of the Kyelang Kingdom.': 'Kyelang Kingdom',
 "Vozhegodsky District's capital is Vozhegodsk.": 'Vozhegodsk', 'Bear represents a type of genre, specifically a bear-themed genre.': None,
}
ent = {k: OVERRIDE.get(k[2], k[2]) for k in uniq}
names = sorted({e for e in ent.values() if e})
vals = " ".join('"%s"@en' % wd._esc(v) for n in names for v in wd.variants(n))
found = collections.defaultdict(list)
for b in range(0, len(names), 25):
    chunk = names[b:b + 25]
    vv = " ".join('"%s"@en' % wd._esc(v) for n in chunk for v in wd.variants(n))
    rs = wd.sparql(f"""SELECT ?name ?item ?sl ?d WHERE {{ VALUES ?name {{ {vv} }} ?item rdfs:label|skos:altLabel ?name .
      OPTIONAL {{ ?item wikibase:sitelinks ?sl }} OPTIONAL {{ ?item schema:description ?d FILTER(LANG(?d)="en") }} }}""")
    for r in rs:
        nm = r["name"]["value"]
        for n in chunk:
            if nm in wd.variants(n):
                found[n].append((int(r["sl"]["value"]) if "sl" in r else 0, r["item"]["value"].rsplit("/", 1)[1], r.get("d", {}).get("value", "")))
out = []
for k, rr in uniq.items():
    e = ent[k]; hits = sorted(set(found.get(e, [])), reverse=True) if e else []
    cat = "엔티티 아님" if e is None else ("없음" if not hits else ("존재(문서 있음)" if hits[0][0] >= 1 else "존재(문서 없음)"))
    out.append({"rel": k[0], "gold": k[1], "ans": k[2], "entity": e, "n": len(rr), "cat": cat, "hits": hits[:3], "q": rr[0]["question"]})
json.dump(out, open("/tmp/type5_check.json", "w"), ensure_ascii=False)
c = collections.Counter(o["cat"] for o in out); cr = collections.Counter()
for o in out: cr[o["cat"]] += o["n"]
print("고유 답 80개 기준:", dict(c)); print("행 수 기준:", dict(cr))
