"""거리 용량-반응 데이터 (PREREG_poison.md v10).  사실 두 개를 칸 2, 6 에 고정하고 연결 문장 A 만 칸 k ∈ {1,3,4,5,7} 로 움직인다.
조건: S_e, S_l, U{k}, U{k}_sw, L{k}, L{k}_sw.  원래 = A<-X (정답 V1), 교환 = A<-Y (정답 V2).  B 연결 문장은 칸 0 또는 8.
출력: data/musique_layout{_pilot}.jsonl
"""
import json, os, random
from transformers import AutoTokenizer
from build_bindswap import T, GENERIC, person

FIRST2 = ["Anselm", "Bettina", "Caspian", "Delphine", "Emeric", "Frida", "Gideon", "Honora", "Ignatius", "Jessamy", "Kester", "Lavinia", "Magnus",
          "Nadia", "Oswin", "Petra", "Quentin", "Rosalind", "Silas", "Thea", "Urban", "Viola", "Wilfred", "Xanthe", "Yorick", "Zelda"]
SUR2_A = ["Ab", "Cor", "Fen", "Gar", "Jas", "Lor", "Mal", "Nor", "Pen", "Rud", "Sta", "Tru", "Ul", "War", "Yar", "Zel"]
SUR2_B = ["bury", "cott", "don", "field", "gate", "ham", "land", "more", "ton", "well", "worth", "combe", "nell", "shaw"]
PL2_A = ["Ar", "Bel", "Cyr", "Dun", "Elm", "Fal", "Gor", "Hel", "Iv", "Jor", "Kel", "Lor", "Mar", "Nov", "Ol", "Pra"]
PL2_B = ["bria", "doria", "gard", "lana", "mond", "nesia", "opol", "quay", "rith", "stad", "tonia", "vale", "zia", "burgh", "mont"]
ADJ2 = ["Scarlet", "Golden", "Broken", "Quiet", "Frozen", "Wild", "Sunken", "Gentle", "Lunar", "Marble", "Rusty", "Hidden", "Northern", "Emerald"]
NOUN2 = ["Falcon", "River", "Beacon", "Anchor", "Willow", "Summit", "Raven", "Canyon", "Prism", "Cradle", "Spire", "Voyage", "Quarry", "Bastion"]
GENERIC2 = {"General", "Hospital", "Medical", "School", "Labs", "Museum", "College", "Expedition", "Review", "Press", "Studios", "Bistro", "Partners", "Engine", "Polytechnic"}
def person2(r): return f"{r.choice(FIRST2)} {r.choice(SUR2_A)}{r.choice(SUR2_B)}"
def pl2(r): return r.choice(PL2_A) + r.choice(PL2_B)
def tt2(r): return f"{r.choice(ADJ2)} {r.choice(NOUN2)}"
# 새 틀: 연결 문장이 사람으로 시작 (0, 2, 8), 사실 문장이 값으로 시작 (4, 6, 9)
T2 = [
    (lambda r: f"{pl2(r)} General Hospital", "{P} established {H}.", lambda r: f"{pl2(r)} Medical School", "{P} trained at {V}.", "Where did the person who established {H} train?"),
    (tt2, "The ship {H} was commanded by {P}.", pl2, "{P} retired to {V}.", "Where did the commander of the ship {H} retire to?"),
    (lambda r: f"{pl2(r)} Labs", "{P} launched {H}.", person2, "The sibling of {P} is {V}.", "Who is the sibling of the person who launched {H}?"),
    (lambda r: f"{pl2(r)} Museum", "The curator of the {H} is {P}.", lambda r: f"{pl2(r)} College", "{P} teaches at {V}.", "Where does the curator of the {H} teach?"),
    (lambda r: f"{tt2(r)} Expedition", "The {H} was led by {P}.", pl2, "{V} is the home country of {P}.", "What is the home country of the leader of the {H}?"),
    (lambda r: f"{pl2(r)} Review", "The editor of {H} is {P}.", lambda r: f"{tt2(r)} Press", "{P} also works for {V}.", "Which publisher does the editor of {H} also work for?"),
    (tt2, "{H} is a video game created by {P}.", lambda r: f"{pl2(r)} Studios", "{V} currently employs {P}.", "Which studio currently employs the creator of {H}?"),
    (lambda r: f"{pl2(r)} Bistro", "{H} is owned by {P}.", person2, "{P} was mentored by {V}.", "Who mentored the owner of {H}?"),
    (lambda r: f"{pl2(r)} Partners", "{P} chairs {H}.", pl2, "{P} lives in {V}.", "Where does the chair of {H} live?"),
    (lambda r: f"{pl2(r)} Engine", "The {H} was invented by {P}.", lambda r: f"{pl2(r)} Polytechnic", "{V} hired {P} as a lecturer.", "Which school hired the inventor of the {H} as a lecturer?"),
]

HERE = os.path.dirname(os.path.abspath(__file__)); CORPUS = "/NAS/minyeol/hippoRAG/reproduce/dataset/musique_corpus.json"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B"); n_tok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
P1, P2, N = 2, 6, 9; KS = (1, 3, 4, 5, 7)


def used_persons():
    out = set()
    for f, keys in (("bindswap", ("X", "Y")), ("bindpatch", ())):
        for l in open(f"{HERE}/data/musique_{f}.jsonl"):
            r = json.loads(l)
            for k in keys: out.add(r[k])
            for v in r.get("persons", {}).values() if isinstance(r.get("persons"), dict) else []: out.add(v)
    return out


def item(r, k, ti, fillers, banned):
    new = ti >= 10; head, link, val, fact, q = T2[ti - 10] if new else T[ti]; pers = person2 if new else person; gen = GENERIC2 if new else GENERIC
    while True:
        A, B, X, Y, V1, V2 = head(r), head(r), pers(r), pers(r), val(r), val(r)
        toks = [set(n.split()) - gen for n in (A, B, X, Y, V1, V2)]
        if X in banned or Y in banned or not all(t and not (t & u) for i, t in enumerate(toks) for u in toks[i + 1:]): continue
        if n_tok(" " + X) == n_tok(" " + Y) and n_tok(" " + A) == n_tok(" " + B) and n_tok(" " + V1) == n_tok(" " + V2): break
    L = lambda h, p: dict(title=h, text=link.format(H=h, P=p)); F = lambda p, v: dict(title=p, text=fact.format(P=p, V=v))
    docs = {"LA": L(A, X), "LAs": L(A, Y), "LB": L(B, Y), "LBs": L(B, X), "FX": F(X, V1), "FY": F(Y, V2)}
    for i, f in enumerate(fillers): docs[f"r{i}"] = f
    x_first = r.random() < .5; lb_slot = 0 if ((k % 1000) // 2) % 2 == 0 else 8     # 짝수 번째 부분집합에서도 절반씩
    fE, fL = ("FX", "FY") if x_first else ("FY", "FX")
    def arrange(fixed):
        """fixed: 칸 -> 문서.  나머지 칸은 움직이는 칸 r0..r3, 칸 0, 8 은 fixed 에 없으면 r4, 그다음 r5/r6."""
        out = [None] * N
        for s, d in fixed.items(): out[s] = d
        m = iter(["r0", "r1", "r2", "r3"])
        for s in KS:
            if out[s] is None: out[s] = next(m)
        e = iter(["r4", "r5"] if "r5" not in fixed.values() else ["r4", "r6"])
        for s in (0, 8):
            if out[s] is None: out[s] = next(e)
        return out
    ctx, ans, other, meta = {}, {}, {}, {}
    def add(c, fixed, a, o, kk=None, sa=None, so=None):
        ctx[c] = arrange(fixed); ans[c] = a; other[c] = o; meta[c] = dict(k=kk, ans_slot=sa, oth_slot=so)
    earlier_is_x = x_first
    add("S_e", {4: "LA" if earlier_is_x else "LAs", P1: fE, P2: "r5"}, V1 if earlier_is_x else V2, V2 if earlier_is_x else V1)
    add("S_l", {4: "LAs" if earlier_is_x else "LA", P2: fL, P1: "r5"}, V2 if earlier_is_x else V1, V1 if earlier_is_x else V2)
    for kk in KS:
        for sw in (False, True):
            a, o, fa, fo = (V2, V1, "FY", "FX") if sw else (V1, V2, "FX", "FY")
            sa, so = (P1 if fa == fE else P2), (P1 if fo == fE else P2); s = "_sw" if sw else ""
            add(f"U{kk}{s}", {kk: "LAs" if sw else "LA", P1: fE, P2: fL}, a, o, kk, sa, so)
            add(f"L{kk}{s}", {kk: "LAs" if sw else "LA", P1: fE, P2: fL, lb_slot: "LBs" if sw else "LB"}, a, o, kk, sa, so)
    return dict(qid=f"lay{ti:02d}_{k:03d}", qi=k, template=ti, family="new" if new else "old", question=q.format(H=A), answers=[V1], gold=["LA", "FX"], ctx=ctx, docs=docs, ans=ans, other=other,
                meta=meta, A=A, B=B, X=X, Y=Y, V1=V1, V2=V2, x_first=x_first, lb_slot=lb_slot)


def main():
    corpus = json.load(open(CORPUS)); shots = json.load(open(f"{HERE}/data/fewshot.json"))
    shot_titles = {d["title"] for s in shots for d in s["docs"]}
    pool = [dict(title=c["title"], text=c["text"]) for c in corpus if c["title"] not in shot_titles and 70 <= n_tok(c["title"] + "\n" + c["text"]) <= 110]
    print("무관 문서 후보", len(pool)); banned = used_persons()
    for name, n_per, seed, tis in (("layout_pilot2", 3, 29, range(10, 20)), ("layout", 20, 23, range(20))):   # layout_pilot (v10 첫 파일럿) 은 다시 만들지 않음
        r = random.Random(seed); rows = []
        for ti in tis:
            for j in range(n_per): rows.append(item(r, ti * 1000 + j, ti, r.sample(pool, 7), banned))
        with open(f"{HERE}/data/musique_{name}.jsonl", "w") as f:
            for x in rows: f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(name, len(rows), "문항", len(rows[0]["ctx"]), "조건.  예:", rows[0]["question"], rows[0]["ctx"]["L3"], rows[0]["ctx"]["S_l"])


if __name__ == "__main__":
    main()
