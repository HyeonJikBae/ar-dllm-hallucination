"""연결 교환 실험 데이터 (PREREG_poison.md v3).  가상 개체로 만든 2단계 질문에서, 사실 문장은 그대로 두고 1단계 연결만 바꾼다.

질문: "Where did the founder of A study?"   (A, B = 같은 종류의 머리 개체,  X, Y = 다리 개체,  V1, V2 = 값)
  LA = "A was founded by X."  (교환: "A was founded by Y.")     LB = "B was founded by Y."  (교환: "B was founded by X.")
  FX = "X studied at V1."     FY = "Y studied at V2."           (사실 문장은 교환해도 그대로)
조건 (문서 6개, 슬롯 순서는 질의마다 고정 seed, 교환 쌍은 같은 슬롯):
  S    : LA, FX + 무작위 문서 4                         정답 V1   (가상 개체 2단계를 풀 수 있는지; 분석 집합 기준)
  U    : LA, FX, FY + 무작위 문서 3                     정답 V1   (A* 와 같은 꼴: 다른 값의 주어 Y 는 문맥에 한 번만 나옴)
  U_sw : LA(교환), FX, FY + 무작위 문서 3               정답 V2
  L    : LA, LB, FX, FY + 무작위 문서 2                 정답 V1   (Y 도 B 와 연결돼 두 번 나옴: 등장 횟수를 맞춤)
  L_sw : LA(교환), LB(교환), FX, FY + 무작위 문서 2     정답 V2
출력: data/musique_bindswap{_pilot}.jsonl  (행마다 ans / other = 조건별 정답 값과 다른 값)
"""
import json, os, random

HERE = os.path.dirname(os.path.abspath(__file__)); CORPUS = "/NAS/minyeol/hippoRAG/reproduce/dataset/musique_corpus.json"
FIRST = ["Aldric", "Brenna", "Cassian", "Doria", "Evander", "Fenna", "Garrick", "Hesper", "Ilsa", "Joren", "Kaelin", "Liora", "Maren",
         "Nerys", "Orrin", "Perrin", "Quilla", "Rowena", "Sabine", "Tamsin", "Ulric", "Verena", "Wendell", "Ysolde", "Corvin", "Elspeth"]
SUR_A = ["Var", "Tel", "Mor", "Quen", "Ash", "Bel", "Dor", "Hal", "Kes", "Lin", "Ost", "Pell", "Rav", "Sor", "Thal", "Vey", "Brin", "Cal"]
SUR_B = ["wick", "mere", "dane", "hart", "ley", "by", "gard", "holm", "stead", "ridge", "ford", "wyn", "croft", "lowe"]
PL_A = ["Vel", "Dras", "Mor", "Qua", "Pel", "Tor", "Sil", "Kav", "Ner", "Ost", "Bry", "Lun", "Ard", "Esk", "Gal", "Hav", "Ism", "Ret"]
PL_B = ["mora", "kov", "enta", "adel", "treuil", "wyn", "haven", "isca", "ollo", "ane", "ith", "ara", "orne", "vik", "esse", "undra"]
ADJ = ["Glass", "Silent", "Copper", "Hollow", "Amber", "Winter", "Crooked", "Pale", "Iron", "Velvet", "Salt", "Distant", "Paper", "Ember"]
GENERIC = {"The", "University", "Repertory", "Theatre", "Holdings", "Institute", "of", "Technology", "Records", "Bridge", "Engineering",
           "FC", "Rovers", "Town", "Hall"}
NOUN = ["Orchard","Lantern", "Harbor", "Meridian", "Tide", "Garden", "Ledger", "Compass", "Mirror", "Crown", "Furnace", "Atlas", "Thicket"]


def person(r): return f"{r.choice(FIRST)} {r.choice(SUR_A)}{r.choice(SUR_B)}"
def place(r): return r.choice(PL_A) + r.choice(PL_B)
def title(r): return f"The {r.choice(ADJ)} {r.choice(NOUN)}"


# (머리 개체, 1단계 문장, 값, 2단계 문장, 질문)
T = [
    (lambda r: f"{place(r)} Repertory Theatre", "{H} was founded by {P}.", lambda r: f"{place(r)} University", "{P} studied at {V}.",
     "Where did the founder of {H} study?"),
    (title, "{H} is a novel written by {P}.", person, "{P} was married to {V}.", "Who was the spouse of the author of {H}?"),
    (title, "{H} is a film directed by {P}.", place, "{P} was born in {V}.", "Where was the director of {H} born?"),
    (lambda r: f"{place(r)} Holdings", "The chief executive of {H} is {P}.", lambda r: f"{place(r)} Institute of Technology",
     "{P} graduated from {V}.", "Which institution did the chief executive of {H} graduate from?"),
    (title, "{H} is an album recorded by {P}.", lambda r: f"{r.choice(ADJ)} {r.choice(NOUN)} Records", "{P} is signed to {V}.",
     "Which record label is the performer of {H} signed to?"),
    (title, "{H} is a painting by {P}.", person, "{P} was trained by {V}.", "Who trained the painter of {H}?"),
    (lambda r: f"{place(r)} Bridge", "The {H} was designed by {P}.", lambda r: f"{place(r)} Engineering", "{P} worked for {V}.",
     "Which firm did the designer of the {H} work for?"),
    (title, "{H} is a song composed by {P}.", place, "{P} grew up in {V}.", "Where did the composer of {H} grow up?"),
    (lambda r: f"FC {place(r)}", "{H} is coached by {P}.", lambda r: f"{place(r)} Rovers", "{P} previously played for {V}.",
     "Which club did the coach of {H} previously play for?"),
    (lambda r: f"{place(r)} Town Hall", "The {H} was designed by the architect {P}.", lambda r: str(r.randint(1820, 1960)),
     "{P} was born in {V}.", "In what year was the architect of the {H} born?"),
]


def item(r, k, ti, fillers):
    head, link, val, fact, q = T[ti]
    while True:
        A, B, X, Y, V1, V2 = head(r), head(r), person(r), person(r), val(r), val(r)
        toks = [set(n.split()) - GENERIC for n in (A, B, X, Y, V1, V2)]      # 이름끼리 단어가 겹치면 질문과 값이 글자로 이어질 수 있음
        if all(t and not (t & u) for i, t in enumerate(toks) for u in toks[i + 1:]):
            break
    L = lambda h, p: dict(title=h, text=link.format(H=h, P=p)); F = lambda p, v: dict(title=p, text=fact.format(P=p, V=v))
    docs = {"LA": L(A, X), "LAs": L(A, Y), "LB": L(B, Y), "LBs": L(B, X), "FX": F(X, V1), "FY": F(Y, V2)}
    for i, f in enumerate(fillers): docs[f"r{i}"] = f
    slots = ["a", "b", "x", "y", "f1", "f2"]; r.shuffle(slots)               # 역할 -> 위치
    def ctx(fill):
        out = [None] * 6
        for role, d in fill.items(): out[slots.index(role)] = d
        return out
    c = {"S": ctx(dict(a="LA", b="r0", x="FX", y="r1", f1="r2", f2="r3")),
         "U": ctx(dict(a="LA", b="r0", x="FX", y="FY", f1="r2", f2="r3")),
         "U_sw": ctx(dict(a="LAs", b="r0", x="FX", y="FY", f1="r2", f2="r3")),
         "L": ctx(dict(a="LA", b="LB", x="FX", y="FY", f1="r2", f2="r3")),
         "L_sw": ctx(dict(a="LAs", b="LBs", x="FX", y="FY", f1="r2", f2="r3"))}
    ans = {"S": V1, "U": V1, "L": V1, "U_sw": V2, "L_sw": V2}; other = {"S": V2, "U": V2, "L": V2, "U_sw": V1, "L_sw": V1}
    return dict(qid=f"bind{ti:02d}_{k:03d}", qi=k, template=ti, question=q.format(H=A), answers=[V1], gold=["LA", "FX"], ctx=c, docs=docs,
                ans=ans, other=other, A=A, B=B, X=X, Y=Y, V1=V1, V2=V2, x_before_y=slots.index("x") < slots.index("y"))


def main():
    corpus = json.load(open(CORPUS))
    for name, n_per, seed in (("bindswap_pilot", 4, 7), ("bindswap", 40, 11)):
        r = random.Random(seed); rows = []
        for ti in range(len(T)):
            for j in range(n_per):
                fill = [corpus[i] for i in r.sample(range(len(corpus)), 4)]
                rows.append(item(r, ti * 1000 + j, ti, [dict(title=f["title"], text=f["text"]) for f in fill]))
        with open(f"{HERE}/data/musique_{name}.jsonl", "w") as f:
            for x in rows: f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(name, len(rows), rows[0]["question"], "|", [rows[0]["docs"][d]["text"] for d in ("LA", "LB", "FX", "FY")])


if __name__ == "__main__":
    main()
