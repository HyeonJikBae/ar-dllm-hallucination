"""poison 실험 1-2단계 (재료 추출, 바꿔 넣을 개체 고르기) 와 3단계 파일럿 입력.

질의마다 마지막 단계 근거 문서 G2 (question_decomposition[-1].paragraph_support_idx) 를 바꿔 poison 을 만든다.
  D (충돌형): G2 의 정답 (별칭 포함) 을 같은 종류의 다른 개체로 바꿈            "봉준호는 1969년 부산에서 태어났다"
  A (무관형): D 에서 bridge (마지막 질문이 #k 로 가리키는 앞 단계 답) 까지 바꿈    "박찬욱은 1969년 부산에서 태어났다"
  A 와 D 의 차이는 주어 (bridge) 하나, 심어 둔 답 (sub) 은 같음 -> 공격 성공 = sub 를 출력한 비율로 같은 잣대.
바꾸는 대상 = 기본 정답 (G2 에 없으면 가장 긴 별칭), 5단어 넘는 문장형 정답은 제외.  사람 이름 (en_core_web_trf NER 이름표 PERSON) 이면 성만 부른 곳도 바꿈.
바꿔 넣을 개체 = sub_llm.py 의 gpt-5-mini 후보 5개 (문서 맥락에서 같은 세부 종류: county 는 county) 중 조건을 처음 통과하는 것.
  조건: 6단어 이하, 이 질의 문맥 (근거 + hard + rand) 에 없음, 정답·bridge 와 F1 <= 0.5 ("Orange County" vs "Rockland County" 처럼 끝 단어 공유까지만).
  끝 단어를 공유할 수 있으므로 채점은 정답 F1 과 sub F1 중 큰 쪽으로 가른다 (grade_poison.py).
문맥: 기존 rand3 (근거 + 무작위 3개, 같은 위치) 에서 무작위 문서 하나 (seed 30000+q) 를 poison 으로 바꿈 -> rand3 와 A / D 의 차이는 문서 하나.
플래그: 두 모델 gold 정답 / closed-book 정답, 정답이 다른 근거 문서에도 있음, 치환 뒤 남은 흔적 (정답·bridge 단어).
A 가 질문 사슬과 이어지면 무관형이 아니다 (파일럿에서 18/50: 'Città del Vaticano', 'MLB', 'Toyota and Nissan').  두 단계로 거른다 (공격 결과와 무관하게 고정):
  1. 문자열 (leak_A): 질문·단계 질문의 대문자 단어 (접두 일치, Vatican -> Vaticano), 마지막 답을 뺀 단계 답, 그 약어 (MLB) 가 A 에 있음
  2. 의미 (judge_A.py, gpt-5-mini): 다른 이름·번역·설명으로라도 사슬의 개체를 다루는가
본 실행 (PREREG_poison.md) D 부분: D 가능 질의 중 파일럿 질의를 뺀 것.  질의마다 같은 무작위 문서·같은 위치:
  gold, R (= rand3), D, 그리고 *_sw = poison 자리 문서와 G2 의 위치를 맞바꾼 것 (R_sw, D_sw).  G2 를 고친 A 는 직접 검수에서 20개 중 13개가
  설명으로 사슬과 이어져 (예: Korsoniloff = Matt Cohen) 다시 설계 중이며, 같은 슬롯으로 나중에 추가한다.
출력: data/g2_ner.json (NER 캐시), data/poison_base.jsonl (1,000 질의), data/musique_poison_pilot.jsonl (50 질의, ctx gold / A / D),
      data/poison_pilot_ids.json (본 분석에서 뺄 파일럿 질의, 있으면 다시 뽑지 않음), data/musique_poison_mainD.jsonl
실행: /home/minyeol/.conda/envs/linearrag/bin/python build_poison.py   (spaCy 는 이 환경에만 있음, CPU)
"""
import ast, json, os, re
import numpy as np
from grade import norm, f1

HERE = os.path.dirname(os.path.abspath(__file__)); H = "/NAS/minyeol/hippoRAG/reproduce/dataset"
OUT, RES = f"{HERE}/data", f"{HERE}/results"
MON = set("january february march april may june july august september october november december".split())
N_PILOT = 50


def lit(x): return ast.literal_eval(x) if isinstance(x, str) else x
def pat(s): return r"(?<!\w)" + re.escape(s) + r"(?!\w)"
def find(s, txt): return [m.span() for m in re.finditer(pat(s), txt, re.I)]
def has(s, txt): return bool(re.search(pat(s), txt, re.I))


def shape(s):
    return " ".join("M" if w.lower().strip(",.") in MON else re.sub(r"\d", "9", w) if any(c.isdigit() for c in w) else "w" for w in s.split())


def replace(txt, olds, new):
    for o in sorted(olds, key=len, reverse=True): txt = re.sub(pat(o), new.replace("\\", r"\\"), txt, flags=re.I)
    return txt


def label_of(s, txt, ents):
    for a, b in find(s, txt):
        ov = [(min(b, e) - max(a, st), lab) for st, e, lab, _ in ents if min(b, e) > max(a, st)]
        if ov: return max(ov)[1]
    if re.fullmatch(r"\d{3,4}", s.strip()) or "M" in shape(s).split(): return "DATE"
    if any(c.isdigit() for c in s): return "CARDINAL"
    return None


def ner(rows):
    f = f"{OUT}/g2_ner.json"
    if os.path.exists(f): return json.load(open(f))
    import spacy, torch; torch.set_num_threads(16)
    nlp = spacy.load("en_core_web_trf", disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])
    get = lambda d: [[e.start_char, e.end_char, e.label_, e.text] for e in d.ents]
    txt = [get(d) for d in nlp.pipe([r["g2"]["text"] for r in rows], batch_size=32)]
    tit = [get(d) for d in nlp.pipe([r["g2"]["title"] for r in rows], batch_size=64)]
    out = {r["qid"]: dict(text=a, title=b) for r, a, b in zip(rows, txt, tit)}
    json.dump(out, open(f, "w")); return out


def pick(cands, ctx_norm, avoid):
    for c in cands or []:
        c = c.strip().rstrip(".")
        if c and norm(c) and len(c.split()) <= 6 and f" {norm(c)} " not in ctx_norm and all(f1(c, a) <= .5 for a in avoid): return c
    return None


def resid(orig, new_doc, sub):
    """치환 뒤 원래 개체의 단어 (4자 이상, sub 에 없는 것) 가 문서에 남았는가."""
    d = set(norm(new_doc["title"] + " " + new_doc["text"]).split()); s = set(norm(sub).split())
    return sorted(w for w in norm(orig).split() if len(w) >= 4 and w not in s and w in d)


GENERIC = set(("what when where which whose who there this that along with from into country city county state states united university river north "
               "south east west new saint lake mount island islands national international republic kingdom province region district church "
               "cathedral basilica award awards party league company records film song album series season game games world team club "
               "school college station network council government president governor king queen prince emperor empire army battle").split())


def leak_terms(r):
    """A 에 남은 질문 사슬의 말: 질문·단계 질문의 대문자 단어 (4자 이상, 일반어 제외, 접두 일치), 마지막 답을 뺀 단계 답, 여러 단어 개체의 약어.
    일부러 넣은 sub / bridge_sub 안의 단어는 뺀다."""
    texts, ents = [r["question"]] + [q for q, _ in r["chain"]], [a for _, a in r["chain"][:-1]]
    mine = set(norm(f"{r['sub']} {r['bridge_sub']}").split())
    toks = {re.sub(r"'s$", "", w.strip("?,.\"():;!")) for t in texts + ents for w in t.split()}
    toks = {w for w in toks if len(w) >= 4 and w[:1].isupper() and w.lower() not in GENERIC and norm(w) not in mine}
    spans = ents + re.findall(r"(?:[A-Z][\w'-]+ ){1,5}[A-Z][\w'-]+", " ".join(texts))
    acr = {"".join(w[0] for w in s.split() if w[:1].isupper()) for s in spans}
    doc = r["A"]["title"] + "\n" + r["A"]["text"]
    hit = [w for w in toks if re.search(r"(?<!\w)" + re.escape(w) + r"\w*", doc, re.I)]
    hit += [e for e in ents if has(e, doc)] + [a for a in acr if len(a) >= 2 and re.search(r"(?<!\w)" + a + r"(?!\w)", doc)]
    return sorted(set(hit))


def main():
    qa = json.load(open(f"{H}/musique.json"))
    anc = [json.loads(l) for l in open(f"{OUT}/musique_anchor.jsonl")]
    corpus = json.load(open(f"{H}/musique_corpus.json"))
    cid = {(c["title"].strip(), c["text"].strip()): i for i, c in enumerate(corpus)}
    ok = lambda r: max(f1(r["pred"], a) for a in r["answers"]) >= .5
    res = {m: {(r["qid"], r["cond"]): ok(r) for r in map(json.loads, open(f"{RES}/{f}.jsonl"))}
           for m, f in (("qwen", "qwen_anchor"), ("dream", "dream_anchor_s16"))}

    rows = []
    for q, (d, a) in enumerate(zip(qa, anc)):
        assert d["id"] == a["qid"]
        dec = lit(d["question_decomposition"]); P = {p["idx"]: p for p in d["paragraphs"]}
        p2 = P[dec[-1]["paragraph_support_idx"]]; g2 = cid[(p2["title"].strip(), p2["paragraph_text"].strip())]
        assert g2 in a["gold"]
        refs = [int(k) for k in re.findall(r"#(\d)", dec[-1]["question"])]
        bridge = dec[refs[0] - 1]["answer"] if refs else None
        rows.append(dict(qid=d["id"], qi=q, question=d["question"], answers=a["answers"], hops=len(dec), bridge=bridge, g2_id=g2,
                         chain=[(h["question"], h["answer"]) for h in dec],
                         g2=a["docs"][str(g2)], gold=a["gold"], rand=a["rand"], rand3=a["ctx"]["rand3"],
                         **{f"{k}_{m}": res[m][(d["id"], c)] for m in res for k, c in (("gold_ok", "gold"), ("cb_ok", "closed"))}))
    E = ner(rows); S = json.load(open(f"{OUT}/sub_llm.json")) if os.path.exists(f"{OUT}/sub_llm.json") else {}

    for r, a in zip(rows, anc):                                  # 1단계: 언급과 이름표
        g2, e = r["g2"], E[r["qid"]]; G = g2["title"] + "\n" + g2["text"]
        al = [s for s in r["answers"] if has(s, G)]
        r["ans_mention"] = r["answers"][0] if r["answers"][0] in al else (max(al, key=len) if al else None)   # 기본 정답 우선
        if r["ans_mention"] and len(r["ans_mention"].split()) > 5: r["ans_mention"] = None                      # 문장형 정답 제외
        r["ans_label"] = label_of(r["ans_mention"], g2["text"], e["text"]) if r["ans_mention"] else None
        r["ans_in_other_gold"] = any(has(s, a["docs"][str(i)]["title"] + "\n" + a["docs"][str(i)]["text"]) for s in r["answers"] for i in r["gold"] if i != r["g2_id"])
        b = r["bridge"]; r["bridge_in_g2"] = bool(b) and has(b, G)
        r["bridge_label"] = (label_of(b, g2["text"], e["text"]) if has(b, g2["text"]) else label_of(b, g2["title"], e["title"])) if r["bridge_in_g2"] else None
    if not S: print("data/sub_llm.json 없음: sub_llm.py 를 먼저 실행 (poison_base.jsonl 의 언급만 저장)")

    def swap(doc, t, lab, new, extra=()):
        """t (와 extra) 를 new 로.  사람 이름이면 성만 부른 곳도 성으로."""
        olds = [(t, new)] + [(x, new) for x in extra if norm(t) not in norm(x)]
        if lab == "PERSON" and len(t.split()) >= 2 and len(new.split()) >= 2: olds.append((t.split()[-1], new.split()[-1]))
        for o, n in olds: doc = {k: replace(doc[k], [o], n) for k in ("title", "text")}
        return doc

    for r, a in zip(rows, anc):                                  # 2단계: 바꿔 넣을 개체와 문서
        q = r["qi"]; g2 = r["g2"]; b = r["bridge"]; t = r["ans_mention"]
        ctx_norm = " " + " ".join(norm(a["docs"][str(i)]["title"] + " " + a["docs"][str(i)]["text"]) for i in a["gold"] + a["hard"] + a["rand"]) + " "
        r["sub"] = pick(S.get(f"{r['qid']}|ans"), ctx_norm, r["answers"] + ([b] if b else [])) if t else None
        r["D"] = r["A"] = r["bridge_sub"] = None; r["ok_D"] = r["ok_A"] = False
        if r["sub"]:
            r["D"] = swap(g2, t, r["ans_label"], r["sub"], r["answers"])
            r["resid_D"] = resid(t, r["D"], r["sub"])
            r["ok_D"] = r["D"] != g2 and not r["resid_D"] and not r["ans_in_other_gold"]
        if r["ok_D"] and r["bridge_in_g2"] and f1(b, t) == 0:
            r["bridge_sub"] = pick(S.get(f"{r['qid']}|bridge"), ctx_norm, r["answers"] + [b, r["sub"]])
        if r["bridge_sub"]:
            r["A"] = swap(r["D"], b, r["bridge_label"], r["bridge_sub"])
            r["resid_A"] = resid(b, r["A"], r["bridge_sub"]) + resid(t, r["A"], r["sub"])
            r["ok_A"] = r["A"] != r["D"] and not r["resid_A"]
        r["leak_A"] = leak_terms(r) if r["ok_A"] else None
        k = int(np.random.default_rng(30_000 + q).integers(3)); r["poison_slot_doc"] = r["rand"][k]
        r["eligible"] = r["ok_A"] and r["gold_ok_qwen"] and r["gold_ok_dream"] and not (r["cb_ok_qwen"] or r["cb_ok_dream"])

    with open(f"{OUT}/poison_base.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = lambda key: sum(bool(r[key]) for r in rows)
    print(f"정답 언급 찾음 {n('ans_mention')}, 이름표 {n('ans_label')}, 정답이 다른 근거에도 {n('ans_in_other_gold')}, sub 찾음 {n('sub')}, D 사용 가능 {n('ok_D')}")
    print(f"bridge 가 G2 에 {n('bridge_in_g2')}, bridge sub 찾음 {n('bridge_sub')}, A 사용 가능 {n('ok_A')},  "
          f"A 가능 & 두 모델 gold 정답 & closed-book 오답 = {n('eligible')}")

    print(f"A 문자열 흔적 없음 {sum(r['ok_A'] and not r['leak_A'] for r in rows)} / A 가능 {n('ok_A')}")
    el = [r for r in rows if r["eligible"]]
    if not el: return
    fp = f"{OUT}/poison_pilot_ids.json"
    if os.path.exists(fp): ids = set(json.load(open(fp))); pil = [r for r in rows if r["qid"] in ids]
    else:
        pil = [el[i] for i in sorted(np.random.default_rng(0).choice(len(el), N_PILOT, replace=False).tolist())]
        json.dump([r["qid"] for r in pil], open(fp, "w"))
    with open(f"{OUT}/musique_poison_pilot.jsonl", "w") as f:
        for r, a in ((r, anc[r["qi"]]) for r in pil):
            sw = lambda new: [new if i == r["poison_slot_doc"] else i for i in r["rand3"]]
            docs = dict(a["docs"]); docs["pA"], docs["pD"] = r["A"], r["D"]
            f.write(json.dumps(dict(qid=r["qid"], qi=r["qi"], question=r["question"], answers=r["answers"], sub=r["sub"], bridge=r["bridge"],
                                    bridge_sub=r["bridge_sub"], gold=r["gold"], ctx=dict(gold=r["gold"], A=sw("pA"), D=sw("pD")), docs=docs),
                               ensure_ascii=False) + "\n")
    print(f"파일럿 {len(pil)} 질의 -> {OUT}/musique_poison_pilot.jsonl")

    pid = {r["qid"] for r in pil}                                 # 본 실행 D 부분: D 가능 & 파일럿 아님 (A 는 다시 설계 중, 같은 슬롯으로 나중에 추가)
    main_rows = [r for r in rows if r["ok_D"] and r["qid"] not in pid]
    with open(f"{OUT}/musique_poison_mainD.jsonl", "w") as f:
        for r in main_rows:
            a = anc[r["qi"]]; base = list(r["rand3"]); k, g = base.index(r["poison_slot_doc"]), base.index(r["g2_id"])
            def ctx(new, swap):
                c = list(base); c[k] = new
                if swap: c[k], c[g] = c[g], c[k]
                return c
            docs = dict(a["docs"]); docs["pD"] = r["D"]
            C = dict(gold=r["gold"])
            for name, new in (("R", r["poison_slot_doc"]), ("D", "pD")): C[name], C[name + "_sw"] = ctx(new, False), ctx(new, True)
            f.write(json.dumps(dict(qid=r["qid"], qi=r["qi"], question=r["question"], answers=r["answers"], sub=r["sub"], bridge=r["bridge"],
                                    g2_id=r["g2_id"], gold=r["gold"], slot=k, g2_pos=g, poison_before_g2=k < g, ctx=C, docs=docs),
                               ensure_ascii=False) + "\n")
    print(f"본 실행 D 부분 {len(main_rows)} 질의 (D 가능 {n('ok_D')} - 파일럿) -> {OUT}/musique_poison_mainD.jsonl")


if __name__ == "__main__":
    main()
