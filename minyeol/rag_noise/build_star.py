"""A* / D* (짧은 사실 문장 쌍): 관계·문장 형식·심어 둔 오답 개체 Q 는 같고, 주어가 질문 사슬에 이어지는지만 다르다 (PREREG_poison.md v2).

  D*: "Wittenberg is located in Lower Saxony."   X = 사슬의 bridge (마지막 단계 질문이 #k 로 가리키는 답, G2 가 다루는 주어)
  A*: "Halle is located in Lower Saxony."        Y = X 와 같은 세부 종류, 질문 사슬과 무관, 'Y + 관계 + Q' 가 그럴듯하지만 거짓
  Q* = 같은 호출에서 받은 객관식 오답 보기 (정답과 같은 종류·형식) 중 첫 통과.  D 의 sub 는 'John Cabot was the father of Christopher Columbus' 처럼
       터무니없는 것이 많아 A*/D* 에는 쓰지 않는다 (sub 로 만든 문장 S 는 검증에서 실제로 참인지만 보고 D 분석의 제외에 씀).
  A* = D* 문장에서 X 를 Y 로 바꾼 것 (문장 형식 동일).  문서 제목 = 주어.
Y 를 다른 MuSiQue 질의의 bridge 로 빌려 오면 종류는 같아도 'Myanmar was led by Henk Kamp' 처럼 터무니없는 A* 가 되어 (D* 는 그럴듯함)
그럴듯함이 섞이므로, gpt-5-mini (reasoning low) 가 마지막 단계의 참 문장과 Y 후보 5개 (X 와 같은 종류·나라·지역·시대, 사슬과 무관), 각 Y 의 실제 값을 낸다.
  거짓 문장은 LLM 에 요청하지 않는다 (요청하면 실존 인물·장소에 대한 거짓이라며 거절: medium 에서 191건).  D* = 참 문장의 정답을 Q 로, A* = D* 의 X 를 Y 로 코드에서 바꿈.
  (minimal 로 거짓 문장까지 만든 첫 판은 270쌍 직접 검수에서 절반 가까이가 'Auckland shares a border with Mecklenburg County' 같은 A*, 참인 D*, 터무니없는 Q 라 버림)
Y 조건: X 와 단어 안 겹침, 이 질의 문맥 (근거 + hard + rand) 에 없음, 질문·사슬의 대문자 단어 (일반어 제외)·답과 겹치지 않음,
        실제 값이 있고 Q* 와 F1 0.5 이하·같지 않음 (A*, D* 모두 거짓 주장).  후보 순서대로 첫 번째.
문장 조건: 참 문장에 X 와 정답이 정확히 한 번씩.  Y 는 Q 와도 단어 안 겹침.
대상: D 가능 & 파일럿 아님 & 마지막 단계가 앞 단계 하나만 가리킴.
검증 (별도 호출, CHECK): D* / A* 각각 참·거짓·불확실, 비전문가 기준 그럴듯함 1-5, A* 주어가 사슬과 이어지는지.
자동 기준 (v2.1): D*, A* 둘 다 실제로 참이 아님, A* 주어 무관.  그럴듯함은 공변량 (검증 모델이 사실을 알면 D* 도 1-2 점이라 걸러내는 데 쓰지 않음).
통과분은 전부 직접 검수 (data/star_review.md -> data/star_exclude.json).  D* 가 참 (Q 가 실제 정답) 이면 D 부분 분석에서도 뺀다.
문맥: D 부분 본 실행과 같은 슬롯 (poison 자리) 과 G2 맞바꿈 -> 조건 As, As_sw, Ds, Ds_sw (gold / R / R_sw 는 D 부분 실행 결과를 함께 씀).
실행: /NAS/minyeol/limit/.venv-promptriever/bin/python build_star.py
출력: data/star_llm.json, data/star_check.json, data/star_base.jsonl, data/star_review.md, data/musique_poison_star.jsonl (star_exclude.json 이 있을 때)
"""
import json, os, re
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from grade import norm, f1
from sub_llm import key

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = f"{HERE}/data"; MODEL = "gpt-5-mini"
GENERIC = set(("what when where which whose who there this that along with from into country city county state states united university river north "
               "south east west new saint lake mount island islands national international republic kingdom province region district church "
               "cathedral basilica award awards party league company records film song album series season game games world team club "
               "school college station network council government president governor king queen prince emperor empire army battle").split())
MAKE = ('A multi-hop question and its true reasoning chain:\nQuestion: {question}\n{chain}\n\n'
        'Final step: "{last}" (true answer: {true}).\n'
        '1. Write ONE short, plain, TRUE encyclopedic sentence that answers the final step. It must contain "{x}" exactly once and "{true}" exactly once, '
        'and nothing else specific to this question.\n'
        '2. List 5 other real entities of exactly the same specific kind as "{x}", from the same country, region and era as "{x}" where that applies '
        '(for example another county in the same state, another actor of the same generation), that do not appear in and are not related to the question or its chain. '
        'For each, give its true answer to the same final step.\n'
        '3. For a multiple-choice version of the final step, list 5 plausible but incorrect answer options of the same kind and format as "{true}" '
        '(for example a nearby year for a year, another county in the same state for a county, another actress of the same era for an actress).\n'
        'Reply as JSON only: {{"sentence": "...", "subjects": [{{"name": "...", "real_value": "..."}}, ...], "distractors": ["...", ...]}}')
CHECK = ('Question: {question}\nTrue reasoning chain:\n{chain}\n\nStatement D: {d}\nStatement A: {a}\nStatement S: {s}\n\n'
         'For each statement, say whether it is TRUE, FALSE or UNSURE in the real world, and rate how believable it would sound to an educated non-expert '
         '(1 = absurd, 5 = completely believable). Also say whether the subject of Statement A ("{y}") appears in, or is related to, the question or its chain '
         '(YES or NO). Reply as JSON only: {{"D_truth": "...", "D_believable": n, "A_truth": "...", "A_believable": n, "A_related": "...", '
         '"S_truth": "...", "S_believable": n}}')


def chain_text(r):
    out, ans = [], []
    for i, (q, a) in enumerate(r["chain"]):
        out.append(f"{i + 1}. {re.sub(r'#(\d)', lambda m: ans[int(m.group(1)) - 1], q)} -> {a}"); ans.append(a)
    return "\n".join(out)


def last_step(r):
    ans = [a for _, a in r["chain"]]
    return re.sub(r"#(\d)", lambda m: ans[int(m.group(1)) - 1], r["chain"][-1][0])


def llm(prompts, effort="low"):
    client = OpenAI(api_key=key())
    def ask(p):
        try: return client.responses.create(model=MODEL, reasoning={"effort": effort}, input=p).output_text.strip()
        except Exception as e: return f"ERR {type(e).__name__}"
    with ThreadPoolExecutor(16) as ex: return list(ex.map(ask, prompts))


def connected(y, r, ctx_norm):
    """Y 가 이 질의와 이어지는 문자열 근거 (없으면 None)."""
    if f" {norm(y)} " in ctx_norm: return "문맥에 있음"
    texts = [r["question"]] + [q for q, _ in r["chain"]]
    toks = {re.sub(r"'s$", "", w.strip("?,.\"():;!")) for t in texts + [a for _, a in r["chain"]] for w in t.split()}
    toks = {w for w in toks if len(w) >= 4 and w[:1].isupper() and w.lower() not in GENERIC}
    hit = [w for w in toks if re.search(r"(?<!\w)" + re.escape(w) + r"\w*", y, re.I)]
    if hit: return "사슬 단어 " + ",".join(sorted(hit))
    if any(f1(y, a) > 0 for _, a in r["chain"]) or any(f1(y, a) > 0 for a in r["answers"]): return "사슬 답과 겹침"
    return None


def count(s, t): return len(re.findall(r"(?<!\w)" + re.escape(s) + r"(?!\w)", t))


def main():
    rows = [json.loads(l) for l in open(f"{OUT}/poison_base.jsonl")]
    anc = [json.loads(l) for l in open(f"{OUT}/musique_anchor.jsonl")]
    pilot = set(json.load(open(f"{OUT}/poison_pilot_ids.json")))
    cand = [r for r in rows if r["ok_D"] and r["qid"] not in pilot and r["bridge"] and len(re.findall(r"#(\d)", r["chain"][-1][0])) == 1]

    fl = f"{OUT}/star_llm.json"; L = json.load(open(fl)) if os.path.exists(fl) else {}
    todo = [r for r in cand if r["qid"] not in L]
    outs = llm([MAKE.format(question=r["question"], chain=chain_text(r), last=last_step(r), true=r["chain"][-1][1], q=r["sub"], x=r["bridge"]) for r in todo])
    L.update({r["qid"]: o for r, o in zip(todo, outs)}); json.dump(L, open(fl, "w"), ensure_ascii=False, indent=0)

    star, why = [], {}
    for r in cand:
        try: o = json.loads(re.sub(r"^```(json)?|```$", "", L[r["qid"]].strip()).strip())
        except Exception: why[r["qid"]] = "JSON 아님"; continue
        if not isinstance(o, dict): why[r["qid"]] = "JSON 아님"; continue
        x, q, sent = r["bridge"], r["sub"], o.get("sentence") or ""
        t = r["chain"][-1][1]
        if count(x, sent) != 1 or count(t, sent) != 1: why[r["qid"]] = "참 문장에 X / 정답이 한 번씩이 아님"; continue
        a = anc[r["qi"]]
        ctx_norm = " " + " ".join(norm(a["docs"][str(i)]["title"] + " " + a["docs"][str(i)]["text"]) for i in a["gold"] + a["hard"] + a["rand"]) + " "
        swap = lambda new: re.sub(r"(?<!\w)" + re.escape(t) + r"(?!\w)", new.replace("\\", r"\\"), sent)
        s_sub = swap(r["sub"])                                                         # D 부분의 sub 로 만든 문장 (sub 가 실제로 참인지 확인용)
        q = next((str(d).strip() for d in o.get("distractors") or [] if str(d).strip() and all(f1(str(d), aa) <= .5 for aa in r["answers"])
                  and f1(str(d), x) == 0 and f" {norm(str(d))} " not in ctx_norm and norm(str(d))), None)
        if not q: why[r["qid"]] = "Q* 후보 모두 탈락"; continue
        sent = swap(q)                                                                 # 참 문장의 정답 -> Q*
        pick = None
        for s in o.get("subjects") or []:
            if not isinstance(s, dict): continue
            y, yv = str(s.get("name", "")).strip(), str(s.get("real_value", "")).strip()
            if not y or f1(y, x) > 0 or count(y, sent) or not yv or f1(yv, q) > .5 or norm(yv) == norm(q) or f1(y, q) > 0 or connected(y, r, ctx_norm): continue
            pick = (y, yv); break
        if not pick: why[r["qid"]] = "Y 후보 모두 탈락"; continue
        y, yv = pick
        Ds = dict(title=x, text=sent); As = dict(title=y, text=re.sub(r"(?<!\w)" + re.escape(x) + r"(?!\w)", y.replace("\\", r"\\"), sent))
        star.append(dict(r=r, Y=y, Y_true=yv, Q=q, As=As, Ds=Ds, S=s_sub))
    print(f"후보 {len(cand)} -> A*/D* {len(star)}  탈락: {dict((v, list(why.values()).count(v)) for v in set(why.values()))}")

    fc = f"{OUT}/star_check.json"; C = json.load(open(fc)) if os.path.exists(fc) else {}
    sig = lambda s: s["Ds"]["text"] + " || " + s["As"]["text"] + " || " + s["S"]
    todo = [s for s in star if C.get(s["r"]["qid"], {}).get("sig") != sig(s)]
    res = llm([CHECK.format(question=s["r"]["question"], chain=chain_text(s["r"]), d=s["Ds"]["text"], a=s["As"]["text"], s=s["S"], y=s["Y"]) for s in todo])
    for s, v in zip(todo, res):
        try: c = json.loads(re.sub(r"^```(json)?|```$", "", v.strip()).strip())
        except Exception: c = {}
        C[s["r"]["qid"]] = dict(c, sig=sig(s))
    json.dump(C, open(fc, "w"), ensure_ascii=False, indent=0)

    def auto(s):
        """자동 기준 (PREREG_poison.md v2.1): D*, A* 가 실제로 참이 아님 (FALSE 또는 UNSURE), A* 주어가 사슬과 무관.  그럴듯함은 공변량으로만 기록."""
        c = C[s["r"]["qid"]]
        if not c: return "검증 응답 없음"
        if str(c.get("D_truth", "")).upper() == "TRUE": return "D* 가 참"
        if str(c.get("A_truth", "")).upper() == "TRUE": return "A* 가 참"
        if str(c.get("A_related", "")).upper() != "NO": return "A* 주어가 사슬과 이어짐"
        return None
    for s in star: s["auto"] = auto(s)
    av = [s["auto"] for s in star]
    print(f"자동 기준 통과 {av.count(None)} / {len(star)}  탈락 상위: "
          f"{sorted(((re.sub(r'[0-9]', 'n', v), sum(re.sub(r'[0-9]', 'n', w or '') == re.sub(r'[0-9]', 'n', v) for w in av)) for v in set(av) if v), key=lambda x: -x[1])[:6]}")

    with open(f"{OUT}/star_base.jsonl", "w") as f:
        for s in star:
            r = s["r"]
            f.write(json.dumps(dict(qid=r["qid"], qi=r["qi"], question=r["question"], answers=r["answers"], chain=r["chain"], X=r["bridge"], Y=s["Y"],
                                    Y_true=s["Y_true"], Q=s["Q"], sub=r["sub"], S=s["S"], g2_id=r["g2_id"], check={k: v for k, v in C[r["qid"]].items() if k != "sig"},
                                    auto=s["auto"], As=s["As"], Ds=s["Ds"]), ensure_ascii=False) + "\n")
    with open(f"{OUT}/star_review.md", "w") as f:
        f.write("# A* / D* 직접 검수 (자동 기준 통과분)\n\n")
        for i, s in enumerate(star):
            if s["auto"]: continue
            r = s["r"]; c = C[r["qid"]]
            f.write(f"[{i}] {r['qid']} | 마지막 단계: {last_step(r)} -> {r['chain'][-1][1]}  (그럴듯함 D {c.get('D_believable')} / A {c.get('A_believable')})\n"
                    f"   D*: {s['Ds']['text']}\n   A*: {s['As']['text']}  (실제: {s['Y_true']})\n")

    fx = f"{OUT}/star_exclude.json"
    if not os.path.exists(fx): print("data/star_exclude.json 없음: 검수 뒤 다시 실행"); return
    X = json.load(open(fx)); D = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{OUT}/musique_poison_mainD.jsonl")}
    keep = [s for s in star if not s["auto"] and s["r"]["qid"] not in X]
    with open(f"{OUT}/musique_poison_star.jsonl", "w") as f:
        for s in keep:
            m = D[s["r"]["qid"]]; base = m["ctx"]["R"]; k, g = m["slot"], m["g2_pos"]
            def ctx(new, swap):
                c = list(base); c[k] = new
                if swap: c[k], c[g] = c[g], c[k]
                return c
            docs = {kk: v for kk, v in m["docs"].items() if kk != "pD"}; docs["pAs"], docs["pDs"] = s["As"], s["Ds"]
            f.write(json.dumps(dict(qid=m["qid"], qi=m["qi"], question=m["question"], answers=m["answers"], sub=s["Q"], X=s["r"]["bridge"], Y=s["Y"],
                                    g2_id=m["g2_id"], gold=m["gold"], poison_before_g2=m["poison_before_g2"],
                                    ctx=dict(As=ctx("pAs", False), As_sw=ctx("pAs", True), Ds=ctx("pDs", False), Ds_sw=ctx("pDs", True)), docs=docs),
                               ensure_ascii=False) + "\n")
    print(f"본 실행 A*/D*: {len(keep)} 질의 (자동 탈락 {sum(bool(s['auto']) for s in star)}, 검수 제외 {len(X)}) -> {OUT}/musique_poison_star.jsonl")


if __name__ == "__main__":
    main()
