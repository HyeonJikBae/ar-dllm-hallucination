"""MuSiQue 에서 검색 top-5 에 정답 근거가 모두 들어온 질의만 골라 생성 입력을 만든다.

검색 = LACE (새 구성 8 피처, 8 seed 평균 점수; Graph_retrieval/for_WWW/feedback/t20_newconfig.py 의 T1 과 같은 모델).
질의마다 저장: 질문, 정답 + 별칭, 근거 문서 id, top-5 문서 id (검색 순위), 방해 문서 id (= top-5 - 근거), 문서 본문.
few-shot 예시 2개는 MuSiQue train 에서 고정 (근거 2 + 방해 1, 순서 고정).
실행 (fbcore 가 있는 환경):  /NAS/minyeol/limit/.venv-promptriever/bin/python build_inputs.py
출력: data/musique_full5.jsonl, data/fewshot.json
"""
import json, os, sys, ast
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = f"{HERE}/data"; os.makedirs(OUT, exist_ok=True)
FB = "/NAS/minyeol/Graph_retrieval/for_WWW/feedback"; sys.path.insert(0, FB)
import fbcore as F
from t18_direct_damping import named

H = "/NAS/minyeol/hippoRAG/reproduce/dataset"
COLS = ["s_D", "lr_D", "s_T", "lr_T", "rho", "l1_0.99", "cons_0.5", "imb_0.5"]


def lace_scores():
    tr, ev = F.load("musique"); ct, ce = named("musique", tr, "tr"), named("musique", ev, "ev")
    M = lambda c: np.column_stack([c[k] for k in COLS]).astype(np.float32)
    m = F.fit(M(ct), tr["y"], tr["g"], ns=8)
    r = F.evaluate(m, M(ce), ev["y"], ev["g"], ev["gsz"])
    print(f"LACE R@5 (8 seed) {r['r']:.2f}  (t20 T1 new 74.42)", flush=True)
    return ev, r["score"]


def main():
    ev, score = lace_scores()
    qa = json.load(open(f"{H}/musique.json")); corpus = json.load(open(f"{H}/musique_corpus.json"))
    z = np.load("/NAS/minyeol/Graph_retrieval/for_WWW/cache/mhqa_musique.npz", allow_pickle=True)
    qs = [str(x) for x in z["questions"]]; go, gf = z["gold_offsets"], z["gold_flat"]
    gold = [gf[a:b].tolist() for a, b in zip(go[:-1], go[1:])]
    assert len(qa) == len(qs) and all(a["question"] == b for a, b in zip(qa, qs)), "질의 순서 불일치"
    rows, n_full = [], 0
    for q, ix in enumerate(F.spans(ev["g"])):
        top = ix[np.argsort(-score[ix], kind="stable")[:5]]
        top5 = [int(d) for d in ev["doc"][top]]
        assert set(ev["doc"][ix[ev["y"][ix] == 1]].tolist()) <= set(gold[q]), q          # 후보 밖 근거가 있을 수 있음
        if not set(gold[q]) <= set(top5): continue
        n_full += 1
        al = qa[q]["answer_aliases"]; al = ast.literal_eval(al) if isinstance(al, str) else al
        rows.append(dict(qid=qa[q]["id"], qi=q, question=qa[q]["question"], answers=[qa[q]["answer"]] + list(al),
                         gold=[d for d in top5 if d in gold[q]], top5=top5, distractors=[d for d in top5 if d not in gold[q]],
                         docs={str(d): dict(title=corpus[d]["title"], text=corpus[d]["text"]) for d in top5}))
    with open(f"{OUT}/musique_full5.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ng = np.bincount([len(r["gold"]) for r in rows])
    print(f"top-5 에 근거 전부: {n_full} / {len(qa)}   근거 수 분포 {dict(enumerate(ng.tolist()))}", flush=True)

    # few-shot: train 에서 2-hop, 짧은 정답, 근거 2 + 방해 1
    shots, rng = [], np.random.default_rng(0)
    for line in open("/NAS/minyeol/Graph_retrieval/for_WWW/rawmusique/musique_train.jsonl"):
        d = json.loads(line)
        if not d.get("answerable", True) or not d["id"].startswith("2hop") or len(d["answer"].split()) > 3: continue
        sup = [p for p in d["paragraphs"] if p["is_supporting"]]; neg = [p for p in d["paragraphs"] if not p["is_supporting"]]
        if len(sup) != 2 or any(len(p["paragraph_text"].split()) > 90 for p in sup + neg[:1]): continue
        ps = [sup[0], neg[0], sup[1]]
        shots.append(dict(question=d["question"], answer=d["answer"], docs=[dict(title=p["title"], text=p["paragraph_text"]) for p in ps]))
        if len(shots) == 2: break
    json.dump(shots, open(f"{OUT}/fewshot.json", "w"), ensure_ascii=False, indent=1)
    print("few-shot:", [(s["question"], s["answer"]) for s in shots])


if __name__ == "__main__":
    main()
