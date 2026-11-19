"""기준 세트: MuSiQue 1,000 질의 전부, 근거 문서는 항상 전부 넣는다 (근거 완비를 구성으로 보장).

질의마다 문맥 조건 (문서 id 순서 리스트, closed 는 None):
  gold   근거 문서만, 질의 번호 seed 로 섞은 순서
  hard3  gold + LACE 점수 상위 비근거 문서 3개 (검색이 올린 어려운 방해 문서)
  rand3  gold + 코퍼스에서 무작위로 뽑은 비근거 문서 3개 (쉬운 노이즈 대조)
  hard3 / rand3 은 같은 seed 로 같은 위치에 끼워 넣는다 (근거끼리의 순서 유지): 두 조건의 차이는 방해 문서 내용뿐.
분석용으로 질문 분해의 중간 단계 답 (bridges) 도 저장.
LACE 점수 = build_inputs.lace_scores (새 구성, 8 seed 평균).
실행:  /NAS/minyeol/limit/.venv-promptriever/bin/python build_anchor.py     출력: data/musique_anchor.jsonl
"""
import ast, json, os
import numpy as np
from build_inputs import lace_scores, F, H, OUT


def main():
    ev, score = lace_scores()
    qa = json.load(open(f"{H}/musique.json")); corpus = json.load(open(f"{H}/musique_corpus.json"))
    z = np.load("/NAS/minyeol/Graph_retrieval/for_WWW/cache/mhqa_musique.npz", allow_pickle=True)
    go, gf = z["gold_offsets"], z["gold_flat"]; gold = [gf[a:b].tolist() for a, b in zip(go[:-1], go[1:])]
    assert all(a["question"] == str(b) for a, b in zip(qa, z["questions"])), "질의 순서 불일치"
    rows = []
    for q, ix in enumerate(F.spans(ev["g"])):
        rng = np.random.default_rng(q); g = [int(d) for d in rng.permutation(gold[q])]
        ranked = ev["doc"][ix[np.argsort(-score[ix], kind="stable")]]
        hard = [int(d) for d in ranked if int(d) not in g][:3]
        pool = np.setdiff1d(np.arange(len(corpus)), g); rand = [int(d) for d in np.random.default_rng(10_000 + q).choice(pool, 3, replace=False)]
        pos = np.random.default_rng(20_000 + q)
        slots = [int(pos.integers(0, len(g) + i + 1)) for i in range(3)]       # 끼워 넣을 위치 (hard3 / rand3 공통)
        def insert(ds):
            c = list(g)
            for s, d in zip(slots, ds): c.insert(s, d)
            return c
        dec = qa[q]["question_decomposition"]; dec = ast.literal_eval(dec) if isinstance(dec, str) else dec
        al = qa[q]["answer_aliases"]; al = ast.literal_eval(al) if isinstance(al, str) else al
        ctx = dict(closed=None, gold=g, hard3=insert(hard), rand3=insert(rand))
        rows.append(dict(qid=qa[q]["id"], qi=q, question=qa[q]["question"], answers=[qa[q]["answer"]] + list(al),
                         bridges=[d["answer"] for d in dec[:-1]], gold=g, hard=hard, rand=rand, ctx=ctx,
                         docs={str(d): dict(title=corpus[d]["title"], text=corpus[d]["text"]) for d in set(g + hard + rand)}))
    with open(f"{OUT}/musique_anchor.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} 질의 -> {OUT}/musique_anchor.jsonl   근거 수 {dict(zip(*np.unique([len(r['gold']) for r in rows], return_counts=True)))}")


if __name__ == "__main__":
    main()
