"""바꿔 넣을 개체 후보 (LLM): data/poison_base.jsonl 의 G2 와 대상 개체 (정답 언급, G2 안의 bridge) 마다 같은 세부 종류의 다른 실제 개체 5개.

NER 이름표 (GPE, ORG ...) 는 county / 국가, 대학 / 음반사를 가르지 못해서, 문서 맥락을 보고 세부 종류를 맞춘 후보를 받는다.
날짜·수는 build_poison.py 가 규칙으로 바꾸고, 규칙으로 못 바꾸는 경우 (예: 'spring') 에만 이 후보를 쓴다.
모델 gpt-5-mini (reasoning effort minimal).  후보의 제약 확인 (문맥에 없음, 정답과 단어 안 겹침) 은 build_poison.py 가 한다.
이미 받은 키는 건너뜀 (중단 후 다시 실행 가능).  API 키 = ../.env 의 OPENAI_API_KEY.
실행: /NAS/minyeol/limit/.venv-promptriever/bin/python sub_llm.py [--limit N]     출력: data/sub_llm.json  {"<qid>|ans": [...], "<qid>|bridge": [...]}
"""
import argparse, json, os, re
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = f"{HERE}/data"; MODEL = "gpt-5-mini"
ASK = ('Passage:\n{title}\n{text}\n\nThe passage mentions "{ent}". List 5 different real-world entities of exactly the same specific type as "{ent}" '
       '(for example a county for a county, a university for a university, a footballer for a footballer, a TV network for a TV network) '
       'that could replace "{ent}" in this passage and still read naturally. Do not use "{ent}" or anything else mentioned in the passage. '
       'Write one name per line, names only, in the same style as "{ent}".')


def key():
    for l in open(f"{HERE}/../.env"):
        if l.startswith("OPENAI_API_KEY="): return l.split("=", 1)[1].strip().strip('"').strip("'")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
    rows = [json.loads(l) for l in open(f"{OUT}/poison_base.jsonl")]
    jobs = [(f"{r['qid']}|ans", r["g2"], r["ans_mention"]) for r in rows if r["ans_mention"]]
    jobs += [(f"{r['qid']}|bridge", r["g2"], r["bridge"]) for r in rows if r["bridge_in_g2"]]
    f = f"{OUT}/sub_llm.json"; out = json.load(open(f)) if os.path.exists(f) else {}
    jobs = [j for j in jobs if j[0] not in out][: a.limit or None]
    client = OpenAI(api_key=key()); use = [0, 0]

    def ask(j):
        k, g, e = j
        try: r = client.responses.create(model=MODEL, reasoning={"effort": "minimal"}, input=ASK.format(title=g["title"], text=g["text"], ent=e))
        except Exception as err: print(f"  {k}: {type(err).__name__} -> 후보 없음", flush=True); return k, []      # 정책 필터 등
        use[0] += r.usage.input_tokens; use[1] += r.usage.output_tokens
        return k, [re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", s).strip().strip('"') for s in r.output_text.split("\n") if s.strip()][:5]

    with ThreadPoolExecutor(16) as ex:
        for i, (k, c) in enumerate(ex.map(ask, jobs)):
            out[k] = c
            if i % 200 == 0: json.dump(out, open(f, "w"), ensure_ascii=False); print(f"{i + 1}/{len(jobs)}  tokens in {use[0]} out {use[1]}", flush=True)
    json.dump(out, open(f, "w"), ensure_ascii=False)
    print(f"done {len(jobs)}  tokens in {use[0]} out {use[1]}  (~${use[0] * .25e-6 + use[1] * 2e-6:.2f})")


if __name__ == "__main__":
    main()
