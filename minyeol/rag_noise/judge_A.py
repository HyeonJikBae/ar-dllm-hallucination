"""A (무관형 poison) 의 의미 연결 판정: 문자열 필터 (build_poison.py leak_A) 를 통과한 A 문서가 다른 이름·번역·설명으로라도 질문 사슬의 개체를 다루는가.

모델 gpt-5-mini (reasoning effort minimal), 답 YES / NO.  YES 면 본 실행에서 뺀다.  공격 결과를 보기 전에 실행하고 고정한다 (PREREG_poison.md).
실행: /NAS/minyeol/limit/.venv-promptriever/bin/python judge_A.py     출력: data/judge_A.json  {qid: "YES" | "NO" | "ERR"}
"""
import json, os, re
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from sub_llm import key

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = f"{HERE}/data"; MODEL = "gpt-5-mini"
ASK = ("Question: {q}\nReasoning chain (true facts):\n{chain}\n\nPassage:\n{title}\n{text}\n\n"
       "Does the passage mention, describe, or give information about any entity in this question or its reasoning chain, "
       "including under another name, abbreviation, translation, or an identifying description? "
       "A passage that only states a similar kind of fact about a different, unrelated entity does NOT count. "
       "Answer with exactly one word: YES or NO.")


def main():
    rows = [r for r in map(json.loads, open(f"{OUT}/poison_base.jsonl")) if r["ok_A"] and not r["leak_A"]]
    client = OpenAI(api_key=key())

    def chain(r):
        out, ans = [], []
        for i, (q, a) in enumerate(r["chain"]):
            out.append(f"{i + 1}. {re.sub(r'#(\d)', lambda m: ans[int(m.group(1)) - 1], q)} -> {a}"); ans.append(a)
        return "\n".join(out)

    def ask(r):
        try:
            x = client.responses.create(model=MODEL, reasoning={"effort": "minimal"},
                                        input=ASK.format(q=r["question"], chain=chain(r), title=r["A"]["title"], text=r["A"]["text"]))
            v = x.output_text.strip().upper()
            return r["qid"], "YES" if v.startswith("YES") else "NO" if v.startswith("NO") else "ERR"
        except Exception as e: print(f"  {r['qid']}: {type(e).__name__}", flush=True); return r["qid"], "ERR"

    with ThreadPoolExecutor(16) as ex: J = dict(ex.map(ask, rows))
    json.dump(J, open(f"{OUT}/judge_A.json", "w"), indent=0)
    print({v: sum(x == v for x in J.values()) for v in ("YES", "NO", "ERR")})


if __name__ == "__main__":
    main()
