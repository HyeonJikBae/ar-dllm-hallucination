"""patch_recover.py 채점 (PREREG_poison.md v9).  usage: python grade_recover.py     출력: results/summary_recover.json"""
import json, os
import numpy as np
from grade_bindswap import boot_items

HERE = os.path.dirname(os.path.abspath(__file__)); RES = f"{HERE}/results"; CATS = ["복구", "지속", "복사", "주는쪽다른", "기타"]


def main():
    R = [json.loads(l) for l in open(f"{RES}/recover_qwen.jsonl")]
    err = [r for r in R if r["recv_err"]]; dn = [d for r in err for d in r["donors"]]
    use = [r for r in err if any(d["ok"] for d in r["donors"])]
    S = dict(n_cases=len(R), n_recv_err=len(err), donors=len(dn), donor_ok=round(100 * np.mean([d["ok"] for d in dn]), 1),
             cases_with_ok_donor=len(use), donor_ok_by_cond={c: round(100 * np.mean([d["ok"] for r in err if r["cond"] == c for d in r["donors"]]), 1)
                                                            for c in ("U", "U_sw", "L", "L_sw")})
    print(json.dumps(S, ensure_ascii=False))
    for site in ("last", "qall"):
        A = []
        for r in use:
            a = np.zeros((29, len(CATS))); ds = [d for d in r["donors"] if d["ok"]]
            for d in ds:
                m = {r["ans"]: 0, r["oth"]: 1, d["ans"]: 2, d["oth"]: 3}
                for l, t in enumerate(d[site]): a[l, m.get(t, 4)] += 1 / len(ds)
            A.append(a)
        A = np.stack(A); S[site] = (A.mean(0) * 100).round(1).tolist()
        best = int(A[:, :, 0].mean(0).argmax())
        S[f"{site}_best"] = dict(layer=best - 1, 복구=boot_items(lambda ix: A[ix, best, 0].mean(), len(A)),
                                 복사=boot_items(lambda ix: A[ix, best, 2].mean(), len(A)))
        print(f"\n{site} (층: " + "  ".join(CATS) + ")")
        for l, row in enumerate(A.mean(0)): print(f"  {'emb' if l == 0 else l - 1:>3} " + " ".join(f"{100 * v:5.1f}" for v in row))
        print(" 최대 복구 층:", S[f"{site}_best"])
    json.dump(S, open(f"{RES}/summary_recover.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
