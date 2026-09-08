"""PopQA gold answer가 Wikidata 상 유일한 정답이 아닐 수 있는 케이스를 찾아낸다.

각 fact의 subject 엔티티를 Wikidata에서 다시 조회해서, PopQA가 gold로 고른
relation(Property)에 실제로 값이 몇 개 등록되어 있는지 확인한다. 값이 2개 이상이면
(deprecated rank 제외) gold 외에 다른 정답도 인정될 수 있는 후보로 표시한다.
"""

import argparse
import json
import os
import time

import requests

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
HEADERS = {"User-Agent": "popqa-audit/0.1 (research use)"}

# PopQA 논문(Mallen et al. 2023)에서 사용하는 16개 relation과 실제 Wikidata Property ID 매핑.
RELATION_TO_PID = {
    "occupation": "P106",
    "place of birth": "P19",
    "genre": "P136",
    "father": "P22",
    "mother": "P25",
    "country": "P17",
    "producer": "P162",
    "director": "P57",
    "capital of": "P1376",
    "screenwriter": "P58",
    "composer": "P86",
    "color": "P462",
    "religion": "P140",
    "sport": "P641",
    "author": "P50",
    "capital": "P36",
}

BATCH_SIZE = 50


def qid_from_uri(uri: str) -> str:
    return uri.rstrip("/").split("/")[-1]


def load_raw(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def request_with_retry(params: dict, max_retries: int = 6) -> dict:
    delay = 2.0
    for attempt in range(max_retries):
        resp = requests.get(WIKIDATA_API, params=params, headers=HEADERS, timeout=30)
        if resp.status_code == 429:
            wait = float(resp.headers.get("Retry-After", delay))
            print(f"  429 rate limited, sleeping {wait}s...")
            time.sleep(wait)
            delay *= 2
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()
    return resp.json()


def fetch_claims_batch(qids: list[str]) -> dict:
    data = request_with_retry(
        {
            "action": "wbgetentities",
            "ids": "|".join(qids),
            "props": "claims",
            "format": "json",
        }
    )
    return data.get("entities", {})


def fetch_labels_batch(qids: list[str]) -> dict:
    labels = {}
    for i in range(0, len(qids), BATCH_SIZE):
        chunk = qids[i : i + BATCH_SIZE]
        data = request_with_retry(
            {
                "action": "wbgetentities",
                "ids": "|".join(chunk),
                "props": "labels",
                "languages": "en",
                "format": "json",
            }
        )
        entities = data.get("entities", {})
        for qid, ent in entities.items():
            label = ent.get("labels", {}).get("en", {}).get("value")
            labels[qid] = label or qid
        time.sleep(0.5)
    return labels


def extract_values(claims: dict, pid: str) -> list[str]:
    """해당 property의 값 QID 목록을 반환한다 (deprecated rank 제외, 중복 제거, 순서 보존)."""
    seen = set()
    values = []
    for claim in claims.get(pid, []):
        if claim.get("rank") == "deprecated":
            continue
        mainsnak = claim.get("mainsnak", {})
        if mainsnak.get("snaktype") != "value":
            continue
        datavalue = mainsnak.get("datavalue", {})
        if datavalue.get("type") != "wikibase-entityid":
            continue
        qid = datavalue["value"]["id"]
        if qid not in seen:
            seen.add(qid)
            values.append(qid)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/popqa_raw.jsonl")
    parser.add_argument(
        "--output", default="analysis/multi_answer_candidates.jsonl"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="디버그용: 앞에서부터 N개 row만 처리"
    )
    parser.add_argument(
        "--claims-cache",
        default="analysis/.wikidata_claims_cache.jsonl",
        help="엔티티별 claims를 캐시해서 중단 시 이어서 진행할 수 있게 한다",
    )
    args = parser.parse_args()

    rows = load_raw(args.input)
    if args.limit:
        rows = rows[: args.limit]

    unknown_relations = {r["prop"] for r in rows} - set(RELATION_TO_PID)
    if unknown_relations:
        raise SystemExit(f"매핑되지 않은 relation 발견: {unknown_relations}")

    unique_qids = sorted({qid_from_uri(r["s_uri"]) for r in rows})
    print(f"rows={len(rows)} unique_subjects={len(unique_qids)}")

    entity_claims: dict[str, dict] = {}
    if os.path.exists(args.claims_cache):
        with open(args.claims_cache, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                entity_claims[obj["qid"]] = obj["claims"]
        print(f"loaded {len(entity_claims)} cached entities from {args.claims_cache}")

    todo_qids = [q for q in unique_qids if q not in entity_claims]
    os.makedirs(os.path.dirname(args.claims_cache), exist_ok=True)
    with open(args.claims_cache, "a", encoding="utf-8") as cache_f:
        for i in range(0, len(todo_qids), BATCH_SIZE):
            chunk = todo_qids[i : i + BATCH_SIZE]
            entities = fetch_claims_batch(chunk)
            for qid in chunk:
                claims = entities.get(qid, {}).get("claims", {})
                entity_claims[qid] = claims
                cache_f.write(json.dumps({"qid": qid, "claims": claims}, ensure_ascii=False) + "\n")
            cache_f.flush()
            print(f"  fetched claims {i + len(chunk)}/{len(todo_qids)}", end="\r")
            time.sleep(0.5)
    print()

    candidates = []
    for row in rows:
        pid = RELATION_TO_PID[row["prop"]]
        subj_qid = qid_from_uri(row["s_uri"])
        claims = entity_claims.get(subj_qid, {})
        values = extract_values(claims, pid)
        gold_qid = qid_from_uri(row["o_uri"])

        if len(values) <= 1:
            continue

        candidates.append(
            {
                "sample_id": str(row["id"]),
                "subject": row["subj"],
                "subject_qid": subj_qid,
                "relation": row["prop"],
                "property_id": pid,
                "question": row["question"],
                "gold_object": row["obj"],
                "gold_qid": gold_qid,
                "gold_in_current_values": gold_qid in values,
                "all_value_qids": values,
                "num_values": len(values),
            }
        )

    print(f"multi-value candidates: {len(candidates)} / {len(rows)}")

    alt_qids = sorted(
        {qid for c in candidates for qid in c["all_value_qids"]}
    )
    print(f"resolving labels for {len(alt_qids)} qids...")
    labels = fetch_labels_batch(alt_qids)

    for c in candidates:
        alt_labels = [labels.get(q, q) for q in c["all_value_qids"]]
        c["all_value_labels"] = alt_labels
        others = [
            lbl
            for q, lbl in zip(c["all_value_qids"], alt_labels)
            if q != c["gold_qid"]
        ]
        c["reason"] = (
            f"Wikidata에 {c['subject']}의 {c['relation']}(P{c['property_id'][1:]}) 값이 "
            f"{c['num_values']}개 등록되어 있음: {', '.join(alt_labels)}. "
            f"PopQA는 '{c['gold_object']}'만 gold로 선택했지만 "
            f"{', '.join(others)}{'도' if len(others) == 1 else '도'} 정답으로 인정될 수 있음."
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"saved {len(candidates)} candidates to {args.output}")


if __name__ == "__main__":
    main()
