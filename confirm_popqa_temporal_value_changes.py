"""PopQA의 시점 위험 relation(capital, capital of, color, religion, country)에서
실제로 값이 바뀐 이력이 있는 문항만 정밀하게 제외한다.

기준: 해당 subject-property 조합에
  - deprecated rank로 남은 값이 있으면(예전 값이 superseded 되어 남은 흔적) 실제 변경 이력 있음
  - 또는 normal/preferred rank 값에 end-time(P582) qualifier가 있으면(그 값이 특정 시점까지만
    유효했다는 뜻이므로 "현재"를 묻는 질문에는 부적합할 수 있음) 실제 변경 이력 있음
이 두 경우에만 제외하고, 그 외(값이 하나뿐이고 무기한 유효)는 remaining에 남긴다.
"""

import argparse
import json
import os
import tempfile
import time

import requests

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
HEADERS = {"User-Agent": "popqa-temporal-audit/0.1 (research use)"}
BATCH_SIZE = 50

RELATION_TO_PID = {
    "capital": "P36",
    "capital of": "P1376",
    "color": "P462",
    "religion": "P140",
    "country": "P17",
}

REASON_BY_RELATION = {
    "capital": "Wikidata에 예전 수도(deprecated) 또는 특정 기간까지만 유효했던 수도 기록이 있어 실제로 변경된 이력이 확인됨",
    "capital of": "Wikidata에 이 도시가 예전에만 수도였거나 이후 변경된 기록이 확인됨",
    "color": "Wikidata에 이전 상징색(deprecated) 또는 기간 한정 색상 기록이 있어 실제로 변경된 이력이 확인됨",
    "religion": "Wikidata에 개종 등으로 인한 이전 종교(deprecated) 또는 기간 한정 종교 기록이 확인됨",
    "country": "Wikidata에 이전 소속 국가(deprecated) 또는 국경 변화로 인한 기간 한정 기록이 확인됨",
}


def qid_from_uri(uri: str) -> str:
    return uri.rstrip("/").split("/")[-1]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl_atomic(path, rows):
    directory = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".filter-", suffix=".jsonl", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def request_with_retry(params: dict, max_retries: int = 6) -> dict:
    delay = 2.0
    for _ in range(max_retries):
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


def has_confirmed_change(claims: dict, pid: str) -> bool:
    prop_claims = claims.get(pid, [])
    for claim in prop_claims:
        if claim.get("rank") == "deprecated":
            return True
        qualifiers = claim.get("qualifiers", {})
        if "P582" in qualifiers:  # end time
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remaining", default="data/popqa/3. popqa_remaining.jsonl")
    parser.add_argument("--excluded", default="data/popqa/2. popqa_excluded.jsonl")
    args = parser.parse_args()

    remaining = load_jsonl(args.remaining)
    excluded = load_jsonl(args.excluded)
    existing_ids = {str(r["sample_id"]) for r in excluded}

    target_rows = [r for r in remaining if r["prop"] in RELATION_TO_PID]
    other_rows = [r for r in remaining if r["prop"] not in RELATION_TO_PID]
    print(f"target rows (5 relations): {len(target_rows)}")

    unique_qids = sorted({qid_from_uri(r["s_uri"]) for r in target_rows})
    print(f"unique subjects: {len(unique_qids)}")

    entity_claims = {}
    for i in range(0, len(unique_qids), BATCH_SIZE):
        chunk = unique_qids[i : i + BATCH_SIZE]
        entities = fetch_claims_batch(chunk)
        for qid in chunk:
            entity_claims[qid] = entities.get(qid, {}).get("claims", {})
        print(f"  fetched {i + len(chunk)}/{len(unique_qids)}", end="\r")
        time.sleep(0.5)
    print()

    kept = []
    newly_excluded = []
    for row in target_rows:
        rel = row["prop"]
        pid = RELATION_TO_PID[rel]
        subj_qid = qid_from_uri(row["s_uri"])
        claims = entity_claims.get(subj_qid, {})

        if has_confirmed_change(claims, pid):
            sample_id = str(row["id"])
            if sample_id in existing_ids:
                raise ValueError(f"duplicate excluded ID: {sample_id}")
            newly_excluded.append(
                {
                    "sample_id": sample_id,
                    "subject": row["subj"],
                    "subject_qid": subj_qid,
                    "relation": rel,
                    "question": row["question"],
                    "gold_object": row["obj"],
                    "gold_qid": qid_from_uri(row["o_uri"]),
                    "exclusion_type": "confirmed_temporal_change",
                    "reason": REASON_BY_RELATION[rel],
                }
            )
            existing_ids.add(sample_id)
        else:
            kept.append(row)

    final_remaining = other_rows + kept
    write_jsonl_atomic(args.excluded, excluded + newly_excluded)
    write_jsonl_atomic(args.remaining, final_remaining)

    print(f"newly_excluded={len(newly_excluded)}")
    print(f"excluded_total={len(excluded) + len(newly_excluded)}")
    print(f"remaining={len(final_remaining)}")


if __name__ == "__main__":
    main()
