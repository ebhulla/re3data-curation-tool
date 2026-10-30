"""Match every distinct re3data institution name to ROR (top 3 candidates each).

Resumable: results append to data/ror_matches.jsonl and finished names are skipped on restart.
"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

URL = "https://api.ror.org/v2/organizations"
OUT = "data/ror_matches.jsonl"


def compact(item):
    org = item["organization"]
    display = next((n["value"] for n in org["names"] if "ror_display" in n["types"]), None)
    return {
        "ror_id": org["id"],
        "ror_name": display,
        "score": item["score"],
        "chosen": item["chosen"],
        "matching_type": item["matching_type"],
        "types": org["types"],
        "countries": [l["geonames_details"]["country_code"] for l in org["locations"]],
        "relationships": [{"type": r["type"], "id": r["id"], "label": r["label"]} for r in org["relationships"]],
    }


def match(name):
    for attempt in range(6):
        try:
            resp = requests.get(URL, params={"affiliation": name}, timeout=60)
            if resp.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            resp.raise_for_status()
            return {"name": name, "candidates": [compact(i) for i in resp.json()["items"][:3]]}
        except Exception as exc:
            err = str(exc)
            time.sleep(2)
    return {"name": name, "error": err}


if __name__ == "__main__":
    names = sorted(pd.read_csv("data/institutions_clean.csv").institution_name.dropna().unique())
    done = set()
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            done = {json.loads(line)["name"] for line in f if "error" not in json.loads(line)}
    todo = [n for n in names if n not in done]
    print(f"{len(names)} names, {len(done)} done, {len(todo)} to go", flush=True)
    with open(OUT, "a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=3) as pool:
        for n, rec in enumerate(pool.map(match, todo), 1):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if n % 250 == 0:
                print(f"{n}/{len(todo)}", flush=True)
    print("finished", flush=True)
