"""Re-pull institution records from re3data as clean UTF-8 (one row per repo-institution link)."""
import csv
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import requests

NS = {"r3d": "http://www.re3data.org/schema/4-0"}
BASE = "https://www.re3data.org/api/v40"


def text(elem, path):
    found = elem.find(path, NS)
    return found.text.strip() if found is not None and found.text else None


def get_ids():
    resp = requests.get(f"{BASE}/repositories", timeout=60)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)  # bytes, so the XML's own encoding is respected
    return [e.find("id").text for e in root.findall("repository")]


def pull(repo_id):
    for _ in range(3):
        try:
            resp = requests.get(f"{BASE}/repository/{repo_id}", timeout=60)
            resp.raise_for_status()
            repo = ET.fromstring(resp.content).find("r3d:repository", NS)
            return [
                {
                    "repo_id": repo_id,
                    "institution_name": text(i, "r3d:institutionName"),
                    "institution_country": text(i, "r3d:institutionCountry"),
                    "institution_type": text(i, "r3d:institutionType"),
                    "institution_url": text(i, "r3d:institutionUrl"),
                }
                for i in repo.findall("r3d:institution", NS)
            ]
        except Exception as exc:
            err = exc
    print(f"FAILED {repo_id}: {err}", flush=True)
    return []


if __name__ == "__main__":
    ids = get_ids()
    print(f"{len(ids)} repositories", flush=True)
    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for n, result in enumerate(pool.map(pull, ids), 1):
            rows.extend(result)
            if n % 250 == 0:
                print(f"{n}/{len(ids)}", flush=True)
    with open("data/institutions_clean.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["repo_id", "institution_name", "institution_country", "institution_type", "institution_url"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows", flush=True)
