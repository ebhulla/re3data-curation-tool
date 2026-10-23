"""LLM judge for institution-name pairs (duplicate / hierarchy / unrelated).

Follows Zheng et al. 2023 (arXiv 2306.05685): every pair is judged twice with the
names swapped to control position bias, and only pairs where both runs agree keep a label.
Provider-agnostic: any OpenAI-compatible chat endpoint, configured through .env.

Usage:  python judge.py --input data/gold_pairs_30.csv --out data/judged_gold.jsonl
"""
import argparse
import json
import os
import time

import pandas as pd
import requests

LABELS = {"duplicate", "hierarchy", "unrelated", "uncertain"}

RUBRIC = """You compare pairs of institution names from a research-data registry.
For each pair, choose exactly one label:
- duplicate: the SAME organization; the names differ only in spelling, capitalization, punctuation, abbreviation, a leading "The", a legal suffix, translation, or minor wording.
- hierarchy: DIFFERENT entities where one is part of the other (a department, institute, library, campus, program, or division inside a university, agency or company), or one is a parent/umbrella of the other.
- unrelated: different organizations with no part-of relationship.
- uncertain: you cannot decide from the names alone.
A specific unit and its parent are NOT duplicates, even if one name contains the other.
Reply with JSON only: {"results": [{"id": <int>, "reason": "<at most 15 words>", "label": "<label>"}, ...]}"""

PROVIDERS = {
    "groq": {"base_url": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY", "model": "openai/gpt-oss-120b"},
}


def load_env(path=".env"):
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


CONTEXT_NOTE = """
Some pairs include a line of ROR registry facts. ROR matches are automatic and can be wrong, so treat
them as evidence to weigh against the names, not as the answer."""


def build_messages(pairs, contexts=None):
    """pairs: list of (name_a, name_b); contexts: optional list of ROR-facts strings, one per pair.
    Ids are positions 0..n-1 within the batch."""
    lines = []
    for i, (a, b) in enumerate(pairs):
        lines.append(f'{i}. A: "{a}" | B: "{b}"')
        if contexts:
            lines.append(f"   {contexts[i]}")
    system = RUBRIC + (CONTEXT_NOTE if contexts else "")
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n".join(lines)}]


def parse_response(text, n):
    """Return a list of n (label, reason) tuples; missing or invalid entries become ('error', '')."""
    out = [("error", "")] * n
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        for item in json.loads(text[start:end])["results"]:
            i, label = int(item["id"]), str(item["label"]).strip().lower()
            if 0 <= i < n and label in LABELS:
                out[i] = (label, str(item.get("reason", "")))
    except (ValueError, KeyError, TypeError):
        pass
    return out


def merge_swapped(forward, backward):
    """Keep a label only if both orderings agree; otherwise mark the pair as 'disagree'."""
    return forward if forward == backward else "disagree"


def call_api(cfg, messages, retries=6):
    headers = {"Authorization": f"Bearer {os.environ[cfg['key_env']]}"}
    payload = {"model": cfg["model"], "messages": messages, "temperature": 0,
               "response_format": {"type": "json_object"}}
    for attempt in range(retries):
        resp = requests.post(f"{cfg['base_url']}/chat/completions", headers=headers, json=payload, timeout=120)
        if resp.status_code == 429:
            time.sleep(float(resp.headers.get("retry-after", 10 * (attempt + 1))))
            continue
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    raise RuntimeError("rate-limited too many times")


def judge_batch(cfg, pairs, ctx_ab=None, ctx_ba=None):
    fwd = parse_response(call_api(cfg, build_messages(pairs, ctx_ab)), len(pairs))
    bwd = parse_response(call_api(cfg, build_messages([(b, a) for a, b in pairs], ctx_ba)), len(pairs))
    return [
        {"label": merge_swapped(f[0], b[0]), "label_ab": f[0], "label_ba": b[0], "reason_ab": f[1], "reason_ba": b[1]}
        for f, b in zip(fwd, bwd)
    ]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV with name_a,name_b (gold file columns 'Name A','Name B' also accepted)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--provider", default="groq")
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--use-context", action="store_true", help="input CSV must have ctx_ab / ctx_ba columns (see ror_context.py)")
    args = ap.parse_args()

    load_env()
    cfg = PROVIDERS[args.provider]
    df = pd.read_csv(args.input).rename(columns={"Name A": "name_a", "Name B": "name_b"})
    if args.limit:
        df = df.head(args.limit)

    done = set()
    if os.path.exists(args.out):
        done = {(r["name_a"], r["name_b"]) for r in map(json.loads, open(args.out, encoding="utf-8"))}
    todo = df[[(a, b) not in done for a, b in zip(df.name_a, df.name_b)]]
    print(f"{len(df)} pairs, {len(done)} already judged, {len(todo)} to go", flush=True)

    with open(args.out, "a", encoding="utf-8") as f:
        for s in range(0, len(todo), args.batch):
            chunk = todo.iloc[s:s + args.batch]
            ctx_ab = chunk.ctx_ab.tolist() if args.use_context else None
            ctx_ba = chunk.ctx_ba.tolist() if args.use_context else None
            results = judge_batch(cfg, list(zip(chunk.name_a, chunk.name_b)), ctx_ab, ctx_ba)
            for (_, row), res in zip(chunk.iterrows(), results):
                f.write(json.dumps({"name_a": row.name_a, "name_b": row.name_b, **res}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{min(s + args.batch, len(todo))}/{len(todo)}", flush=True)
