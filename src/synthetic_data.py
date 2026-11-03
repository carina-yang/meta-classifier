"""
synthetic_data.py
-----------------
Generates synthetic column metadata using dual-LLM approach for training data.

Output: (table_name, column_name, data_type -> pii_class)

Generation pipeline:
1. GENERATE: Generator LLM (claude-haiku-5-5) creates realistic column metadata
   - Varies domain, naming style, and generation mode to prevent mode collapse
   - Modes: typical, abbreviated-legacy, ambiguous-table, hard-negative, generic non-PII

2. LABEL (BLIND): Labeler LLM (claude-sonnet-5-5) independently labels each row
   - Uses full taxonomy without seeing intended class
   - Stronger model for quality control

3. QUALITY CONTROL: Keep only rows where generator and labeler agree
   - Disagreements saved to synthetic_disagreements.csv for review
   - Disagreements often expose taxonomy ambiguities
   - Use --trust-labeler to accept high-confidence labeler labels instead

4. POST-PROCESSING: Dedupe, cap repeats, adaptive oversampling across rounds
   - Default: 10,000 rows, 30% non_pii, rest evenly distributed
   - Continues until all classes meet quotas

Usage:
    export ANTHROPIC_API_KEY=...
    python src/synthetic_data.py                      # 10,000 rows
    python src/synthetic_data.py --total 5000         # custom size
    python src/synthetic_data.py --total 600 --mock   # offline test, no API
    python src/synthetic_data.py --resume             # continue interrupted run
"""
import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
from taxonomy import (  # noqa: E402
    LABELING_GUIDANCE, LABELS, NON_PII, PII_LABELS, TAXONOMY,
    nonpii_excludes, render_taxonomy,
)

DOMAINS = [
    "retail e-commerce", "retail banking", "insurance claims", "HR / payroll",
    "telecom billing", "logistics and shipping", "higher education",
    "SaaS product analytics", "public sector services", "travel and hospitality",
    "online gaming", "utilities / energy billing", "CRM / marketing automation",
    "ticketing and events", "real estate / property management",
    "non-profit donations", "automotive dealership", "airline loyalty program",
    "mortgage and lending", "subscription media",
]
NAMING_STYLES = [
    "snake_case", "camelCase", "PascalCase", "UPPER_SNAKE_CASE",
    "terse legacy abbreviations with vowels dropped (cust_nm, addr_ln1)",
    "prefix style (txt_, str_, col_, fld_, c_)",
    "mixed conventions as in a messy legacy warehouse",
    "verbose descriptive names", "ETL staging style (stg_, src_, _raw, _clean suffixes)",
]
POSITIVE_MODES = [("typical", 0.5), ("abbreviated_legacy", 0.3), ("ambiguous_table", 0.2)]
SYSTEM_GEN = ("You generate realistic relational-database column metadata used to train a "
              "data-classification model. Output ONLY a JSON array, no prose.")
SYSTEM_LABEL = ("You are an expert data-governance analyst labeling database columns using a fixed "
                "taxonomy.\n\n" + render_taxonomy() + "\n" + LABELING_GUIDANCE +
                "\n\nOutput ONLY a JSON array of objects "
                '{"id": <int>, "label": "<one of the labels>", "confidence": "high|medium|low"}.')


# ----------------------------------------------------------------------------
# LLM clients
# ----------------------------------------------------------------------------
class AnthropicLLM:
    def __init__(self):
        from anthropic import Anthropic  # pip install anthropic
        self.client = Anthropic()  # reads ANTHROPIC_API_KEY

    def complete(self, model: str, system: str, user: str, temperature: Optional[float] = None, max_tokens: int = 8000) -> str:
        last = None
        for attempt in range(6):
            try:
                # Build kwargs conditionally - some newer models don't support temperature
                kwargs = {
                    "model": model,
                    "max_tokens": max_tokens,
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                }
                if temperature is not None:
                    kwargs["temperature"] = temperature

                resp = self.client.messages.create(**kwargs)
                return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            except Exception as exc:  # rate limits, overloads, transient network
                last = exc
                time.sleep(min(60, 2 ** attempt + random.random()))
        raise RuntimeError(f"LLM call failed after retries: {last}")


class MockLLM:
    """Offline stand-in for smoke-testing the whole pipeline without an API."""
    PREFIXES = ["", "", "", "cust_", "emp_", "primary_", "secondary_", "src_", "legacy_", "stg_", "new_", "old_"]
    SUFFIXES = ["", "", "", "_1", "_2", "_x", "_txt", "_raw"]
    TABLES = ["customers", "tbl_01", "data", "employees", "orders", "accounts", "stg_users", "tmp_load",
              "dim_person", "invoice", "devices", "events", "tickets", "members", "payments"]
    DTYPES = ["varchar(255)", "varchar(50)", "text", "int", "bigint", "date", "timestamp", "char(16)", "boolean", "decimal(10,2)"]
    GENERIC = ["price", "status", "quantity", "created_at", "country", "id", "business_license_id",
               "software_license_key", "updated_at", "total_amount", "sku", "discount_pct", "is_active", "notes_len"]

    def __init__(self):
        self.lookup = {}
        for label, spec in TAXONOMY.items():
            for name in spec["include"]:
                self.lookup[self._norm(name)] = label
        for label, spec in TAXONOMY.items():
            for name, route in spec["exclude"]:
                self.lookup[self._norm(name)] = route

    @staticmethod
    def _norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())

    def _strip(self, s: str) -> str:
        s = s.lower()
        changed = True
        while changed:
            changed = False
            for p in self.PREFIXES:
                if p and s.startswith(p):
                    s, changed = s[len(p):], True
            for x in self.SUFFIXES:
                if x and s.endswith(x):
                    s, changed = s[: -len(x)], True
        return self._norm(s)

    def complete(self, model, system, user, temperature: Optional[float] = None, max_tokens=8000) -> str:
        if "JOB_SPEC:" in user:
            job = json.loads(user.split("JOB_SPEC:")[1].split("\n")[0])
            if job["mode"] == "hard_negative":
                bases = nonpii_excludes(job["source_class"])
            elif job["intended_class"] == NON_PII:
                bases = self.GENERIC
            else:
                bases = TAXONOMY[job["intended_class"]]["include"]
            items = []
            for _ in range(job["n"]):
                col = random.choice(self.PREFIXES) + random.choice(bases) + random.choice(self.SUFFIXES)
                items.append({"table_name": random.choice(self.TABLES), "column_name": col,
                              "data_type": random.choice(self.DTYPES)})
            return json.dumps(items)
        items = json.loads(user.split("ITEMS:")[1].split("\nReturn")[0])
        out = []
        for it in items:
            label = self.lookup.get(self._strip(it["column_name"]), NON_PII)
            if random.random() < 0.03:
                label = random.choice(LABELS)  # simulate labeler noise
            out.append({"id": it["id"], "label": label, "confidence": "high"})
        return json.dumps(out)


# ----------------------------------------------------------------------------
# Prompting
# ----------------------------------------------------------------------------
def parse_json_array(text: str) -> List[dict]:
    try:
        return json.loads(text[text.index("["): text.rindex("]") + 1])
    except (ValueError, json.JSONDecodeError):
        return []


def class_quotas(total: int, non_pii_fraction: float) -> Dict[str, int]:
    per = total * (1 - non_pii_fraction) / len(PII_LABELS)
    quotas = {l: int(round(per)) for l in PII_LABELS}
    quotas[NON_PII] = total - sum(quotas.values())
    return quotas


def weighted_choice(rng: random.Random, pairs):
    r, acc = rng.random(), 0.0
    for value, w in pairs:
        acc += w
        if r <= acc:
            return value
    return pairs[-1][0]


def make_job(label: str, n: int, rng: random.Random, idx: int, rnd: int) -> dict:
    job = {"job_id": f"r{rnd}_{label}_{idx}", "intended_class": label, "n": n,
           "domain": rng.choice(DOMAINS), "naming_style": rng.choice(NAMING_STYLES),
           "multilingual": rng.random() < 0.10, "source_class": None}
    if label == NON_PII:
        if idx % 2 == 0:
            job["mode"] = "generic_non_pii"
        else:
            sources = [l for l in PII_LABELS if nonpii_excludes(l)]
            job["mode"] = "hard_negative"
            job["source_class"] = sources[(idx // 2) % len(sources)]
    else:
        job["mode"] = weighted_choice(rng, POSITIVE_MODES)
    return job


def build_generation_prompt(job: dict) -> str:
    n, label = job["n"], job["intended_class"]
    head = f"JOB_SPEC: {json.dumps(job)}\n"
    style = (f"Domain of the source system: {job['domain']}.\nNaming style: {job['naming_style']}."
             + (" About 10% of names may use French or Spanish words." if job["multilingual"] else "")
             + "\nChoose a realistic SQL data_type for each column (e.g. varchar(255), nvarchar(100), char(2), "
               "text, int, bigint, date, timestamp, decimal(10,2), boolean, uuid, json).\n")
    fmt = (f"\nReturn a JSON array of exactly {n} DISTINCT objects with keys "
           '"table_name", "column_name", "data_type". Vary table names widely (do not reuse one table). '
           "IMPORTANT: Generate realistic database messiness:\n"
           "- Include numbered variants: email_1, email_2, phone_primary, phone_secondary, address_line1, addr_line_01\n"
           "- Use diverse prefixes: old_, new_, legacy_, tmp_, stg_, src_, primary_, secondary_, billing_, shipping_, cust_, usr_\n"
           "- Use common suffixes: _txt, _cd, _flg, _ind, _id, _key, _v2, _old, _new, _raw, _bak, _dt, _nbr, _nm\n"
           "- Mix abbreviation styles inconsistently: cust_first_nm and customer_last_name in same batch\n"
           "- Abbreviate by dropping vowels (addr, nbr, nm, eml, ph, tel, desc) and truncating (cust, desc, addr)\n"
           "Do not include health/medical columns. Do not copy the examples verbatim more than once or twice.")
    if job["mode"] == "hard_negative":
        src = job["source_class"]
        spec = TAXONOMY[src]
        body = (f"Generate columns that look superficially like `{src}` ({spec['definition']}) but are NOT personal "
                f"data of that kind; their correct class is non_pii. Examples of this pattern: "
                + ", ".join(nonpii_excludes(src)) + ". Invent many other similar hard negatives "
                "(flags, counts, ids, codes, types, templates, metadata about the attribute).\n")
    elif job["mode"] == "generic_non_pii":
        body = ("Generate ordinary business/technical columns that contain NO personal data (prices, statuses, "
                "quantities, timestamps, internal ids, license keys, flags, descriptions of products, etc.). "
                "Examples: " + ", ".join(TAXONOMY[NON_PII]["include"]) + ". Include some bare names like `id`, "
                "`status`, `country`, and a bare `date` in uninformative tables.\n")
    else:
        spec = TAXONOMY[label]
        mode_text = {
            "typical": "Typical, clearly-named columns.",
            "abbreviated_legacy": "Terse, abbreviated or legacy-style names a real warehouse would have.",
            "ambiguous_table": "Columns placed in UNINFORMATIVE tables (tbl_01, data, tmp_load, stg_x) so "
                               "the column name and type carry the signal"
                               + (f"; include bare forms for the fallback rule: {spec['fallback']}" if spec["fallback"] else "")
                               + ".",
        }[job["mode"]]
        body = (f"Generate columns whose correct class is `{label}`: {spec['definition']} "
                f"Examples of what belongs: {', '.join(spec['include'])}. "
                f"Things that do NOT belong (do not generate these): "
                f"{', '.join(n for n, _ in spec['exclude'])}.\n{mode_text}\n")
    return head + body + style + fmt


def build_label_prompt(items: List[dict]) -> str:
    payload = [{"id": i, "table_name": r["table_name"], "column_name": r["column_name"],
                "data_type": r["data_type"]} for i, r in enumerate(items)]
    return "ITEMS:\n" + json.dumps(payload) + "\nReturn only the JSON array of labels for every id."


# ----------------------------------------------------------------------------
# Pipeline stages
# ----------------------------------------------------------------------------
def colkey(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def valid_item(it) -> Optional[dict]:
    if not isinstance(it, dict):
        return None
    vals = [str(it.get(k, "")).strip() for k in ("table_name", "column_name", "data_type")]
    if not all(vals) or len(vals[1]) > 64 or len(vals[0]) > 64 or len(vals[2]) > 40:
        return None
    return {"table_name": vals[0], "column_name": vals[1], "data_type": vals[2]}


def run_generation(llm, jobs, model, workers) -> List[dict]:
    out: List[dict] = []

    def one(job):
        raw = llm.complete(model, SYSTEM_GEN, build_generation_prompt(job), temperature=1.0)
        rows = []
        for it in parse_json_array(raw):
            v = valid_item(it)
            if v:
                v.update(intended_class=job["intended_class"], mode=job["mode"],
                         source_class=job["source_class"], job_id=job["job_id"])
                rows.append(v)
        return rows

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, j) for j in jobs]
        for k, f in enumerate(as_completed(futs), 1):
            try:
                out.extend(f.result())
            except Exception as exc:
                print(f"  generation job failed: {exc}")
            if k % 10 == 0 or k == len(futs):
                print(f"  generated {k}/{len(futs)} jobs ({len(out)} candidates)")
    return out


def dedupe(cands: List[dict], pool: List[dict], cap: int) -> List[dict]:
    seen = {(r["table_name"].lower(), r["column_name"].lower(), r["data_type"].lower()) for r in pool}
    colcount = Counter(colkey(r["column_name"]) for r in pool)
    kept = []
    random.shuffle(cands)
    for r in cands:
        key = (r["table_name"].lower(), r["column_name"].lower(), r["data_type"].lower())
        ck = colkey(r["column_name"])
        if key in seen or colcount[ck] >= cap:
            continue
        seen.add(key)
        colcount[ck] += 1
        kept.append(r)
    return kept


def run_labeling(llm, cands, model, workers, batch=50) -> List[dict]:
    chunks = [cands[i:i + batch] for i in range(0, len(cands), batch)]

    def one(chunk):
        # Don't pass temperature for newer models that have deprecated it
        raw = llm.complete(model, SYSTEM_LABEL, build_label_prompt(chunk))
        by_id = {}
        for o in parse_json_array(raw):
            if isinstance(o, dict) and o.get("label") in LABELS and isinstance(o.get("id"), int):
                by_id[o["id"]] = (o["label"], str(o.get("confidence", "medium")).lower())
        res = []
        for i, r in enumerate(chunk):
            if i in by_id:
                r = dict(r, labeled_class=by_id[i][0], confidence=by_id[i][1])
                res.append(r)
        return res

    out = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, c) for c in chunks]
        for k, f in enumerate(as_completed(futs), 1):
            try:
                out.extend(f.result())
            except Exception as exc:
                print(f"  labeling batch failed: {exc}")
            if k % 10 == 0 or k == len(futs):
                print(f"  labeled {k}/{len(futs)} batches")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Generate LLM-labeled synthetic column metadata.")
    p.add_argument("--total", type=int, default=config.SYNTH_TOTAL_ROWS)
    p.add_argument("--non-pii-fraction", type=float, default=config.SYNTH_NON_PII_FRACTION)
    p.add_argument("--out", default=config.RAW_CSV)
    p.add_argument("--gen-model", default=config.SYNTH_GEN_MODEL)
    p.add_argument("--label-model", default=config.SYNTH_LABEL_MODEL)
    p.add_argument("--workers", type=int, default=config.SYNTH_WORKERS)
    p.add_argument("--batch-size", type=int, default=config.SYNTH_BATCH_SIZE)
    p.add_argument("--max-rounds", type=int, default=config.SYNTH_MAX_ROUNDS)
    p.add_argument("--max-per-column-name", type=int, default=config.SYNTH_MAX_PER_COLUMN_NAME)
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--trust-labeler", action="store_true",
                   help="Keep high-confidence labeler labels even when they disagree with the intended class.")
    p.add_argument("--mock", action="store_true", help="Offline smoke test (no API calls).")
    args = p.parse_args()

    random.seed(args.seed)
    rng = random.Random(args.seed)
    llm = MockLLM() if args.mock else AnthropicLLM()
    quotas = class_quotas(args.total, args.non_pii_fraction)
    print(f"Target quotas ({args.total} rows): {quotas}")

    base, ext = os.path.splitext(args.out)
    audit_path, disagree_path = base + "_audit.csv", base + "_disagreements.csv"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    pool: List[dict] = []
    disagreements: List[dict] = []
    if args.resume and os.path.exists(audit_path):
        pool = pd.read_csv(audit_path, dtype=str).fillna("").to_dict("records")
        print(f"Resumed {len(pool)} rows from {audit_path}")
    yield_by_class: Dict[str, float] = {}

    for rnd in range(1, args.max_rounds + 1):
        counts = Counter(r["pii_class"] for r in pool)
        deficits = {l: quotas[l] - counts.get(l, 0) for l in LABELS}
        if all(d <= 0 for d in deficits.values()):
            break
        print(f"\n=== Round {rnd}: deficits { {l: d for l, d in deficits.items() if d > 0} } ===")
        jobs = []
        for label, need in deficits.items():
            if need <= 0:
                continue
            over = min(4.0, max(1.2, 1.0 / yield_by_class[label])) if label in yield_by_class else config.SYNTH_OVERSAMPLE
            n_jobs = math.ceil(need * over / args.batch_size)
            jobs += [make_job(label, args.batch_size, rng, i, rnd) for i in range(n_jobs)]
        rng.shuffle(jobs)
        cands = run_generation(llm, jobs, args.gen_model, args.workers)
        cands = dedupe(cands, pool, args.max_per_column_name)
        print(f"  {len(cands)} candidates after dedupe/cap; labeling...")
        labeled = run_labeling(llm, cands, args.label_model, args.workers)

        gen_count, kept_count = Counter(), Counter()
        for r in labeled:
            gen_count[r["intended_class"]] += 1
            agree = r["labeled_class"] == r["intended_class"]
            take = agree or (args.trust_labeler and r["confidence"] == "high")
            if take:
                row = dict(r, pii_class=r["labeled_class"], round=rnd, agreement=str(agree))
                pool.append(row)
                kept_count[r["intended_class"]] += 1
            else:
                disagreements.append(r)
        for l in gen_count:
            yield_by_class[l] = max(0.05, kept_count[l] / gen_count[l])
        print(f"  kept {sum(kept_count.values())}/{len(labeled)}; yield by class: "
              f"{ {l: round(v, 2) for l, v in yield_by_class.items()} }")
        pd.DataFrame(pool).to_csv(audit_path, index=False)  # checkpoint for --resume

    # ---- finalize: trim to quotas, write outputs ----
    final = []
    for label in LABELS:
        rows = [r for r in pool if r["pii_class"] == label]
        random.shuffle(rows)
        if len(rows) < quotas[label]:
            print(f"WARNING: only {len(rows)}/{quotas[label]} rows for {label}. Re-run with --resume / more rounds.")
        final += rows[: quotas[label]]
    random.shuffle(final)
    df = pd.DataFrame(final)
    df[["table_name", "column_name", "data_type", "pii_class"]].to_csv(args.out, index=False)
    pd.DataFrame(final).to_csv(audit_path, index=False)
    pd.DataFrame(disagreements).to_csv(disagree_path, index=False)

    print(f"\nWrote {len(df)} rows -> {args.out}")
    print(f"Audit (with intended class, mode, confidence): {audit_path}")
    print(f"Disagreements for human review ({len(disagreements)} rows): {disagree_path}")
    print("\nClass distribution:\n" + df["pii_class"].value_counts().to_string())
    if disagreements:
        d = pd.DataFrame(disagreements)
        print("\nMost common intended->labeled disagreements (taxonomy-ambiguity hints):")
        print(d.groupby(["intended_class", "labeled_class"]).size().sort_values(ascending=False).head(10).to_string())


if __name__ == "__main__":
    main()
