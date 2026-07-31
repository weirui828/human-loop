#!/usr/bin/env python3
"""
llm_label.py

Checkpointed harness for labeling reconstructed TWCS threads by an LLM (`claude-opus-5`) against
the contract in prompts.py, with no API calls.

The labeler here is a human (or an assistant reading in-thread) rather than a
batch API client, so the run is split into two halves that can be separated by
an arbitrary amount of time:

    next    emit the next N unlabeled threads as a numbered worksheet
    ingest  validate a JSONL of labels and append them to the output CSV

Progress lives entirely in the output CSV, keyed by thread_id, so a session can
stop anywhere and the next `next` call picks up exactly where it left off.

The labeler sees ONLY `first_customer_text` -- same contract as build_eval_prompt:
no agent replies, no turn counts. `turn_count` is carried into the output solely
so the post-hoc confound check in `stats` can run; it is never shown at
labeling time.
"""

import os
import csv
import json
import random
import argparse
from collections import Counter, defaultdict
from typing import Dict, Any, List

csv.field_size_limit(10 ** 9)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_CSV = os.path.join(BASE_DIR, "data", "twcs", "reconstructed_threads_20k.csv")

# The labeled dataset is the 5,000-thread random sample and nothing else.
# `llm_labeled_out_of_sample.csv` holds the 639 threads labeled during the
# initial sequential pass that fall outside the manifest; they are kept as extra
# rows but are deliberately NOT part of this file, so anything computed here is
# computed on a uniform sample.
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "twcs", "llm_labeled_5k.csv")
POOL_TXT = os.path.join(BASE_DIR, "data", "twcs", "sample_pool_5000.txt")

FIELDNAMES = [
    "thread_id",
    "escalated",
    "category",
    "reason",
    "labeler",
    "turn_count",
    "first_customer_text",
]

ESCALATE_CATEGORIES = {
    "contact_human_agent",
    "contact_customer_service",
    "complaint",
    "payment_issue",
    "get_refund",
    "check_cancellation_fee",
    "registration_problems",
}
SELF_SERVICE_CATEGORIES = {"self_service"}
ALL_CATEGORIES = ESCALATE_CATEGORIES | SELF_SERVICE_CATEGORIES

try:
    from src.twcs.prompts import extract_first_customer_message
except ImportError:
    from prompts import extract_first_customer_message


def load_input(input_file: str) -> List[Dict[str, Any]]:
    with open(input_file, mode="r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_done(output_file: str) -> Dict[str, Dict[str, Any]]:
    """thread_id -> labeled row, for every thread already written."""
    if not os.path.exists(output_file):
        return {}
    with open(output_file, mode="r", encoding="utf-8") as f:
        return {row["thread_id"]: row for row in csv.DictReader(f) if row.get("thread_id")}


def load_pool(pool_file: str) -> List[str]:
    """Reads a sampling manifest: one thread_id per line, comments with #."""
    with open(pool_file, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


def cmd_sample(args):
    """
    Draws a fixed random sample of threads and writes the manifest.

    Sampling is over EVERY thread with a customer turn, labeled or not, so the
    manifest is one uniform draw from the corpus rather than a random block
    bolted onto whatever was labeled first. Threads that happen to be labeled
    already are simply reused -- `next --pool` skips them -- so the overlap costs
    nothing and the sample keeps its uniformity.

    The manifest is the labeling target from then on: `next --pool` walks it in
    order, so the sample is decided once and does not drift as sessions resume.
    """
    threads = load_input(args.input)
    done = load_done(args.output)

    eligible = [t["thread_id"] for t in threads if extract_first_customer_message(t)]
    if args.n > len(eligible):
        raise SystemExit(f"Asked for {args.n:,} but only {len(eligible):,} threads have a customer turn.")

    rng = random.Random(args.seed)
    sample = rng.sample(eligible, args.n)
    reused = sum(1 for tid in sample if tid in done)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"# {args.n} threads sampled from {len(eligible):,} with a customer turn, seed={args.seed}\n")
        f.write(f"# Source: {os.path.basename(args.input)}; uniform draw over the whole corpus.\n")
        f.write(f"# {reused:,} were already labeled and are reused as-is; {args.n - reused:,} still to label.\n")
        f.write("\n".join(sample) + "\n")

    print(f"Sampled {args.n:,} of {len(eligible):,} eligible threads (seed {args.seed}) -> {args.out}")
    print(f"  Already labeled, reused: {reused:,}")
    print(f"  Still to label:          {args.n - reused:,}")
    print(f"  Labeled but out of sample: {len(done) - reused:,} (kept in the CSV as extra rows)")


def cmd_next(args):
    threads = load_input(args.input)
    done = load_done(args.output)

    if args.pool:
        pool = load_pool(args.pool)
        by_id = {t["thread_id"]: t for t in threads}
        candidates = [by_id[tid] for tid in pool if tid in by_id]
        pool_done = sum(1 for tid in pool if tid in done)
        scope = f"{pool_done:,} of {len(pool):,} sampled threads labeled"
    else:
        candidates = threads
        scope = f"{len(done):,} of {len(threads):,} total threads labeled"

    batch = []
    for thread in candidates:
        if thread.get("thread_id") in done:
            continue
        msg = extract_first_customer_message(thread)
        if not msg:
            # No customer turn: nothing to triage. Skipped permanently, never
            # written to the output, so it stays out of the labeled dataset.
            continue
        batch.append((thread["thread_id"], msg))
        if len(batch) >= args.n:
            break

    lines = [
        f"# Worksheet: {len(batch)} threads | {scope}",
        "# Label each against prompts.py. Customer's opening message only.",
        "",
    ]
    for tid, msg in batch:
        lines.append(f"{tid}\t{msg}")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {len(batch)} threads to {args.out}")
    print(f"Progress: {scope}")


def cmd_ingest(args):
    threads = {t["thread_id"]: t for t in load_input(args.input)}
    done = load_done(args.output)

    with open(args.labels, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    rows, errors, duplicates = [], [], 0
    seen = set()
    for i, rec in enumerate(records, 1):
        tid = rec.get("thread_id", "")
        cat = rec.get("category", "")
        esc = rec.get("escalated")

        if tid not in threads:
            errors.append(f"line {i}: unknown thread_id {tid!r}")
            continue
        if tid in done or tid in seen:
            duplicates += 1
            continue
        if esc not in (0, 1):
            errors.append(f"line {i} ({tid}): escalated must be 0 or 1, got {esc!r}")
            continue
        if cat not in ALL_CATEGORIES:
            errors.append(f"line {i} ({tid}): unknown category {cat!r}")
            continue
        # The category vocabulary already encodes the routing decision, so a
        # mismatch means one of the two fields is a typo. Reject rather than
        # guess which.
        expected = 1 if cat in ESCALATE_CATEGORIES else 0
        if esc != expected:
            errors.append(f"line {i} ({tid}): category {cat!r} implies escalated={expected}, got {esc}")
            continue
        if not str(rec.get("reason", "")).strip():
            errors.append(f"line {i} ({tid}): empty reason")
            continue

        seen.add(tid)
        rows.append({
            "thread_id": tid,
            "escalated": esc,
            "category": cat,
            "reason": rec["reason"],
            "labeler": args.labeler,
            "turn_count": threads[tid].get("turn_count", ""),
            "first_customer_text": extract_first_customer_message(threads[tid]) or "",
        })

    if errors:
        print(f"REJECTED {len(errors)} record(s); nothing written:")
        for e in errors[:30]:
            print(f"  {e}")
        raise SystemExit(1)

    write_header = not os.path.exists(args.output)
    with open(args.output, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    total = len(done) + len(rows)
    # Report against the sample when there is one: the corpus size is not the
    # target, and quoting it makes a finished run look 4% done.
    target = len(load_pool(POOL_TXT)) if os.path.exists(POOL_TXT) else len(threads)
    label = "sampled threads" if os.path.exists(POOL_TXT) else "threads"
    print(f"Appended {len(rows)} labels ({duplicates} already present, skipped)")
    print(f"Progress: {total:,} / {target:,} {label} labeled ({total / target:.1%})")


def cmd_stats(args):
    done = load_done(args.output)
    if not done:
        raise SystemExit("No labels yet.")

    esc = [int(r["escalated"]) for r in done.values()]
    cats = Counter(r["category"] for r in done.values())

    print(f"Labeled:         {len(done):,}")
    print(f"Escalation rate: {sum(esc) / len(esc):.1%}")
    print("\nCategory distribution:")
    for cat, n in cats.most_common():
        print(f"  {cat:28s} {n:6,d}  ({n / len(done):.1%})")

    # The confound check from prompts.py: if escalation rate climbs steeply and
    # monotonically with turn_count, the labels are partly measuring thread
    # length rather than customer need.
    by_turn = defaultdict(list)
    for r in done.values():
        try:
            by_turn[int(r["turn_count"])].append(int(r["escalated"]))
        except (ValueError, TypeError):
            continue
    print("\nEscalation rate by turn_count (confound check):")
    for tc in sorted(by_turn):
        vals = by_turn[tc]
        if len(vals) < 5:
            continue
        print(f"  turn_count={tc:2d}  n={len(vals):5,d}  escalated={sum(vals) / len(vals):.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM labeling harness for TWCS threads (no API calls)")
    parser.add_argument("--input", default=INPUT_CSV)
    parser.add_argument("--output", default=OUTPUT_CSV)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_samp = sub.add_parser("sample", help="Draw a fixed random sample of unlabeled threads")
    p_samp.add_argument("-n", type=int, required=True)
    p_samp.add_argument("--seed", type=int, default=20260729)
    p_samp.add_argument("--out", default=os.path.join(BASE_DIR, "data", "twcs", "sample_pool.txt"))
    p_samp.set_defaults(func=cmd_sample)

    p_next = sub.add_parser("next", help="Emit the next N unlabeled threads as a worksheet")
    p_next.add_argument("-n", type=int, default=100)
    p_next.add_argument("--out", default=os.path.join(BASE_DIR, "worksheet.tsv"))
    p_next.add_argument("--pool", default=POOL_TXT,
                        help="Manifest from `sample`; restricts labeling to those thread_ids. "
                             "Pass --pool '' to walk the whole corpus instead.")
    p_next.set_defaults(func=cmd_next)

    p_ing = sub.add_parser("ingest", help="Validate and append a JSONL of labels")
    p_ing.add_argument("--labels", required=True)
    p_ing.add_argument("--labeler", default="claude-opus-5-in-thread")
    p_ing.set_defaults(func=cmd_ingest)

    p_stat = sub.add_parser("stats", help="Label distribution and turn_count confound check")
    p_stat.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    args.func(args)
