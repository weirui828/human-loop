#!/usr/bin/env python3
"""
label_twcs.py

Auto-labels reconstructed Twitter Customer Support (TWCS) conversation threads
with binary escalation targets (`escalated` = 1 for human agent needed, 0 for automated bot).

Supports:
1. Live LLM API labeling via Google Gemini API if GEMINI_API_KEY or GOOGLE_API_KEY is set in `.env` or environment.
2. High-accuracy heuristic simulation (--simulated or fallback if no key set) for offline testing.
"""

import os
import csv
import time
import json
import argparse
from typing import List, Dict, Any, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_JSON = os.path.join(BASE_DIR, "data", "twcs", "reconstructed_threads_20k.json")
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "twcs", "labeled_twcs_20k.csv")
OUTPUT_JSON = os.path.join(BASE_DIR, "data", "twcs", "labeled_twcs_20k.json")


def label_thread_heuristic(thread: Dict[str, Any]) -> Dict[str, Any]:
    """
    Applies domain heuristics to classify if a Twitter thread requires human escalation.
    Safely handles both formatted dicts (`Dict[str, Any]`) and raw turn lists (`List[Dict[str, Any]]`).
    """
    if isinstance(thread, list):
        try:
            from reconstruct_conversations import format_thread
            thread = format_thread(thread, 0)
        except ImportError:
            pass

    customer_text = str(thread.get("customer_text", "")).lower()
    full_text = str(thread.get("full_thread_text", "")).lower()
    turns = int(thread.get("turn_count", 1))
    
    # Escalation markers in customer text
    escalation_keywords = [
        "dead end", "doesn't work", "tried several times", "still haven't heard",
        "😡", "not happy", "concern", "manager", "lawyer", "supervisor",
        "unauthorized", "failed", "broken", "worst", "terrible", "disregarded",
        "human", "agent", "call me", "voicemail", "error message", "suffering"
    ]
    
    # Self-service / minor inquiry markers
    simple_keywords = [
        "thanks", "thank you", "got it", "hello", "hi", "just asking",
        "when does", "what time", "how to"
    ]
    
    escalation_score = sum(1 for kw in escalation_keywords if kw in customer_text)
    simple_score = sum(1 for kw in simple_keywords if kw in customer_text)
    
    # Decision logic
    if escalation_score > 0 or turns >= 4:
        escalated = 1
        reasons = [kw for kw in escalation_keywords if kw in customer_text]
        if turns >= 4:
            reasons.append(f"high multi-turn loop ({turns} turns)")
        reason = "Escalated due to: " + ", ".join(reasons if reasons else ["complex inquiry"])
    elif simple_score > 0 and escalation_score == 0:
        escalated = 0
        reason = "Self-service/simple inquiry handled by standard response."
    else:
        # Default fallback based on length/complexity
        if len(customer_text.split()) > 25:
            escalated = 1
            reason = "Escalated due to detailed problem description requiring troubleshooting."
        else:
            escalated = 0
            reason = "Standard initial inquiry eligible for automated triage."
            
    return {
        "escalated": escalated,
        "reason": reason,
        "label_method": "heuristic_simulated"
    }


def label_thread_llm(thread: Dict[str, Any], api_key: str, model: str = "gemini-1.5-flash", max_retries: int = 3, delay: float = 4.1) -> Dict[str, Any]:
    """
    Labels a single thread using the Google Gemini API (`gemini-1.5-flash`).
    Enforces pace control (e.g. 4.1s delay to respect 15 Requests Per Minute free tier limit)
    and handles exponential backoff on 429 RESOURCE_EXHAUSTED errors.
    """
    try:
        import requests
    except ImportError:
        return label_thread_heuristic(thread)
        
    prompt = f"""You are an expert customer service evaluator.
Analyze the following customer support conversation thread from Twitter:

{thread.get('full_thread_text', '')}

Determine whether the customer's issue requires human escalation (`escalated` = 1, e.g., complex failure, dead ends, emotional frustration, account/billing dispute) OR if it can/could be resolved by an automated bot (`escalated` = 0, e.g., simple FAQ, basic instruction check, initial greeting).

Respond ONLY with valid JSON in the format:
{{"escalated": 0 or 1, "reason": "brief explanation"}}"""

    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json"
        }
    }
    
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=20)
            if resp.status_code == 429:
                sleep_time = 15 * (2 ** attempt)
                print(f"  [Rate Limit 429] RESOURCE_EXHAUSTED for {thread.get('thread_id', 'unknown')}. Backing off for {sleep_time}s...")
                time.sleep(sleep_time)
                continue
                
            resp.raise_for_status()
            raw_text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            result = json.loads(raw_text)
            
            # Pace control after successful call
            if delay > 0:
                time.sleep(delay)
                
            return {
                "escalated": int(result.get("escalated", 0)),
                "reason": result.get("reason", "LLM evaluation"),
                "label_method": f"llm_{model}"
            }
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                sleep_time = 15 * (2 ** attempt)
                print(f"  [Rate Limit Error] {e}. Backing off for {sleep_time}s...")
                time.sleep(sleep_time)
                continue
            print(f"  [Warning] LLM labeling failed for {thread.get('thread_id', 'unknown')}: {e}. Falling back to heuristic.")
            return label_thread_heuristic(thread)
            
    print(f"  [Warning] Exhausted {max_retries} retries due to rate limit for {thread.get('thread_id', 'unknown')}. Falling back to heuristic.")
    return label_thread_heuristic(thread)


def run_labeling(simulated: bool = False, full: bool = False, input_path: Optional[str] = None, output_csv: Optional[str] = None, output_json: Optional[str] = None, from_raw: bool = False, delay: float = 4.1, max_retries: int = 3, max_samples: Optional[int] = None):
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    use_llm = (api_key is not None) and (not simulated)
    
    if full:
        default_in_json = os.path.join(BASE_DIR, "data", "twcs", "reconstructed_threads_full.json")
        default_in_csv = os.path.join(BASE_DIR, "data", "twcs", "reconstructed_threads_full.csv")
        out_csv = output_csv or os.path.join(BASE_DIR, "data", "twcs", "labeled_twcs_full.csv")
        out_json = output_json or os.path.join(BASE_DIR, "data", "twcs", "labeled_twcs_full.json")
    else:
        default_in_json = INPUT_JSON
        default_in_csv = os.path.join(BASE_DIR, "data", "twcs", "reconstructed_threads_20k.csv")
        out_csv = output_csv or OUTPUT_CSV
        out_json = output_json or OUTPUT_JSON

    in_file = input_path
    if not in_file and not from_raw:
        if os.path.exists(default_in_json):
            in_file = default_in_json
        elif os.path.exists(default_in_csv):
            in_file = default_in_csv
        else:
            in_file = INPUT_JSON

    if use_llm:
        model = "gemini-1.5-flash"
        print(f"Starting LLM-assisted labeling using model: {model}")
        print(f"Enforcing pace control: {delay}s delay (~{60/delay:.1f} RPM) | Max retries on 429: {max_retries}")
    else:
        print("Running domain heuristic evaluation (simulated/offline mode)...")

    fieldnames = [
        "thread_id", "root_tweet_id", "turn_count", "customer_turn_count",
        "escalated", "reason", "label_method", "customer_text", "full_thread_text"
    ]

    processed_ids = set()
    file_mode = "w"
    if os.path.exists(out_csv):
        try:
            with open(out_csv, mode="r", encoding="utf-8") as f_check:
                reader = csv.DictReader(f_check)
                for r in reader:
                    if "thread_id" in r and r["thread_id"]:
                        processed_ids.add(r["thread_id"])
            if processed_ids:
                print(f"Resuming checkpoint! Found {len(processed_ids)} already labeled threads in {out_csv}.")
                file_mode = "a"
        except Exception as e:
            print(f"Could not check checkpoint: {e}. Starting fresh.")

    f_out = open(out_csv, mode=file_mode, encoding="utf-8", newline="")
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    if file_mode == "w":
        writer.writeheader()
        f_out.flush()

    if from_raw:
        print("Reconstructing and labeling threads directly from raw twcs.csv...")
        from reconstruct_conversations import reconstruct_threads, format_thread
        raw_twcs = os.path.join(BASE_DIR, "data", "twcs", "twcs.csv")
        if not os.path.exists(raw_twcs):
            raw_twcs = os.path.join(BASE_DIR, "data", "twcs", "sample.csv")
        raw_threads = reconstruct_threads(raw_twcs)
        threads_iter = (format_thread(t, i) for i, t in enumerate(raw_threads))
    elif in_file and in_file.endswith(".csv"):
        print(f"Reading threads from CSV: {in_file}")
        f_in = open(in_file, mode="r", encoding="utf-8")
        threads_iter = csv.DictReader(f_in)
    elif in_file and os.path.exists(in_file):
        print(f"Reading threads from JSON: {in_file}")
        with open(in_file, mode="r", encoding="utf-8") as f_in:
            threads_iter = json.load(f_in)
    else:
        print(f"Error: Input file {in_file} not found. Run reconstruct_conversations.py first or pass --from-raw.")
        f_out.close()
        return

    labeled_data = []
    escalated_count = 0
    newly_labeled = 0

    for thread in threads_iter:
        tid = thread.get("thread_id", "")
        if tid in processed_ids:
            continue

        if use_llm:
            result = label_thread_llm(thread, api_key, model=model, max_retries=max_retries, delay=delay)
        else:
            result = label_thread_heuristic(thread)
            
        merged = {**thread, **result}
        labeled_data.append(merged)
        if int(merged.get("escalated", 0)) == 1:
            escalated_count += 1

        row = {k: merged.get(k, "") for k in fieldnames}
        writer.writerow(row)
        f_out.flush()
        
        processed_ids.add(tid)
        newly_labeled += 1
        
        if newly_labeled % 50 == 0:
            print(f"  --> Progress: Labeled {newly_labeled} new threads (Total in checkpoint: {len(processed_ids)})")
            
        if max_samples and newly_labeled >= max_samples:
            print(f"Reached max_samples target ({max_samples}). Stopping current run cleanly.")
            break

    f_out.close()
    if in_file and in_file.endswith(".csv"):
        f_in.close()

    total_in_file = len(processed_ids)
    print(f"\nLabeling Session Complete! Labeled {newly_labeled} new threads in this session.")
    print(f"Total labeled threads now in checkpoint ({out_csv}): {total_in_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label TWCS threads with binary escalation target.")
    parser.add_argument("--simulated", action="store_true", help="Force heuristic simulation even if API key is present.")
    parser.add_argument("--full", action="store_true", help="Process full reconstructed dataset and save to `labeled_twcs_full.csv`.")
    parser.add_argument("--input", type=str, help="Path to input JSON or CSV reconstructed threads file.")
    parser.add_argument("--out-csv", type=str, help="Path to output CSV file.")
    parser.add_argument("--out-json", type=str, help="Path to output JSON file.")
    parser.add_argument("--from-raw", action="store_true", help="Directly reconstruct and label from raw `twcs.csv` on the fly without intermediate files.")
    parser.add_argument("--delay", type=float, default=4.1, help="Delay in seconds between API requests to respect rate limits (default: 4.1s for 15 RPM).")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries on 429 RESOURCE_EXHAUSTED errors (default: 3).")
    parser.add_argument("--max-samples", type=int, default=None, help="Stop after labeling this many new threads in the current session.")
    args = parser.parse_args()
    
    run_labeling(
        simulated=args.simulated,
        full=args.full,
        input_path=args.input,
        output_csv=args.out_csv,
        output_json=args.out_json,
        from_raw=args.from_raw,
        delay=args.delay,
        max_retries=args.max_retries,
        max_samples=args.max_samples
    )
