import csv
from typing import Dict, List, Any, Set

def parse_response_ids(val: str) -> List[int]:
    """Helper to parse response_tweet_id values which might be comma-separated."""
    if not val:
        return []
    val_str = val.strip()
    if not val_str:
        return []
    # If the cell is wrapped in quotes, split by comma
    cleaned = val_str.replace('"', '').replace("'", "")
    return [int(x.strip()) for x in cleaned.split(',') if x.strip().isdigit()]

def parse_int_id(val: str) -> int | None:
    """Helper to parse a single tweet ID, returning None if empty."""
    if not val:
        return None
    try:
        # Handle cases like floats representation (e.g. 123.0)
        return int(float(val.strip()))
    except ValueError:
        return None

def reconstruct_threads(csv_path: str) -> List[List[Dict[str, Any]]]:
    """
    Parses a customer support twitter CSV and reconstructs all conversation paths/threads.
    Uses only Python standard library (no pandas needed).
    
    Returns a list of threads, where each thread is a list of tweet dictionaries in chronological order.
    """
    tweets: Dict[int, Dict[str, Any]] = {}
    
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tweet_id = parse_int_id(row.get('tweet_id', ''))
            if tweet_id is None:
                continue
                
            tweets[tweet_id] = {
                "tweet_id": tweet_id,
                "author_id": row.get('author_id', '').strip(),
                "inbound": row.get('inbound', '').strip().lower() == 'true',
                "created_at": row.get('created_at', '').strip(),
                "text": row.get('text', ''),
                "response_tweet_ids": parse_response_ids(row.get('response_tweet_id', '')),
                "in_response_to_tweet_id": parse_int_id(row.get('in_response_to_tweet_id', ''))
            }

    # Find the starting tweets of all conversations (no parent or parent not in dataset)
    root_ids: List[int] = []
    for tid, tinfo in tweets.items():
        parent = tinfo["in_response_to_tweet_id"]
        if parent is None or parent not in tweets:
            root_ids.append(tid)
            
    root_ids.sort()

    threads: List[List[Dict[str, Any]]] = []
    
    def dfs(current_id: int, current_path: List[Dict[str, Any]], visited: Set[int]):
        """Recursively traverse reply chains using Depth-First Search."""
        if current_id in visited:
            threads.append(current_path.copy())
            return
            
        visited.add(current_id)
        tweet = tweets[current_id]
        new_path = current_path + [tweet]
        
        children = [cid for cid in tweet["response_tweet_ids"] if cid in tweets]
        
        if not children:
            threads.append(new_path)
        else:
            for child_id in children:
                dfs(child_id, new_path, visited.copy())

    for root_id in root_ids:
        dfs(root_id, [], set())
        
    return threads

def print_threads(threads: List[List[Dict[str, Any]]], limit: int = 5):
    """Utility to print threads in a human-readable format."""
    print(f"Reconstructed {len(threads)} conversation threads.\n")
    for i, thread in enumerate(threads[:limit]):
        print(f"--- Thread {i+1} (Length: {len(thread)}) ---")
        for tweet in thread:
            role = "Customer" if tweet['inbound'] else "Support Agent"
            print(f"[{role} ({tweet['author_id']}) - {tweet['tweet_id']}]:")
            print(f"  {tweet['text']}\n")
        print("=" * 60 + "\n")

import json
from typing import Optional

def format_thread(thread: List[Dict[str, Any]], index: int) -> Dict[str, Any]:
    """Helper to cleanly format a single raw conversation thread into a structured dictionary without string syntax errors."""
    thread_id = f"TW_THREAD_{index+1:06d}"
    root_tweet_id = thread[0]["tweet_id"] if thread else None
    
    customer_utterances = []
    full_turns = []
    
    for turn in thread:
        role = "Customer" if turn["inbound"] else f"Agent ({turn['author_id']})"
        clean_text = turn["text"].replace('\r', ' ').replace('\n', ' ').strip()
        full_turns.append(f"{role}: {clean_text}")
        if turn["inbound"]:
            customer_utterances.append(clean_text)
            
    customer_text = " ".join(customer_utterances)
    full_thread_text = "\n".join(full_turns)
    
    return {
        "thread_id": thread_id,
        "root_tweet_id": root_tweet_id,
        "turn_count": len(thread),
        "customer_turn_count": len(customer_utterances),
        "customer_text": customer_text,
        "full_thread_text": full_thread_text,
        "raw_turns": thread
    }

def save_threads_to_file(threads: List[List[Dict[str, Any]]], output_csv: str, output_json: str, limit: Optional[int] = None):
    """Saves reconstructed threads into both JSON and CSV formats for downstream labeling and modeling."""
    target_threads = threads[:limit] if (limit is not None and limit > 0) else threads
    
    formatted_threads = [format_thread(thread, i) for i, thread in enumerate(target_threads)]
        
    # Save JSON
    with open(output_json, mode='w', encoding='utf-8') as f:
        json.dump(formatted_threads, f, indent=2, ensure_ascii=False)
        
    # Save CSV
    with open(output_csv, mode='w', encoding='utf-8', newline='') as f:
        fieldnames = ["thread_id", "root_tweet_id", "turn_count", "customer_turn_count", "customer_text", "full_thread_text"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in formatted_threads:
            row = {k: item[k] for k in fieldnames}
            writer.writerow(row)
            
    print(f"Saved {len(formatted_threads)} reconstructed threads to:\n  - {output_csv}\n  - {output_json}")
    return formatted_threads

if __name__ == "__main__":
    import os
    import sys
    import argparse
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    twcs_full = os.path.join(BASE_DIR, "data", "twcs", "twcs.csv")
    twcs_sample = os.path.join(BASE_DIR, "data", "twcs", "sample.csv")
    
    parser = argparse.ArgumentParser(description="Reconstruct conversation threads from Twitter Customer Support CSV.")
    parser.add_argument("--input", type=str, help="Path to input CSV file (`twcs.csv` or `sample.csv`).")
    parser.add_argument("--limit", type=int, help="Maximum number of threads to serialize (default: 50000 for twcs.csv). Set 0 for all.")
    parser.add_argument("--full", action="store_true", help="Process all threads without limit and save to `reconstructed_threads_full.csv/.json`.")
    parser.add_argument("--out-csv", type=str, help="Custom output path for CSV.")
    parser.add_argument("--out-json", type=str, help="Custom output path for JSON.")
    
    args = parser.parse_args()
    
    if args.input:
        csv_file = args.input
    else:
        csv_file = twcs_full if os.path.exists(twcs_full) else twcs_sample
        
    if args.full or args.limit == 0:
        limit = None
        out_csv = args.out_csv or os.path.join(BASE_DIR, "data", "twcs", "reconstructed_threads_full.csv")
        out_json = args.out_json or os.path.join(BASE_DIR, "data", "twcs", "reconstructed_threads_full.json")
    else:
        limit = args.limit if args.limit is not None else (50000 if csv_file == twcs_full else None)
        out_csv = args.out_csv or os.path.join(BASE_DIR, "data", "twcs", "reconstructed_threads.csv")
        out_json = args.out_json or os.path.join(BASE_DIR, "data", "twcs", "reconstructed_threads.json")
        
    if not os.path.exists(csv_file):
        print(f"Input file not found at {csv_file}")
    else:
        print(f"Reconstructing threads from {csv_file}...")
        all_threads = reconstruct_threads(csv_file)
        print_threads(all_threads, limit=3)
        save_threads_to_file(all_threads, out_csv, out_json, limit=limit)
