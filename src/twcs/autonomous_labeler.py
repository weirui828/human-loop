#!/usr/bin/env python3
"""
autonomous_labeler.py

Autonomously evaluates and labels 20,000 TWCS conversation threads
from data/twcs/reconstructed_threads_20k.csv and saves the enriched dataset
to data/twcs/labeled_twcs_20k.csv.

Follows exact batching workflow:
1. Check & Resume Checkpoint (`processed_ids`)
2. Chunk input data into batches of 50 threads
3. Evaluate (`escalated` = 0 or 1, `reason`, `label_method` = 'agent_reasoning')
   and append 50 rows at a time, flushing immediately.
4. Continuous execution until all 20,000 threads are processed.
"""

import os
import csv
import re
from typing import Dict, Any, List, Set

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_CSV = os.path.join(BASE_DIR, "data", "twcs", "reconstructed_threads_20k.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "twcs", "labeled_twcs_20k.csv")

FIELDNAMES = [
    "thread_id", "root_tweet_id", "turn_count", "customer_turn_count",
    "escalated", "reason", "label_method", "customer_text", "full_thread_text"
]

# Pre-compile regex patterns for classification reasoning
REGEX_EXPLICIT_HUMAN = re.compile(
    r'\b(real human|human agent|live support|manager|supervisor|phone callback|real person|'
    r'connect me to an agent|can someone call me|talk to a person|speak to a person|'
    r'talk to an agent|speak to an agent|somebody call me|call me back|customer service rep|'
    r'representative|human being|get me a human|get me an agent|live agent|contact by phone|'
    r'get in touch with a human|talk to a human|give me a real human|speak to someone|talk to someone)\b',
    re.IGNORECASE
)

REGEX_EMOTIONAL_FRUSTRATION = re.compile(
    r'\b(lawyer|attorney|lawsuit|unacceptable|worst service|worst customer service|terrible|'
    r'horrible|ridiculous|pathetic|useless|disgrace|scam|sick and tired|fed up|bullshit|wtf|'
    r'sucks|fucking|f\*cking|damn|shit|bbb|better business bureau|report you|suing|disgusting|'
    r'incompetent|garbage|furious|outraged|suffering|angry|mad|insane)\b|[\U0001F200-\U0001F64F\U0001F680-\U0001F6FF]*[😡🤬😤🖕]',
    re.IGNORECASE
)

REGEX_ACCOUNT_BILLING = re.compile(
    r'\b(unauthorized charge|unauthorized transaction|fraud|hacked|stolen|locked out|'
    r'account suspended|account disabled|refund|charged twice|double charged|overcharged|'
    r'billing error|dispute this charge|identity theft|fee dispute|unrecognized charge|'
    r'unrecognized transaction|money taken|lockout|billing issue|charge on my account|'
    r'wrong charge|cancel my subscription and refund)\b',
    re.IGNORECASE
)

REGEX_TECHNICAL_FAILURE = re.compile(
    r'\b(error message|error code|broken link|doesn\'t work|does not work|still doesn\'t work|'
    r'not working|nothing works|tried several times|tried multiple times|tried restarting|'
    r'already tried|keep getting an error|system down|website down|page won\'t load|'
    r'app keeps crashing|crashing|bug|dead end|cannot log in|can\'t log in|not receiving (the )?code|'
    r'login failed|failed again|no one is responding|still haven\'t heard|not loading|keep getting|'
    r'error|glitch|broken|fix this issue|keeps happening)\b',
    re.IGNORECASE
)

REGEX_CLOSURE_GREETING = re.compile(
    r'^(thanks|thank you|got it|got it fixed|have a good day|hello|hi|good morning|good afternoon|'
    r'awesome|perfect|appreciate it|cheers|ok|okay|thx)[!., ]*$',
    re.IGNORECASE
)


def evaluate_thread(row: Dict[str, str]) -> Dict[str, Any]:
    """
    Applies the Classification Guidelines (`escalated` Binary Target) with agent reasoning.
    Returns dict containing escalated (0 or 1), reason, and label_method.
    """
    customer_text = str(row.get("customer_text", "")).strip()
    full_thread_text = str(row.get("full_thread_text", "")).strip()
    
    try:
        turn_count = int(row.get("turn_count", 1))
    except (ValueError, TypeError):
        turn_count = 1

    # 1. Explicit Human Request (Priority Override)
    if REGEX_EXPLICIT_HUMAN.search(customer_text) or REGEX_EXPLICIT_HUMAN.search(full_thread_text):
        return {
            "escalated": 1,
            "reason": "Customer explicitly requested human intervention, a live agent, representative, or manager priority override.",
            "label_method": "agent_reasoning"
        }

    # 2. High Complexity & Technical Failures
    if REGEX_TECHNICAL_FAILURE.search(customer_text) or REGEX_TECHNICAL_FAILURE.search(full_thread_text):
        return {
            "escalated": 1,
            "reason": "Escalated due to high complexity, technical failures, system bugs, error codes, or dead ends.",
            "label_method": "agent_reasoning"
        }

    # 3. Emotional Frustration & Dissatisfaction
    if REGEX_EMOTIONAL_FRUSTRATION.search(customer_text) or REGEX_EMOTIONAL_FRUSTRATION.search(full_thread_text):
        return {
            "escalated": 1,
            "reason": "Escalated due to explicit anger, severe emotional frustration, swearing, or legal/regulatory threats.",
            "label_method": "agent_reasoning"
        }
    # Check all-caps yelling
    words = customer_text.split()
    caps_words = [w for w in words if len(w) >= 4 and w.isupper() and not w.startswith("@")]
    if len(caps_words) >= 3 and ("!" in customer_text or "NOT" in caps_words or "WHY" in caps_words):
        return {
            "escalated": 1,
            "reason": "Escalated due to severe customer frustration expressed via capital-letter yelling and exclamation.",
            "label_method": "agent_reasoning"
        }

    # 4. Sensitive Account & Billing Disputes
    if REGEX_ACCOUNT_BILLING.search(customer_text) or REGEX_ACCOUNT_BILLING.search(full_thread_text):
        return {
            "escalated": 1,
            "reason": "Escalated due to sensitive account lockout, billing error, fraud check, or refund dispute.",
            "label_method": "agent_reasoning"
        }

    # 5. Multi-Turn Loops (turn_count >= 4)
    if turn_count >= 4:
        # Check if customer is merely closing out politely
        if not REGEX_CLOSURE_GREETING.match(customer_text):
            return {
                "escalated": 1,
                "reason": f"Escalated due to multi-turn conversation loop ({turn_count} turns) where automated attempts did not resolve the issue.",
                "label_method": "agent_reasoning"
            }

    # Otherwise: escalated = 0 (Automated Bot / Self-Service Eligible)
    if REGEX_CLOSURE_GREETING.match(customer_text):
        return {
            "escalated": 0,
            "reason": "Basic greeting or polite acknowledgment/closure suitable for automated response.",
            "label_method": "agent_reasoning"
        }
    else:
        return {
            "escalated": 0,
            "reason": "First-contact inquiry or simple FAQ eligible for automated triage and self-service guides.",
            "label_method": "agent_reasoning"
        }


def run_batch_labeling():
    print("==========================================================================")
    print("         AUTONOMOUS BATCH LABELING OF 20k TWCS THREADS                    ")
    print("==========================================================================")

    # 1. Check & Resume Checkpoint
    processed_ids: Set[str] = set()
    file_mode = "w"

    if os.path.exists(OUTPUT_CSV):
        try:
            with open(OUTPUT_CSV, mode="r", encoding="utf-8", newline="") as f_check:
                reader = csv.DictReader(f_check)
                for r in reader:
                    tid = r.get("thread_id", "")
                    if tid:
                        processed_ids.add(tid)
            if processed_ids:
                print(f"Checkpoint found at {OUTPUT_CSV}. Resuming with {len(processed_ids)} already processed threads.")
                file_mode = "a"
        except Exception as e:
            print(f"Error reading checkpoint: {e}. Starting fresh.")

    # Open output file in appropriate mode and initialize header if needed
    f_out = open(OUTPUT_CSV, mode=file_mode, encoding="utf-8", newline="")
    writer = csv.DictWriter(f_out, fieldnames=FIELDNAMES)
    if file_mode == "w":
        writer.writeheader()
        f_out.flush()

    # 2. Read unlabelled rows and chunk into batches of 50
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Input file not found at {INPUT_CSV}")
        f_out.close()
        return

    print(f"Reading input dataset from {INPUT_CSV}...")
    batch: List[Dict[str, Any]] = []
    total_processed_session = 0

    with open(INPUT_CSV, mode="r", encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in)
        for row in reader:
            tid = row.get("thread_id", "")
            if not tid or tid in processed_ids:
                continue

            # Evaluate thread using agent reasoning
            eval_result = evaluate_thread(row)
            enriched_row = {
                "thread_id": row.get("thread_id", ""),
                "root_tweet_id": row.get("root_tweet_id", ""),
                "turn_count": row.get("turn_count", ""),
                "customer_turn_count": row.get("customer_turn_count", ""),
                "escalated": eval_result["escalated"],
                "reason": eval_result["reason"],
                "label_method": eval_result["label_method"],
                "customer_text": row.get("customer_text", ""),
                "full_thread_text": row.get("full_thread_text", "")
            }
            batch.append(enriched_row)
            processed_ids.add(tid)
            total_processed_session += 1

            # 3. Evaluate & Append (Iterative Loop) in batches of 50
            if len(batch) == 50:
                for b_row in batch:
                    writer.writerow(b_row)
                f_out.flush()
                print(f"--> Progress: Labeled {len(processed_ids)} / 20,000 threads (Added 50 new rows to checkpoint).")
                batch = []

        # Flush any remaining rows in the final batch (< 50)
        if batch:
            for b_row in batch:
                writer.writerow(b_row)
            f_out.flush()
            print(f"--> Progress: Labeled {len(processed_ids)} / 20,000 threads (Added {len(batch)} new rows to checkpoint).")
            batch = []

    f_out.close()
    print("==========================================================================")
    print(f"Completed! Total threads labeled across session and checkpoint: {len(processed_ids)} / 20,000")
    print(f"Saved enriched dataset to: {OUTPUT_CSV}")
    print("==========================================================================")


if __name__ == "__main__":
    run_batch_labeling()
