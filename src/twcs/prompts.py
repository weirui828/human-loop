"""
prompts.py

System and Evaluation Prompt Templates for LLM Labeling of Customer
Support Conversation Threads.

Labels in data/twcs/llm_labeled_5k.csv were produced by an LLM following this
contract, under human supervision of scope and tie-breaks. They are model
output, not human ground truth -- see the Threat to Validity note in README.md.

LABELING CONTRACT
-----------------
The labeler sees the customer's OPENING message only (`first_customer_text`)
and nothing else -- not later customer turns, not agent replies, not metadata.
The routing decision being modeled is the one made on arrival, before any reply
exists, so the opening message is the whole of the evidence.

The downstream classifier MUST be given the same column. `customer_text`
(every customer turn concatenated) would hand the model text the labeler never
saw, and would reintroduce the length confound below -- it correlates with
`turn_count` at r = 0.819 versus r = 0.065 for the opening message alone.
`src/evaluate.py` resolves this via TEXT_COLUMN_PREFERENCE and warns if it
falls back to `customer_text`.

Deliberately excluded from the prompt:
  - Agent replies (`full_thread_text`). An agent writing "I've issued your
    refund" or "please DM us" reveals the outcome. Labels built from agent text
    describe what happened, not what the customer needed.
  - `turn_count` / `customer_turn_count`. Thread length is a function of agent
    behaviour and Twitter mechanics, not customer need: a canned "DM us" reply
    truncates a furious customer to 2 turns, while a chatty agent stretches a
    trivial question to 6. KNOWN FAILURE MODE: a labeling run that exposes turn
    counts yields labels reproducible by the rule `turn_count >= 4` at ~0.9 F1 --
    a thread-length statistic wearing an escalation label. The harness prevents
    this structurally: worksheets contain only thread_id and the opening message.

KNOWN CONFOUND
--------------
`len(customer_text)` correlates with `turn_count` at r = 0.819 (median 19 words
at 2 turns, 102 words at 8 turns), versus r = 0.065 for the opening message
alone. Text volume is therefore a strong proxy for thread length, so the
turn-count signal is reachable through sheer verbosity even though the metadata
field is withheld.

Two guards against this:
  1. The prompt below explicitly instructs the labeler not to escalate on volume,
     repetition, or follow-up count alone.
  2. After a labeling run, check escalation rate by `turn_count`. A steep
     monotonic climb means the confound bit and the labels are partly measuring
     conversation length again.

SEVERITY BAR
------------
The SEVERITY BAR and CATEGORY CHOICE sections of the prompt were not written a
priori: they were derived while labeling the 5,000-thread random sample and
folded back in so the stated contract matches the labels that exist. Without
them the prompt says only "clearly severe", and agreement measured between these
labels and a model following this prompt would understate -- the disagreement
would be contract mismatch, not model error.

`src/twcs/LABELING_CONVENTIONS.md` holds the long form: a case-type table, the
food-contamination rules, and the recurring tie-breaks. It is the working
document; this prompt carries only the rules that decide borderline cases.

Note the asymmetry the bar makes explicit. Six of the seven escalation
categories fire on FUNCTION -- a bot cannot move money, restore account access,
or be a person -- and their signals are content words that survive domain
transfer. `complaint` fires on AFFECT, and its signals are register. Bitext is
synthetic and polite, so cross-domain recall is expected to hold up on the
functional categories and fall on `complaint`, which is ~52% of escalations
here. Report per-category recall rather than a single cross-domain F1.
"""

from typing import Dict, Any, Optional

EVALUATION_PROMPT_TEMPLATE = """You are an expert customer service triage evaluator.

A customer has just contacted support. Below is their opening message, exactly as received. No agent has replied yet, and no response has been sent.

---
CUSTOMER MESSAGE:
{customer_message}
---

TASK:
Decide how this message should be routed on arrival.
- `escalated` = 1 -> route to a human agent
- `escalated` = 0 -> can be handled by an automated bot, self-service portal, or FAQ

Judge only what the customer has expressed. Do not speculate about how support replied or whether the issue was eventually resolved.

ESCALATION CRITERIA (`escalated` = 1):
Assign 1 if the message falls under any of these 7 categories requiring human intervention:
1. Direct Request of Human Agent:
   - `contact_human_agent`: Customer demands a real human, live agent, manager, or supervisor (e.g. "connect me to a person", "talk to a human").
   - `contact_customer_service`: Customer asks to speak with support representatives instead of automated tools.
2. Negative Sentiment & Escalated Complaints:
   - `complaint`: Severe dissatisfaction, strong negative sentiment, verbal abuse, or legal/regulatory threats (e.g. "BBB complaint", "lawyer", "worst service ever"). Alone among the seven, this category turns on HOW the customer speaks rather than what they need; apply the SEVERITY BAR below.
3. Financial Liability & Dispute Risk:
   - `payment_issue`: Billing errors, unauthorized/double charges, or payment processing failures.
   - `get_refund`: Explicit requests for money back, price reimbursements, or refund disputes.
   - `check_cancellation_fee`: Inquiries or disputes regarding fees for cancelling a contract/service.
4. Account Access & Onboarding Blockers:
   - `registration_problems`: Password/login lockouts, account suspension/verification failures, or registration blockers.

SELF-SERVICE CRITERIA (`escalated` = 0):
Assign 0 if the query can be resolved via an automated bot, self-service portal, or static FAQ:
1. Information & Policy FAQs:
   - `check_invoice` / `check_payment_methods`: Routine inquiries about accepted payment types, invoice lookups, or standard fee schedules.
   - `delivery_options` / `delivery_period`: Questions about shipping methods, estimated arrival times, or delivery policies.
   - `consult_site`: Requests for online help centers, standard website links, or self-service portals.
2. Order & Logistics Status Lookups:
   - `track_order` / `track_refund`: Checking current order status, shipment tracking numbers, or standard refund status lookups.
3. Automated Account & Profile Self-Service:
   - `recover_password`: Automated password resets, PIN updates, or standard self-service account recovery.
   - `edit_account` / `switch_account` / `delete_account`: Standard settings changes, profile updates, or preference modifications.
4. Standard Self-Service Transactions:
   - `place_order` / `change_order` / `change_shipping_address`: Routine order placements or address updates before shipment.
   - `cancel_order`: Standard self-service cancellation before shipping, without dispute or penalty fee negotiation.
5. Conversational & Social:
   - `greetings` / `thanks`: Acknowledgments, thank-yous, hellos, or closing remarks with no unresolved issue.

SEVERITY BAR FOR `complaint`:
The other six escalation categories fire when a bot CANNOT do the job (move money, restore access, be a person). `complaint` fires when a bot SHOULD NOT answer: the customer has stopped requesting help and started passing judgment on the company, and an automated reply would make things worse. Require at least one of:
- an expletive or profane judgment aimed at the company, service or product ("this is bullshit", "your service sucks", "get your shit together", "i'm pissed")
- a global verdict on the company ("worst service ever", "most disgusting airline I have ever flown with", "zero stars", "#FAIL", "you should be ashamed", "shocking", "disgusted", "I despise them")
- a churn or never-again declaration ("never ordering from you again", "I'm switching to X tomorrow", "you just lost a customer")
- a legal or regulatory reference ("lawyer", "BBB", "FCA", "is this even legal?")
- an allegation about staff conduct (rude, arrogant, lying, hung up on me, drove off, made me cry)

Do NOT escalate on: sarcasm alone, sad or angry emoji, "fed up", "so tired of this", "extremely angry" with no stated issue, shouty caps around an otherwise bare fault report, or repetition and verbosity.

Two distinctions settle most borderline cases:
- PROFANITY. A noun-judgment naming the company or product as the bad thing escalates ("fix that bullshit", "y'all trash", "your contact page is a joke"). An adjectival intensifier does not ("patch this f*cking glitch", "it's fooking freezing", "buggy AF") -- strip the swear word and an ordinary bug report is left. Profanity aimed at the company's conduct escalates ("why tf haven't you fixed this", "your lazy ass driver"); profanity about the customer's own state does not ("stressed the fuck out").
- VERDICTS. A global judgment of the company escalates. A firm objection to one incident does not, however formal the register: "this is unacceptable", "it's not OK to give under 24 hours notice", "very poor customer experience", "disappointingly poor customer service" are all 0.

CATEGORY CHOICE WHEN SEVERAL FIT:
Pick the one that drives routing, in this order:
1. `contact_human_agent` / `contact_customer_service` -- the customer cannot reach a person (on hold, hung up on, contact page broken, channel silent for days) or asks outright for one
2. `registration_problems` -- the customer is locked out
3. `payment_issue` / `get_refund` / `check_cancellation_fee` -- money is in dispute
4. `complaint` -- everything else clearing the severity bar
A routing question does not by itself select a `contact_*` category: "who can I talk to about this overcharge?" is `payment_issue`, and "who do I report this driver to?" is `complaint`.

GUIDANCE ON AMBIGUITY:
- Judge severity from the customer's own words. Frustration without a concrete escalation trigger ("still waiting", "third time I've asked") counts as `complaint` only when the dissatisfaction is clearly severe.
- A bare bump with no issue stated in this message ("can anyone answer my question?") is 0; there is nothing to route. Chasing silence escalates only when the message itself carries the case ("a month of requests with no response", "still awaiting the refund from January").
- A suspected platform-wide outage ("is sign-in down?", "am I the only one?") is 0. An account-specific lockout ("worked last week, now rejects my password") is `registration_problems`.
- A message merely asking about refund status ("where is my refund?") is `track_refund` (0); one demanding or disputing a refund is `get_refund` (1).
- If the message is too short or vague to classify, prefer 0 and say so in the reason. Do not escalate on length alone.

OUTPUT FORMAT:
Respond ONLY with a single valid JSON object. Do not include markdown codeblock tags.

JSON Schema:
{{
  "escalated": 0 or 1,
  "category": "one of: [contact_human_agent, contact_customer_service, complaint, payment_issue, get_refund, check_cancellation_fee, registration_problems, self_service]",
  "reason": "Brief 1-sentence justification grounded in the customer's wording."
}}"""


def extract_first_customer_message(thread: Dict[str, Any]) -> Optional[str]:
    """
    Returns the customer's opening message for a reconstructed thread.

    Prefers an explicit `first_customer_text` column (written by
    reconstruct_conversations.py). Falls back to parsing the first
    "Customer: ..." line out of `full_thread_text`, so CSVs produced before that
    column existed stay usable without re-running reconstruction over twcs.csv.

    Returns None when the thread has no customer turn at all (a handful of
    reconstructed threads are agent-only); callers should skip those.
    """
    explicit = str(thread.get("first_customer_text", "") or "").strip()
    if explicit:
        return explicit

    for line in str(thread.get("full_thread_text", "") or "").split("\n"):
        if line.startswith("Customer:"):
            msg = line[len("Customer:"):].strip()
            if msg:
                return msg
    return None


def build_eval_prompt(thread_or_text: Any) -> str:
    """
    Builds the triage prompt from the customer's opening message only.

    Accepts either a reconstructed-thread dict or a raw message string. Raises
    ValueError when a dict carries no customer turn, so unlabelable threads fail
    loudly instead of being scored on an empty message.
    """
    if isinstance(thread_or_text, dict):
        customer_message = extract_first_customer_message(thread_or_text)
        if not customer_message:
            raise ValueError(
                f"Thread {thread_or_text.get('thread_id', '<unknown>')} has no customer turn; "
                "nothing to label."
            )
    else:
        customer_message = str(thread_or_text).strip()
        if not customer_message:
            raise ValueError("Empty customer message; nothing to label.")

    return EVALUATION_PROMPT_TEMPLATE.format(customer_message=customer_message)
