# LLM labeling conventions

Tie-breaks adopted while labeling `reconstructed_threads_20k.csv` with an LLM against
the criteria in `prompts.py`. The prompt is the contract; this file only records
how recurring ambiguous cases were resolved, so labels stay consistent across
sessions. If a rule here contradicts `prompts.py`, `prompts.py` wins.

Labeler sees the customer's opening message only. No agent text, no turn counts.

## The sample

The target is 5,000 threads, drawn as one uniform random sample over all 19,995
threads that have a customer turn (`llm_label.py sample -n 5000 --seed
20260729`, manifest at `data/twcs/sample_pool_5000.txt`). Label only what the
manifest lists: `llm_label.py next --pool data/twcs/sample_pool_5000.txt`.

Labeling started as a sequential pass before the sample existed, so
`llm_labeled_20k.csv` also holds 639 threads from the first 850 rows of the
file that fall outside the manifest. Those are a narrow slice -- Halloween 2017,
a handful of brands, heavy on promo chatter -- and are NOT part of the random
sample. Keep them for extra training signal if useful, but exclude them when
estimating the escalation base rate or drawing an evaluation set; the manifest
is the only uniform sample.

## Severity bar for `complaint`

The prompt sets the bar at "clearly severe". Operationalized as: escalate when
the message contains at least one of

- an expletive or profane judgment aimed at the company, service or product
  ("this is bullshit", "sucks ass", "so BS", "fuck you", "i'm pissed", "rufkm")
- a categorical negative judgment ("worst service ever", "zero stars", "#FAIL",
  "I despise them", "ridiculous!!", "you should be ashamed")
- a churn or never-again declaration ("never using X again", "not going back")
- a legal/regulatory reference ("lawyer", "BBB", "until the 2 yrs are up")
- an allegation about staff conduct (rude, arrogant, made me cry, lying driver)

Do NOT escalate on: sarcasm alone, sad emoji, "fed up", "so tired of this",
"extremely angry" with no stated issue, shouty caps around an otherwise bare
fault report, or repetition/verbosity. Volume is a `turn_count` proxy (see the
KNOWN CONFOUND note in `prompts.py`).

### Which profanity counts

Swearing is constant in this data, so decide it on grammar, not on tone:

- **Noun-judgment naming the company or product as the bad thing** -> escalate.
  "fix that bullshit", "y'all trash", "the contact page is a joke", "a crock".
- **Profanity aimed at the company's conduct or inaction** -> escalate.
  "get your shit together", "why tf hasn't Apple fixed this", "your lazy ass
  delivery driver".
- **Adjectival intensifier decorating a noun** -> do not escalate. "patch this
  f*cking glitch", "it's fooking freezing", "mine is all kinds of fucked" --
  strip the swear word and an ordinary bug report or aside is left.
- **Profanity about the customer's own state** -> do not escalate. "stressed
  the fuck out". (Escalate on the substance if there is any: "fuck, im 152
  dollars in the red" is `payment_issue` for the overdraft, not for the word.)
- **"this shit" / "my shit" used as a casual noun for the object** -> do not
  escalate. "this shit stay turning on by itself", "not only my shit doing
  this", "now this shit barely makes it past 3pm" all just mean "this thing".
  Contrast the genuine noun-judgment, which predicates badness of the company
  or its product: "fix that bullshit", "your bullshit internet deals",
  "I pay for this shit service".
- **Profanity with no case attached** -> do not escalate; too vague. "I'm sorry
  but wtf is this @Apple [screenshot]".

### Global verdicts vs incident-specific objections

The bar wants a verdict on the company: "worst service ever", "most disgusting
airline I have ever flown with", "TERRIBLE CUSTOMER SERVICE". A firm, polite
objection to one incident is not that, however formal the register: "this is
unacceptable", "it's not OK to give under 24 hours notice", "disappointingly
poor customer service", "very poor customer experience" all stay at 0.
"Disgusted", "shocking" and "very disrespectful" do clear the bar -- they judge
the company's character, not just the incident.

## Recurring case types

| Case | Label | Rationale |
|---|---|---|
| Parcel late, missing, or "delivered" but absent, no refund demand | 0 `self_service` | `track_order`; a status discrepancy to look up |
| Same, plus an explicit refund/compensation demand | 1 `get_refund` | Money-back dispute |
| Same, plus theft/tampering or goods lost with a stated value | 1 `complaint` | Loss claim with severe sentiment |
| Paid content (game, DLC, code) will not download or credit | 0 `self_service` | Fulfillment retry, unless money is disputed |
| Card charged twice / after cancellation / for a waived item | 1 `payment_issue` | Billing error |
| Purchase or renewal cannot complete | 1 `payment_issue` | Payment processing failure |
| Question about a fee or price schedule | 0 `self_service` | `check_invoice` |
| Objection to a fee with outrage ("ridiculous!!", "shame!") | 1 `complaint` | Severity bar met |
| "DM me", "who can I contact", "are your lines open" | 1 `contact_customer_service` | Asking for a rep over automated channels |
| "No phone number", "get past your automated system", "someone who matters" | 1 `contact_human_agent` | Explicitly wants a human/escalation |
| Automated password reset blocked by failed verification | 1 `registration_problems` | The self-service path is the thing that broke |
| Plain password reset request | 0 `self_service` | `recover_password` |
| Account taken over / email changed by intruder | 1 `registration_problems` | Lockout |
| Device or app will not authenticate any account | 1 `registration_problems` | Access blocker |
| Foreign object in food: non-food, injury risk (metal, plastic) | 1 `complaint` | Contamination with a real hazard |
| Foreign object in food: natural or cosmetic (bone, hair, peel) | 0 `self_service` | Unpleasant, but escalate only if the wording clears the severity bar |
| Technical bug, crash, outage, feature request | 0 `self_service` | Not in any escalation category unless severity bar met |
| Brand marketing, giveaways, promo enthusiasm, memes, thanks | 0 `self_service` | `greetings` / `thanks` |
| Too short or vague to classify ("Any help here?", "I have a serious problem") | 0 `self_service` | Prompt says prefer 0 and say so |
| Non-English messages | Label normally | Judged on the same criteria |
| Churn declaration naming the brand ("never ordering from you again") | 1 `complaint` | A departing customer needs a person |
| Churn scoped to one store ("not going to that location anymore") | 0 `self_service` | A grudge against one franchise routes nowhere |
| Missed appointment or install window, reported once | 0 `self_service` | Dispatch status a bot can answer |
| Same, plus repeat failures AND claimed financial harm | 1 `complaint` | Material harm turns it into a case |
| Chasing an unanswered ticket where the message carries the case | 1 `contact_customer_service` | Support silence is the blocker |
| Bare bump with no issue stated ("can anyone answer my question?") | 0 `self_service` | Nothing in the labeled text to route |
| Suspected platform-wide outage ("is sign-in down?") | 0 `self_service` | Status query |
| Account-specific lockout ("worked last week, now rejects my password") | 1 `registration_problems` | Access blocker |
| Stranded mid-journey, time-critical ("at LHR now, urgent") | 1 `contact_customer_service` | Needs an agent before the trip fails |
| Travel disruption reported after the fact, no verdict | 0 `self_service` | Rebooking starts self-serve |
| Advocacy or campaign message (sourcing, animal welfare, disclosure) | 0 `self_service` | No personal case to route |
| Lost property, first report | 0 `self_service` | Standard lost property process |
| Lost property the company already failed to return | 1 `contact_customer_service` | Needs intervention |

## Category choice when several fit

Pick the one that drives routing, in this order:

1. `contact_human_agent` / `contact_customer_service` — an explicit ask for a person
2. `registration_problems` — the customer is locked out
3. `payment_issue` / `get_refund` / `check_cancellation_fee` — money is in dispute
4. `complaint` — everything else that clears the severity bar

So "you idiots keep sending me to collections, get your shit together" is
`payment_issue`, not `complaint`: the abuse is real but the billing dispute is
what a human has to resolve.

A routing question does NOT by itself select a `contact_*` category. "Who can I
talk to about this overcharge?" is `payment_issue`; "who do I report this driver
to?" is `complaint`. The `contact_*` categories are for customers who cannot
reach a human (on hold, hung up on, contact page broken, channel silent) or who
ask outright for a person or a supervisor.

## Known gap: safety incidents

The vocabulary has no category for physical danger -- a crash mid-ride, medical
distress on board, a device that caught fire, an assault or near-assault
disclosed while chasing support. These are filed under `complaint`, or
`contact_customer_service` when the customer is mainly chasing silence. They are
exactly the threads that most need a human, so if the downstream model is used
for routing, treat this as a rubric gap to close rather than a labeling quirk.
