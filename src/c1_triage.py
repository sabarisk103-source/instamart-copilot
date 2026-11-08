"""Part C: support-ticket triage with guardrails, retries, validation, token budget and routing."""
import json, os, random, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for mock_llm
import pandas as pd
import mock_llm
from mock_llm import TransientLLMError, est_tokens

MAX_INPUT_TOKENS = 400
BASE_DELAY, MAX_RETRIES = 0.05, 3
CATEGORIES = ["billing", "delivery", "technical", "fraud"]

ZERO_SHOT = ('You CLASSIFY customer support tickets into exactly one category: billing, delivery, technical or fraud. '
             'Reply with JSON only, in the form {"category": "...", "reason": "..."}.')
FEW_SHOT = ZERO_SHOT + """
EXAMPLES:
Ticket: I was charged twice for [ORDER_ID]. Please refund the extra payment. -> {"category": "billing", "reason": "double charge"}
Ticket: My order [ORDER_ID] has not arrived and the rider is not picking up. -> {"category": "delivery", "reason": "order not received"}
Ticket: The app crashes every time I open the cart and shows an error. -> {"category": "technical", "reason": "app crash"}
Ticket: There is an unauthorized transaction linked to [ORDER_ID]. I did not place it. -> {"category": "fraud", "reason": "unauthorized transaction"}"""

# redaction patterns, applied in exactly this order
PATTERNS = [("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")),
            ("PHONE", re.compile(r"\b[6-9]\d{9}\b")),
            ("ORDER_ID", re.compile(r"ORD-\d{6}"))]

def redact(text, counts=None):
    for name, pat in PATTERNS:
        text, n = pat.subn(f"[{name}]", text)
        if counts is not None:
            counts[name] += n
    return text

def contains_pii(text):
    return any(p.search(text) for _, p in PATTERNS)

# ---------------- resilient, guarded model call ----------------
retry_log, transient = [], {"errors": 0}

def call_with_retry(messages, ticket_id):
    for m in messages:                                   # privacy guardrail: nothing sensitive leaves the pipeline
        assert not contains_pii(m["content"]), f"PII found in prompt for {ticket_id}"
    for n in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)
        except TransientLLMError:
            transient["errors"] += 1
            if n == MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** n * random.uniform(0.9, 1.15)
            retry_log.append((ticket_id, n, wait))
            print(f"  retry ticket={ticket_id} retry_number={n} wait={wait:.4f}s")
            time.sleep(wait)

# ---------------- output validation ----------------
def validate(content):
    """Returns (status, category). Never guesses."""
    try:
        obj = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        words = set(re.findall(r"[a-z]+", (content or "").lower()))
        found = [c for c in CATEGORIES if c in words]
        return ("classified_keyword", found[0]) if len(found) == 1 else ("flagged_invalid_output", "")
    if isinstance(obj, dict) and obj.get("category") in CATEGORIES:
        return "classified_json", obj["category"]
    return "flagged_invalid_output", ""

# ---------------- pipeline ----------------
def run_pipeline(tickets, system_prompt, token_budget=None, label=""):
    mock_llm.reset()                                     # fresh counters for every full run
    retry_log.clear(); transient["errors"] = 0
    random.seed(42)
    print(f"\n--- run: {label} ---")
    rows, running, pii_counts = [], 0, {"EMAIL": 0, "PHONE": 0, "ORDER_ID": 0}
    prompts_sent = 0
    for t in tickets.itertuples():
        text = t.text
        row = {"ticket_id": t.ticket_id, "customer_id": t.customer_id, "status": "", "category": "", "tokens_used": 0}
        if not text.strip():
            row["status"] = "skipped_empty"
        elif est_tokens(text) > MAX_INPUT_TOKENS:
            row["status"] = "flagged_long"
        elif token_budget is not None and running >= token_budget:
            row["status"] = "budget_exhausted"
        else:
            msgs = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": redact(text, pii_counts)}]
            try:
                resp = call_with_retry(msgs, t.ticket_id)
                prompts_sent += 1
                used = resp.usage.total_tokens
                running += used
                row["tokens_used"] = used
                row["status"], row["category"] = validate(resp.choices[0].message.content)
            except TransientLLMError:
                row["status"] = "failed_after_retries"
        rows.append(row)
    res = pd.DataFrame(rows)
    res.attrs.update(running=running, errors=transient["errors"], retries=len(retry_log), sent=prompts_sent)
    return res

# ---------------- routing ----------------
def add_routing(res, segments):
    seg = segments.set_index("customer_id")
    res = res.copy()
    res["segment_name"] = res.customer_id.map(seg["segment_name"]).fillna("")
    premium = res.customer_id.map(seg["is_premium"]).fillna(False).astype(bool)
    human = res.status.isin(["skipped_empty", "flagged_long", "flagged_invalid_output",
                             "budget_exhausted", "failed_after_retries"])
    route, prio = [], []
    for h, cat, p in zip(human, res.category, premium):
        if h:
            route.append("human_review"); prio.append("NORMAL")
        elif cat == "fraud":
            route.append("security_team"); prio.append("HIGH")
        elif p:
            route.append("premium_desk"); prio.append("HIGH")
        else:
            route.append("support_queue"); prio.append("NORMAL")
    res["priority"], res["route"] = prio, route
    return res[["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]]

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)   # empty texts stay ""
    segments = pd.read_csv("outputs/customer_segments.csv")

    # ---- guardrail and redaction facts (all 164 tickets) ----
    empty = int((tickets.text.str.strip() == "").sum())
    long_ = int(tickets.text.map(lambda x: est_tokens(x) > MAX_INPUT_TOKENS).sum())
    print(f"empty tickets: {empty} | long tickets: {long_} | est_tokens of a long ticket: "
          f"{sorted(set(est_tokens(x) for x in tickets.text if est_tokens(x) > MAX_INPUT_TOKENS))}")
    counts = {"EMAIL": 0, "PHONE": 0, "ORDER_ID": 0}
    redacted = tickets.text.map(lambda x: redact(x, counts))
    print("substitutions:", counts, "total", sum(counts.values()))
    leftovers = sum(len(p.findall(x)) for x in redacted for _, p in PATTERNS)
    print("patterns still found in redacted texts:", leftovers)
    print("system prompt est. tokens: zero-shot", est_tokens(ZERO_SHOT), "| few-shot", est_tokens(FEW_SHOT))

    # ---- zero-shot vs few-shot full runs (no budget) ----
    summary = {}
    for name, prompt in [("zero-shot", ZERO_SHOT), ("few-shot", FEW_SHOT)]:
        res = run_pipeline(tickets, prompt, None, name)
        ok = res[res.status.isin(["classified_json", "classified_keyword"])].merge(
            tickets[["ticket_id", "true_category"]], on="ticket_id")
        acc = (ok.category == ok.true_category).mean()
        # tokens consumed by every call (including the 9 flagged ones, which were still paid for)
        total = res.attrs["running"]
        summary[name] = res
        print(f"[{name}] status counts: {res.status.value_counts().to_dict()}")
        print(f"[{name}] TransientLLMErrors {res.attrs['errors']} | retries logged {res.attrs['retries']} "
              f"| failed_after_retries {(res.status == 'failed_after_retries').sum()}")
        waits = [w for _, n, w in retry_log]
        print(f"[{name}] retry waits min {min(waits):.4f} max {max(waits):.4f} (allowed 0.0450-0.0575 for retry 0)"
              if waits else "")
        assert all(n == 0 and 0.045 <= w <= 0.0575 for _, n, w in retry_log)
        print(f"[{name}] classified {len(ok)} | accuracy {acc*100:.1f}% ({int((ok.category == ok.true_category).sum())} of {len(ok)}) "
              f"| total tokens {total} | tokens per classified ticket {total/len(ok):.1f} "
              f"| tokens per sent ticket {total/res.attrs['sent']:.1f}")
        if name == "zero-shot":
            print("zero-shot confusion table (rows = true, columns = predicted):")
            print(pd.crosstab(ok.true_category, ok.category).to_string())

    # ---- budget demo (zero-shot) ----
    BUDGET = 6000
    demo = run_pipeline(tickets, ZERO_SHOT, BUDGET, f"zero-shot, TOKEN_BUDGET={BUDGET}")
    add_routing(demo, segments).to_csv("outputs/triage_budget_demo.csv", index=False)
    print(f"budget demo: budget {BUDGET} | tokens used {demo.attrs['running']} | "
          f"budget_exhausted {int((demo.status == 'budget_exhausted').sum())}")
    print("budget demo status counts:", demo.status.value_counts().to_dict())

    # ---- final triage: few-shot, TOKEN_BUDGET = 100000 ----
    final = run_pipeline(tickets, FEW_SHOT, 100000, "few-shot, TOKEN_BUDGET=100000")
    out = add_routing(final, segments)
    out.to_csv("outputs/triage_results.csv", index=False)
    print("final status counts:", out.status.value_counts().to_dict())
    print("route x priority:\n", out.groupby(["route", "priority"]).size().to_string())
    print("rows:", len(out))
