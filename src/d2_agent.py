"""Part D, tasks 5-8: tools, ReAct loop, safe stopping, resumable runs."""
import json, os, re, shutil, sqlite3, sys, time, uuid
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for mock_llm
import pandas as pd
import mock_llm
from d1_rag import retrieve

DB = os.path.join("data", "instamart.db")
LEDGER = os.path.join("outputs", "refund_ledger.jsonl")
RUNS = os.path.join("outputs", "agent_runs.jsonl")
STATE = "state"

# ---------------- tools ----------------
def get_order(order_id):
    with sqlite3.connect(DB) as con:
        row = con.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
                          (order_id,)).fetchone()
    if row is None:
        return {"error": "order not found"}
    dm = row[3]
    if dm is not None and pd.notna(dm) and float(dm).is_integer():
        dm = int(dm)
    return {"order_id": row[0], "status": row[1], "order_amount": float(row[2]),
            "delivery_minutes": None if dm is None or pd.isna(dm) else dm}

def search_policy(query):
    return {"results": retrieve(query)}

def calculate_refund(order_amount, delivery_minutes):
    """Late (> 45 min) completed deliveries get 100% of the amount, otherwise 0.
    (Status is checked by the agent before this tool is called.)"""
    if delivery_minutes is None or float(delivery_minutes) <= 45:
        return {"refund_inr": 0}
    return {"refund_inr": round(float(order_amount), 2)}

def _ledger(tool, order_id, refund_inr):
    """Append one JSON line. Idempotent: the same (tool, order) is never recorded twice,
    so a retried or resumed run can never pay out or escalate the same order twice."""
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    if os.path.exists(LEDGER):
        for line in open(LEDGER, encoding="utf-8"):
            e = json.loads(line)
            if e["tool"] == tool and e["order_id"] == order_id:
                return {"refund_inr": refund_inr, "already_recorded": True}
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), "tool": tool,
                            "order_id": order_id, "refund_inr": refund_inr}) + "\n")
    return {"refund_inr": refund_inr}

def issue_refund(order_id, refund_inr):
    if float(refund_inr) > 1000:                        # safety net: never pay above the auto-approval limit
        return {"error": "refund above INR 1000 needs human approval"}
    return _ledger("issue_refund", order_id, refund_inr)

def escalate_to_human(order_id, refund_inr):
    return _ledger("escalate_to_human", order_id, refund_inr)

TOOLS_SPEC = [
    {"name": "get_order", "description": "Look up an order's status, amount and delivery time in the database.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}},
    {"name": "search_policy", "description": "Search the policy documents and return the best matching passages.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "calculate_refund", "description": "Compute the refund in INR from the order amount and delivery minutes.",
     "parameters": {"type": "object", "properties": {"order_amount": {"type": "number"}, "delivery_minutes": {"type": "number"}},
                    "required": ["order_amount", "delivery_minutes"]}},
    {"name": "issue_refund", "description": "Issue an automatic refund of up to INR 1000 and record it in the ledger.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
    {"name": "escalate_to_human", "description": "Send a refund above INR 1000 to a human support lead and record it.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}

AGENT_SYSTEM = ("You are a refund AGENT. Use the tools to look up the order, check the policy, "
                "calculate the refund, then issue it or escalate it. Finish with STATUS: short reason.")

class SimulatedCrash(Exception):
    pass

# ---------------- checkpoints ----------------
def _checkpoint(st):
    os.makedirs(STATE, exist_ok=True)
    with open(os.path.join(STATE, f"{st['run_id']}.json"), "w", encoding="utf-8") as f:
        json.dump(st, f, indent=1, default=str)

def _fallback(st, status, reason):
    done = ", ".join(s["tool"] for s in st["steps"]) or "none"
    st["status"] = status
    st["final_answer"] = (f"HANDOFF to human agent: {reason}. Tools completed so far: {done}. "
                          f"Request: {st['user_message']}")
    return st

# ---------------- the ReAct loop ----------------
def _loop(st, crash_after_tools=None):
    start = time.monotonic()
    while True:
        if time.monotonic() - start >= st["time_limit_seconds"]:          # (c) wall-clock limit, top of each iteration
            return _fallback(st, "FALLBACK_TIMEOUT", f"time limit of {st['time_limit_seconds']} s reached")
        if st["llm_calls"] >= st["max_iterations"]:                       # (a) iteration limit
            return _fallback(st, "FALLBACK_MAX_ITERATIONS", f"no final answer after {st['max_iterations']} model calls")
        resp = mock_llm.chat(st["messages"], tools=TOOLS_SPEC)
        st["llm_calls"] += 1
        st["tokens_used"] += resp.usage.total_tokens
        if st["tokens_used"] > st["token_limit"]:                         # (b) token limit: stop before running the tool
            return _fallback(st, "FALLBACK_TOKEN_LIMIT", f"token limit of {st['token_limit']} exceeded")
        msg = resp.choices[0].message
        if not msg.tool_calls:
            st["final_answer"] = msg.content
            st["status"] = msg.content.split(":")[0].strip()
            return st
        for tc in msg.tool_calls:
            st["messages"].append({"role": "assistant", "content": None,
                                   "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}]})
            fn = TOOL_MAP.get(tc.name)
            obs = fn(**tc.arguments) if fn else {"error": f"unknown tool {tc.name}"}
            st["messages"].append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": json.dumps(obs)})
            st["steps"].append({"tool": tc.name, "arguments": tc.arguments, "observation": obs})
            st["tools_executed"].append(tc.name)
            st["step"] += 1
            _checkpoint(st)                                               # after every tool observation
            if crash_after_tools is not None and len(st["tools_executed"]) == crash_after_tools:
                raise SimulatedCrash(f"simulated crash after tool #{crash_after_tools}")

def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              run_id=None, crash_after_tools=None):
    st = {"run_id": run_id or f"{scenario_id}-{uuid.uuid4().hex[:6]}", "scenario_id": scenario_id,
          "user_message": user_message, "max_iterations": max_iterations, "token_limit": token_limit,
          "time_limit_seconds": time_limit_seconds, "tokens_used": 0, "llm_calls": 0, "step": 0,
          "tools_executed": [], "steps": [], "status": None, "final_answer": None,
          "messages": [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": user_message}]}
    return _loop(st, crash_after_tools)

def resume_agent(run_id):
    st = json.load(open(os.path.join(STATE, f"{run_id}.json"), encoding="utf-8"))
    return _loop(st)

# ---------------- reporting helpers ----------------
def refund_amount(st):
    for s in reversed(st["steps"]):
        if s["tool"] in ("issue_refund", "escalate_to_human") and "refund_inr" in s["observation"]:
            return float(s["observation"]["refund_inr"])
    return 0.0

def log_run(st, label):
    rec = {"scenario_id": st["scenario_id"], "run_id": st["run_id"], "label": label, "status": st["status"],
           "final_answer": st["final_answer"], "llm_calls": st["llm_calls"], "tokens_used": st["tokens_used"],
           "steps": [{"tool": s["tool"], "arguments": s["arguments"], "observation": s["observation"]} for s in st["steps"]]}
    with open(RUNS, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")

if __name__ == "__main__":
    # fresh outputs so that repeated runs give identical files
    for p in (LEDGER, RUNS):
        if os.path.exists(p):
            os.remove(p)
    shutil.rmtree(STATE, ignore_errors=True)
    os.makedirs("outputs", exist_ok=True)
    sc = pd.read_csv("data/raw/agent_scenarios.csv")

    print("Seven scenarios (default limits):")
    rows = []
    for r in sc.itertuples():
        st = run_agent(r.message, r.scenario_id, run_id=r.scenario_id)
        log_run(st, "scenario")
        amt = refund_amount(st)
        exp = r.expected_outcome
        ok_status = st["status"].startswith("FALLBACK") if exp == "FALLBACK" else st["status"] == exp
        ok_amt = abs(amt - float(r.expected_refund_inr)) <= 0.01
        assert ok_status and ok_amt, (r.scenario_id, st["status"], amt)
        rows.append((r.scenario_id, st["status"], amt, st["llm_calls"], st["tokens_used"]))
        print(f"  {r.scenario_id}: {st['status']:26s} refund {amt:9.2f} | calls {st['llm_calls']} | tokens {st['tokens_used']}")

    s1 = sc.iloc[0]
    print("\nStop-safely demos:")
    a = run_agent(s1.message, "S1", token_limit=200, run_id="S1-token200")
    log_run(a, "demo token_limit=200")
    print(f"  S1 token_limit=200 -> {a['status']} (tokens {a['tokens_used']}, calls {a['llm_calls']}, steps {len(a['steps'])})")
    b = run_agent(s1.message, "S1", time_limit_seconds=0, run_id="S1-time0")
    log_run(b, "demo time_limit_seconds=0")
    print(f"  S1 time_limit_seconds=0 -> {b['status']} (calls {b['llm_calls']})")

    print("\nCrash and resume (S1):")
    try:
        run_agent(s1.message, "S1", run_id="S1-crash", crash_after_tools=2)
    except SimulatedCrash as e:
        ck = json.load(open(os.path.join(STATE, "S1-crash.json")))
        print(f"  crashed: {e}; checkpoint has tools {ck['tools_executed']}, step {ck['step']}, tokens {ck['tokens_used']}")
    c = resume_agent("S1-crash")
    log_run(c, "crash+resume")
    print(f"  resumed -> {c['status']} refund {refund_amount(c):.2f}; tools executed in order: {c['tools_executed']}")
    assert c["tools_executed"] == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
    assert c["status"] == "REFUND" and abs(refund_amount(c) - 128.25) <= 0.01
    print("\nledger lines:", sum(1 for _ in open(LEDGER)), "| agent_runs lines:", sum(1 for _ in open(RUNS)),
          "| checkpoints:", len(os.listdir(STATE)))
