# Instamart Customer Intelligence & Support Copilot

## Overview
One repository, four connected parts, all on synthetic data (INR throughout; nothing here is real Instamart data).
Part A cleans raw exports into a SQLite database, Part B segments customers and predicts lapse, Part C triages support tickets with guardrails, Part D is a policy RAG assistant and refund agent.
`generate_data.py` and `mock_llm.py` are the supplied files, unmodified. Code lives in `src/` (a1–a5 Part A, b1–b2 Part B, c1 Part C, d1–d2 Part D), queries in `sql/queries.sql`, results in `outputs/`, agent checkpoints in `state/`. No API key, network or paid service is used.

## Reproduce
```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install pandas numpy scikit-learn joblib
pip freeze > requirements.txt        # (already committed)
python generate_data.py              # last line: generated: customers=1200 products=40 orders_raw=13084 tickets=164
python src/a1_audit.py               # Part A: raw audit
python src/a2_clean.py               # Part A: clean, write data/clean/*.csv and data/instamart.db
python src/a3_validate.py            # Part A: prints ALL CHECKPOINTS PASSED
python src/a4_sql.py                 # Part A: runs sql/queries.sql
python src/a5_pandas_eda.py          # Part A: pandas cross-check and EDA
python src/b1_segments.py            # Part B: RFM, K-means, segments, PCA -> outputs/customer_segments.csv
python src/b2_lapse.py               # Part B: lapse model -> outputs/lapse_model.joblib, outputs/retention_list.csv
python src/c1_triage.py             # Part C: triage -> outputs/triage_results.csv, outputs/triage_budget_demo.csv
python src/d1_rag.py                # Part D: chunking, retrieval, grounding evaluation (overlap 10 and 0)
python src/d2_agent.py              # Part D: tools, 7 scenarios, stop-safely demos, crash-and-resume
```

## Part A

### Audit of the raw data
Shapes: customers (1200, 5), products (40, 4), orders (13084, 9). Data types are as read by pandas 3 (text columns show as `str`, which is `object` in older pandas). Only truly empty cells count as missing; the text `N/A` is read as a string so the audit sees it (the `order_amount` column therefore shows 0 blanks, and its 120 unparseable entries appear in the table below).

| Table | Column | dtype | Missing | Missing % |
|---|---|---|---|---|
| customers | customer_id | str | 0 | 0.00 |
| customers | city | str | 0 | 0.00 |
| customers | signup_date | str | 0 | 0.00 |
| customers | age | float64 | 61 | 5.08 |
| customers | gender | str | 30 | 2.50 |
| products | product_id | str | 0 | 0.00 |
| products | product_name | str | 0 | 0.00 |
| products | category | str | 0 | 0.00 |
| products | price_inr | int64 | 0 | 0.00 |
| orders | order_id | str | 0 | 0.00 |
| orders | customer_id | str | 0 | 0.00 |
| orders | product_id | str | 0 | 0.00 |
| orders | quantity | int64 | 0 | 0.00 |
| orders | order_ts | str | 0 | 0.00 |
| orders | status | str | 0 | 0.00 |
| orders | discount_pct | int64 | 0 | 0.00 |
| orders | order_amount | str | 0 | 0.00 |
| orders | delivery_minutes | float64 | 2,541 | 19.42 |

| Audit figure | Value |
|---|---|
| raw orders rows | 13,084 |
| exact duplicate rows | 119 |
| rows with status pending | 656 |
| distinct order_id | 12,309 |
| non-numeric order_amount | 120 |
| negative delivery_minutes | 106 |
| blank delivery_minutes | 2,541 |
| blank age | 61 |
| age outside 10–100 | 6 |
| distinct raw city spellings | 24 |
| distinct non-blank gender spellings | 10 |
| blank gender | 30 |

### Cleaning rules and what each changed
| # | Rule | Changed |
|---|---|---|
| C1 | city: strip spaces, title case (six cities remain) | 476 values |
| C2 | gender: map to male / female / unknown (blank → unknown) | 30 blanks → unknown |
| C3 | age outside 10–100 → missing, then fill with city median (groupby + transform) | 6 set missing; 67 filled |
| C4 | signup_date → datetime | all rows |
| O1 | order_ts → datetime | all rows |
| O2 | drop exact duplicate rows | 119 rows |
| O3 | per order_id keep the latest order_ts (drops earlier pending records) | 656 rows |
| O4 | order_amount → numeric; unparseable → missing → `round(price × quantity × (1 − discount/100), 2)` | 112 values refilled |
| O5a | negative delivery_minutes → missing | 105 values |
| O5b | completed orders: missing delivery_minutes → median of the product category (completed orders only, groupby + transform) | 774 values |
| O5c | cancelled orders keep missing delivery_minutes (never delivered; imputing would invent data) | 1,199 left missing |

Results after cleaning match every reference: cities Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184; genders female 610, male 560, unknown 30; ages 18–60 with none missing; 12,309 orders (11,110 completed, 1,199 cancelled, 0 pending); completed revenue INR 2,998,668.80. `src/a3_validate.py` asserts 16 checkpoints and prints `ALL CHECKPOINTS PASSED`.

### SQL results (`sql/queries.sql`, run by `src/a4_sql.py`)
| Query | Result |
|---|---|
| Q1 WHERE + HAVING | 30 customers |
| Q2 three-table join | 30 rows; total INR 2,998,668.80; largest Bengaluru / Electronics INR 233,909.45 |
| Q3 scalar subquery | 11 products above the average price |
| Q4 LEFT JOIN | 25 customers with no completed order |
| Q5 CTEs | 550 customers above the average of INR 2,552.06 |
| Q6 window ranks | top per city: Bengaluru C0082 12,856.20; Chennai C0930 10,151.40; Delhi C0291 13,742.60; Hyderabad C0743 12,730.05; Mumbai C0693 14,766.55; Pune C0426 10,377.10; 8 rows with RANK ≠ ROW_NUMBER |
| Q7 running total | 183 daily rows; last cumulative INR 2,998,668.80 |

Why RANK differs from ROW_NUMBER: when customers in a city tie on total spend, RANK gives them the same rank (and skips the next), while ROW_NUMBER still numbers them 1, 2, 3… in arbitrary order, so the two diverge from the first tie onward.

### pandas cross-check and exploratory analysis
`pd.pivot_table(..., margins=True)` grand total = INR 2,998,668.80, equal to the SQL total.

| City | Sum (INR) | Mean (INR) | Count |
|---|---|---|---|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

Completed order_amount: mean 269.91, median 135.00, skewness 4.61, Q1 64.80, Q3 297.00, IQR 232.20, fences −283.50 / 645.30, outliers 0 below and 897 above.
Correlations (completed orders): order_amount–discount_pct −0.095, order_amount–quantity +0.087, all others near 0.

**What is this data about?** Six months (Apr–Sep 2025) of quick-commerce orders: 11,110 completed orders across six cities, five product categories and INR 2.99 million revenue.
**Distributions?** Order value is strongly right-skewed (mean about twice the median, skew 4.61); most baskets are small.
**Outliers?** 897 orders exceed INR 645.30; they are mostly electronics, genuine big baskets rather than errors. Nothing falls below the lower fence.
**Relationships?** The largest |r| is order_amount with discount_pct (−0.095), barely larger than with quantity (+0.087). Correlations this small are weak, and correlation does not establish causation.

## Part B

### RFM table
Snapshot 2025-10-01 00:00:00, completed orders only: **1,175 customers**. Recency median 19 days (min 0, max 180); frequency mean 9.46 (max 33); monetary mean INR 2,552.06 (max INR 14,766.55).

### K-means
Features standardised with `StandardScaler`; `KMeans(n_clusters=K, n_init=10, random_state=42)`.

| K | Inertia | Silhouette |
|---|---|---|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

**Choice: K = 4.** The inertia drops sharply up to K=4 (1268 → 743) and much more slowly afterwards (743 → 617), so the elbow is at 4; K=4 also has the highest silhouette (0.526); and four segments are few enough for the business to run four distinct campaigns.
Inertia always falls as K grows (K=8 is lower than K=2 because more centres always sit closer to the points, and with K = number of customers it would be 0), so a low inertia alone cannot pick K: it must be read together with the elbow, silhouette and interpretability.

### Segment profile (K = 4)
Cluster numbers are only labels and carry no order.

| segment_id | Name | Size | Mean recency (days) | Mean frequency | Mean monetary (INR) | Recommended action |
|---|---|---|---|---|---|---|
| 0 | Occasional Buyers | 436 | 28.86 | 3.77 | 1,057.62 | Basket-building bundles and category nudges |
| 1 | **Premium High-Value** | 138 | 24.12 | 6.77 | 7,129.70 | Early access, electronics bundles, priority support |
| 2 | Loyal Regulars | 419 | 8.77 | 19.40 | 3,410.02 | Loyalty points and repeat-order subscriptions |
| 3 | Lapsing / Low-Engagement | 182 | 109.04 | 2.20 | 685.99 | Win-back coupon before they are lost |

Premium = the segment with the highest mean monetary value (segment 1). PCA on the scaled RFM matrix: explained variance ratio 0.634 and 0.218, total 0.852. Output: `outputs/customer_segments.csv`.

### Lapse problem
Cutoff 2025-09-01. Cohort = customers with at least one completed order before the cutoff: **1,156**, of whom **443 lapsed (38.32%)** (no completed order in September 2025). Split `train_test_split(test_size=0.2, stratify=y, random_state=42)`: 924 train / 232 test.
Features (all from orders dated before 2025-09-01 plus the customer table): recency, frequency, monetary, average discount, cancellation rate, average delivery minutes, favourite category, city, age, gender, tenure in days. Missing values in the feature columns: 0.
Baseline "always not lapsed" on the test set: accuracy 61.6%, lapsed-class F1 0.00.

**Leakage controls.** No order dated on or after 2025-09-01 entered any feature (September orders are used only to build the label). The Part B segment labels were not used as features. Encoding (`OneHotEncoder(drop="first")`) and scaling live inside a scikit-learn `Pipeline`, so they are fitted on training rows only (and, inside GridSearchCV, only on each fold's training part); the oversampler also saw only training rows. No encoder, scaler or resampler saw test rows, and the test set was used only after tuning.
The scaler's learned mean of frequency is 8.1147 against a full-cohort mean of 8.0285: they differ because the scaler is fitted on the 924 training rows only, not on all 1,156.

### Tuning and test results
| Model | Grid | Best parameters | Best CV F1 |
|---|---|---|---|
| Logistic regression | C ∈ {0.01, 0.1, 1, 10, 100}, cv=5 | C = 100 | 0.651 |
| Random forest (100 trees, seed 42) | max_depth ∈ {3,5,8} × min_samples_leaf ∈ {5,10,20}, cv=5 (45 fits) | depth 8, leaf 10 | 0.681 |

The best C sits at the edge of the grid; the reference reports C = 1 because my feature set is somewhat richer, and the grid is fixed by the brief.

Untouched test set, lapsed class:

| Model | TN | FP | FN | TP | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| Logistic regression | 111 | 32 | 31 | 58 | 0.644 | 0.652 | 0.648 |
| Random forest | 113 | 30 | 31 | 58 | 0.659 | 0.652 | 0.655 |

**Selected model: random forest** (rule fixed in advance: higher cross-validated F1, 0.681 vs 0.651; test F1 0.655, above the 0.55 requirement). Missing a lapsing customer means losing all of their future orders, while wrongly sending a retention offer to a loyal customer costs only a small discount or message, so recall matters more than precision here; both models have equal recall (0.652) and the forest has slightly better precision and F1, which is why it is preferred.

### Class-balance experiment
Over-sampling the training set only (570 lapsed + 570 not lapsed; random_state=42), refitting logistic regression (C=100), original test set:

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Un-balanced | 0.644 | 0.652 | 0.648 |
| Balanced | 0.589 | 0.854 | 0.697 |

Recall moved most (+0.20); precision did not rise (it fell to 0.589). The retention team catches 76 of 89 lapsers instead of 58, at the price of 53 instead of 32 wrongly targeted loyal customers; this is worth it if a retention offer is cheap relative to the value of a kept customer.

### Interpretation, persistence, retention list
Top five random-forest features: frequency 0.280, monetary 0.271, recency 0.115, avg_delivery 0.061, cancel_rate 0.053: how often and how much people buy matters most, ahead of delivery experience and cancellations.
Logistic coefficients (standardised): frequency −1.70 (more orders → less likely to lapse), recency +0.04 (almost no effect once frequency is known), monetary +0.28 (positive only after holding frequency fixed, because frequency and monetary are correlated, so it should not be read on its own).
The model is saved with `joblib.dump` to `outputs/lapse_model.joblib`; the reloaded model reproduces the test predictions and probabilities exactly (checked in code).
`outputs/retention_list.csv`: of 40 Premium customers in the test set, the top 10 by predicted lapse probability were written; **1 of the 10 truly lapsed**. This list is illustrative and not a leakage-free evaluation, because the Premium flag comes from whole-window RFM that includes the September window defining the label.

## Part C

`src/c1_triage.py` runs the whole pipeline over `data/raw/tickets.csv` (read with `keep_default_na=False`), in file order, through the supplied `mock_llm.py`. `mock_llm.reset()` is called immediately before each full run, and the random seed for the retry jitter is fixed (42).

### Input guardrails and redaction
Before any model call: an empty or whitespace-only ticket gets `skipped_empty`; a ticket with `est_tokens(text) > 400` gets `flagged_long`. Neither is ever sent to the model. Found: **4 empty, 4 long** (each long ticket estimates to 734 tokens).

Redaction is applied in this order with these exact patterns: email `[\w.+-]+@[\w-]+\.[\w.]+` → `[EMAIL]`; phone `\b[6-9]\d{9}\b` → `[PHONE]`; order id `ORD-\d{6}` → `[ORDER_ID]`. The user message is exactly the redacted text.

| Type | Substitutions (all 164 tickets) |
|---|---|
| Email | 55 |
| Phone | 91 |
| Order id | 132 |
| **Total** | **278** |

Re-applying the three patterns to the redacted texts finds **0** matches. In addition, every call goes through a function that asserts that no pattern matches any message (system or user) before it is sent, so a leak would stop the run.

### Prompts
Both contain the word `CLASSIFY`, the four allowed categories and a JSON-only reply format `{"category": "...", "reason": "..."}`. The zero-shot prompt (33 estimated tokens) has no `EXAMPLES:` line. The few-shot prompt (137 estimated tokens) adds an `EXAMPLES:` line with four labelled examples (billing, delivery, technical, fraud), written with placeholders only.

### Retries
Every model call is wrapped in a retry function: at most 3 retries per ticket, wait before retry n = `0.05 × 2**n × random.uniform(0.9, 1.15)` seconds, one log line per retry, for example:
```
retry ticket=T0002 retry_number=0 wait=0.0530s
retry ticket=T0116 retry_number=0 wait=0.0453s
```
A full run over the 156 eligible tickets produced **19 TransientLLMErrors, all recovered by a retry, 0 permanent errors**. All retries were retry number 0, with waits between 0.0453 s and 0.0562 s (allowed range 0.045–0.0575 s; checked by an assertion).

### Output validation
A reply is parsed as JSON; if `category` is one of the four values → `classified_json`. If it is not valid JSON, the lowercase words are extracted with `re.findall(r"[a-z]+", text.lower())` and the reply is accepted as `classified_keyword` only if exactly one allowed category word is present (so `billing-related` yields `billing`). Anything else, including valid JSON with a category outside the list (the mock returns `shipping`), is `flagged_invalid_output` and goes to human review; nothing is guessed.
One full run over the 156 eligible tickets: **135 classified_json, 12 classified_keyword, 9 flagged_invalid_output** (the same in both prompts).

### Zero-shot versus few-shot (full runs, no budget)
| Run | Tickets classified | Accuracy | Total tokens | Tokens per classified ticket |
|---|---|---|---|---|
| Zero-shot | 147 | 85.7% (126 / 147) | 9,556 | 65.0 |
| Few-shot | 147 | 100.0% (147 / 147) | 25,780 | 175.4 |

Total tokens are the sum of `usage.total_tokens` over all 156 calls, including the 9 flagged replies, which were still paid for.

Zero-shot confusion table (rows true, columns predicted):

| true \ predicted | billing | delivery | fraud | technical |
|---|---|---|---|---|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| fraud | 11 | 0 | 23 | 0 |
| technical | 10 | 0 | 0 | 29 |

**When the few-shot tokens pay off.** The examples cost about 110 extra tokens per classified ticket (175.4 vs 65.0), 16,224 tokens more over the run, but they remove all 21 errors. 11 of those errors were fraud tickets sent to the billing queue, where they would wait for the 24-hour queue instead of reaching security within 2 hours, so when a wrong class has a real cost the extra tokens are clearly worth paying. For a low-stakes task or a very large volume where an error is cheap, the zero-shot prompt at roughly a third of the cost would be enough.

### Budget, routing and outputs
`TOKEN_BUDGET`: before each model call, if the running total of `usage.total_tokens` is already ≥ the budget, no call is made and the ticket gets `budget_exhausted` (→ human review).
**Demo** (zero-shot, budget 6,000; `outputs/triage_budget_demo.csv`): the run stopped calling the model after 6,017 tokens (the check happens before a call, so the last call can overshoot) and **58 tickets ended `budget_exhausted`** (86 classified_json, 7 classified_keyword, 5 flagged_invalid_output, 4 empty, 4 long).

**Final output** `outputs/triage_results.csv` (164 rows; few-shot, budget 100,000, nothing exhausted). Routing, reading `outputs/customer_segments.csv`: skipped_empty / flagged_long / flagged_invalid_output / budget_exhausted → `human_review`, NORMAL; else fraud → `security_team`, HIGH; else premium customer → `premium_desk`, HIGH; else `support_queue`, NORMAL. A customer with no row in the segment file is treated as not premium.

| Route | Priority | Tickets |
|---|---|---|
| human_review | NORMAL | 17 (4 empty, 4 long, 9 invalid output) |
| security_team | HIGH | 34 |
| premium_desk | HIGH | 11 |
| support_queue | NORMAL | 102 |

**What was rule-validated and what went to humans.** 147 tickets (135 by JSON, 12 by the keyword rule) passed the output rules and were routed automatically; 17 tickets went to human review (4 empty and 4 too long, which the model never saw, and 9 whose answer failed validation). Cost of the few-shot prompt: 175.4 tokens per classified ticket, against 65.0 for zero-shot.

## Part D

### Chunking
Each of the five policy files is split into 40-word chunks with 10 words of overlap (next chunk starts 30 words later; stop after the first chunk whose end reaches the last word); the source file name is kept with every chunk (`src/d1_rag.py`).

| Document | Chunks (overlap 10) | Chunks (overlap 0) |
|---|---|---|
| refund_policy.txt | 4 | 3 |
| delivery_policy.txt | 3 | 3 |
| escalation_policy.txt | 3 | 2 |
| payment_policy.txt | 3 | 3 |
| privacy_policy.txt | 2 | 2 |
| **Total** | **15** | **13** |

### Retrieval
`TfidfVectorizer()` is fitted on the chunk texts; a query is transformed with the same fitted vectoriser and chunks are ranked by cosine similarity; `retrieve(query, k=3)` returns dictionaries with `source`, `score`, `text`. For `refund policy late delivery` the top three chunks all come from `refund_policy.txt` (scores 0.272, 0.251, 0.245).
**TF and IDF.** Term frequency measures how often a word occurs in a chunk, and inverse document frequency down-weights words that occur in many chunks, so rare, distinctive words count more. A limitation: TF-IDF matches words, not meaning, so a question and a policy that say the same thing with different words (for example "money back" versus "refund") score poorly.

### Augment and answer
`answer(question)` retrieves three chunks and sends two messages to `mock_llm.chat`: a system message containing `POLICY ASSISTANT` (answer only from the context, otherwise say the information could not be found) and a user message of the form `CONTEXT:`, newline, the three chunks each preceded by `[<file name>]` on its own line, newline, `QUESTION:`, newline, the question. It returns the model text and the sources used.

### Retrieval and grounding evaluation (`data/raw/policy_eval.csv`)
| Setting | Chunks | Answerable: expected_doc in top 3 (of 12) | Answerable: expected_fact in answer (of 12) | Unanswerable: fallback "I could not find…" (of 2) |
|---|---|---|---|---|
| overlap 10 | 15 | 12 / 12 | 12 / 12 | 2 / 2 |
| overlap 0 | 13 | 12 / 12 | **10 / 12** | 2 / 2 |

With zero overlap two questions break because the answer sentence is cut at a chunk boundary and the model only sees half of it: *"What delivery time makes a completed order eligible for a full refund?"* (the phrase "more than 45 minutes" ends up in a different chunk from "full refund of the order amount", so the answer misses the fact) and *"How soon is money returned when a payment fails but is deducted?"* ("within 48 hours" is split from "returned automatically"). The right document is still retrieved; the overlap is what keeps the whole sentence in one chunk.

### Tools
Five tools in `TOOLS_SPEC` (name, one-sentence description, JSON-style parameters) with a `TOOL_MAP` from name to function (`src/d2_agent.py`): `get_order` (reads `data/instamart.db`; returns `{"error": "order not found"}` if missing), `search_policy` (returns the `retrieve` output under `results`), `calculate_refund` (100% of the amount if delivered in more than 45 minutes, else 0), `issue_refund` and `escalate_to_human` (each append one JSON line with timestamp and tool name to `outputs/refund_ledger.jsonl`). Two safety additions: `issue_refund` refuses amounts above INR 1000, and the ledger is idempotent (the same tool and order are never recorded twice), so a retried or resumed run cannot pay out or escalate the same order twice. The model names a tool and an arguments dictionary; the code looks the name up in `TOOL_MAP` and calls the function with those arguments.

### ReAct loop and the seven scenarios
`run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10)` starts with a system message containing `AGENT` and the user message, calls `chat` each iteration, and either takes the content as the final answer or appends the assistant tool-call message, runs the tool and appends the tool message. It returns status, final_answer, steps, llm_calls and tokens_used. Every run is appended to `outputs/agent_runs.jsonl` (10 lines: the 7 scenarios, the 2 stop-safely demos and the crash-and-resume run, each with a `label`).

| Scenario | Status | Refund (INR) | Model calls | Tokens used |
|---|---|---|---|---|
| S1 | REFUND | 128.25 | 5 | 1,027 |
| S2 | REFUND | 135.00 | 5 | 1,027 |
| S3 | ESCALATE | 1,999.00 (sent to a human, not paid) | 5 | 1,014 |
| S4 | DENIED | 0.00 | 4 | 709 |
| S5 | DENIED | 0.00 | 3 | 415 |
| S6 | NOT_FOUND | 0.00 | 2 | 124 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,156 |

Every refund amount equals `expected_refund_inr` within 0.01 (asserted in code); S1 takes 5 model calls; S7 completes all six calls under the default `token_limit=6000`. The ledger holds 3 lines (two issued refunds, one escalation).

### Stopping safely
Three limits, each returning the steps completed so far and a short human-handoff string in `final_answer`: (a) `max_iterations` model calls without a final answer → `FALLBACK_MAX_ITERATIONS`; (b) after each call `usage.total_tokens` is added to `tokens_used` and if it exceeds `token_limit` the run stops without executing the requested tool → `FALLBACK_TOKEN_LIMIT`; (c) at the top of each iteration, `time.monotonic()` elapsed ≥ `time_limit_seconds` → `FALLBACK_TIMEOUT`.

| Demonstration | Final status | Detail |
|---|---|---|
| S7 with default limits | FALLBACK_MAX_ITERATIONS | 6 calls, 3,156 tokens |
| S1 with `token_limit=200` | **FALLBACK_TOKEN_LIMIT** | stopped on call 3 at 432 tokens, 2 tools executed, requested tool not run |
| S1 with `time_limit_seconds=0` | **FALLBACK_TIMEOUT** | stopped before the first model call (0 calls) |

### Resumable runs
After every tool observation a checkpoint `state/<run_id>.json` is written (run id, scenario id, full message list, tokens_used, step number, tools executed, plus steps, call count and limits). `resume_agent(run_id)` loads it and continues. Demonstration on S1: a simulated crash was raised right after the second tool's checkpoint, then `resume_agent("S1-crash")` finished the run with status REFUND and INR 128.25.
**Tools executed across the crash and the resume, in order:** `get_order`, `search_policy` (before the crash), `calculate_refund`, `issue_refund` (after the resume): four executions, none repeated. The resumed `issue_refund` found the order already in the ledger from the earlier normal S1 run and therefore did not write a second line.

### ReAct versus plan-and-execute
ReAct asks the model for one step at a time and sees every observation. In my runs this cost 5 model calls and 1,027 tokens for S1, 4 calls and 709 tokens for S4, 3 calls and 415 tokens for S5, and only 2 calls and 124 tokens for S6. Tokens grow faster than calls because the whole history, including the long policy-search result, is re-sent with every call. A plan-and-execute design would ask for the full step list once (about 1 planning call plus at most 1 summary call, an estimate I did not measure), so it is cheaper on the normal path and sends the history only once. Its weakness is the unexpected observation: the plan is written before the order is seen, so for a NOT_FOUND order (S6) or a cancelled order (S5) the later steps (policy search, refund calculation) are already planned and must be cancelled, or the run must detect the error and replan, which costs an extra call and tokens. ReAct simply stops: S6 ended after 2 calls and S5 after the policy search. For this refund workflow I would choose ReAct: what happens next depends on what each observation says (status, delay, amount above INR 1000), money moves at the end, the path is at most 5 steps so the extra tokens are small, and the iteration, token and time limits bound the loop (S7). A fixed plan with deterministic guards would only be worth it at much higher volumes.

## Limitations and Next Steps
- (Part B) The Premium flag is a post-hoc whole-window label (its RFM includes September, the window that defines the lapse label), so the retention list is illustrative and not a leakage-free evaluation.
- Part B uses one train/test split; the test F1 estimate for 232 rows is noisy.
- (Part C) The mock classifier is keyword-based, so the 100% few-shot accuracy reflects the mock and synthetic templates, not what a real model would score. Token counts are estimates (`est_tokens`), not billing figures. Premium routing depends on the Part B whole-window segments.
- (Part D) The policy QA and agent run on a deterministic mock, so retrieval quality and tool choice are not evidence of how a real model would behave; TF-IDF cannot match synonyms; the 7-day refund window is not checked (the brief treats all requests as in-window); the ledger is a local file, not a payment system.
- (A) Median/group imputation of ages and delivery times adds assumed values; the cleaned data is synthetic, so the findings do not describe real customers.
- Next steps: replace TF-IDF with embeddings plus a reranker, add a real time-window check and customer identity check to the refund tools, evaluate lapse models with a time-based split and re-compute Premium without September data, and add unit tests for each cleaning rule.
