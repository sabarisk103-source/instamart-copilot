"""Part D, tasks 1-4: chunking, TF-IDF retrieval, grounded answers, evaluation."""
import glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for mock_llm
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mock_llm

POLICY_DIR = os.path.join("data", "policies")
CHUNK_WORDS = 40

SYSTEM_PROMPT = ("You are a POLICY ASSISTANT for customer support. Answer only from the supplied context. "
                 "If the answer is not in the context, say you could not find the information.")

def chunk_documents(overlap=10, size=CHUNK_WORDS, folder=POLICY_DIR):
    """Chunks of `size` words; each next chunk starts size-overlap words later; stop after the
    first chunk whose end reaches or passes the last word."""
    chunks, step = [], size - overlap
    for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
        words = open(path, encoding="utf-8").read().split()
        start = 0
        while True:
            chunks.append({"source": os.path.basename(path), "text": " ".join(words[start:start + size])})
            if start + size >= len(words):
                break
            start += step
    return chunks

class PolicyIndex:
    def __init__(self, overlap=10):
        self.chunks = chunk_documents(overlap)
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query, k=3):
        q = self.vectorizer.transform([query])            # same fitted vectoriser as the chunks
        sims = cosine_similarity(q, self.matrix)[0]
        top = sims.argsort()[::-1][:k]
        return [{"source": self.chunks[i]["source"], "score": float(sims[i]), "text": self.chunks[i]["text"]} for i in top]

_default = None
def default_index():
    global _default
    if _default is None:
        _default = PolicyIndex(overlap=10)
    return _default

def retrieve(query, k=3, index=None):
    return (index or default_index()).retrieve(query, k)

def answer(question, index=None):
    hits = retrieve(question, 3, index)
    context = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}"}]
    resp = mock_llm.chat(messages)
    return resp.choices[0].message.content, [h["source"] for h in hits]

def evaluate(overlap, verbose=False):
    idx = PolicyIndex(overlap)
    qa = pd.read_csv("data/raw/policy_eval.csv", keep_default_na=False)
    doc_hit = fact_hit = fallback = 0
    n_ans = n_unans = 0
    failures = []
    for r in qa.itertuples():
        text, sources = answer(r.question, idx)
        if r.expected_doc == "NONE":
            n_unans += 1
            if text.startswith("I could not find"):
                fallback += 1
            else:
                failures.append((r.question, "should have fallen back", text))
        else:
            n_ans += 1
            d, f = r.expected_doc in sources, r.expected_fact in text
            doc_hit += d; fact_hit += f
            if not (d and f):
                failures.append((r.question, f"doc_in_top3={d}, fact_in_answer={f}", text))
        if verbose:
            print(f"  {r.question[:60]:60s} -> {sources} | {text[:70]}")
    return {"chunks": len(idx.chunks), "doc": f"{doc_hit}/{n_ans}", "fact": f"{fact_hit}/{n_ans}",
            "fallback": f"{fallback}/{n_unans}", "failures": failures}

if __name__ == "__main__":
    for ov in (10, 0):
        ch = chunk_documents(ov)
        per_doc = pd.Series([c["source"] for c in ch]).value_counts().sort_index().to_dict()
        print(f"overlap {ov}: {len(ch)} chunks {per_doc}")
    top = retrieve("refund policy late delivery")
    print("top 3 for 'refund policy late delivery':", [(h["source"], round(h["score"], 3)) for h in top])
    for ov in (10, 0):
        res = evaluate(ov, verbose=(ov == 10))
        print(f"\nEVAL overlap={ov}: chunks {res['chunks']} | expected_doc in top-3 {res['doc']} | "
              f"fact in answer {res['fact']} | fallback {res['fallback']}")
        for q, why, text in res["failures"]:
            print("  FAIL:", q, "|", why, "|", text)
