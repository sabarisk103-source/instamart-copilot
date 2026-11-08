"""Part B, tasks 1-3: RFM table, K-means selection, segment profile, PCA, customer_segments.csv."""
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

os.makedirs("outputs", exist_ok=True)
SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
comp = orders[orders.status == "completed"]

# ---- Task 1: RFM ----
rfm = comp.groupby("customer_id").agg(last_ts=("order_ts", "max"),
                                      frequency=("order_id", "nunique"),
                                      monetary=("order_amount", "sum"))
rfm["recency"] = (SNAPSHOT - rfm["last_ts"]).dt.days
rfm = rfm[["recency", "frequency", "monetary"]].reset_index()
print(f"RFM customers: {len(rfm)}")
print(f"recency median {rfm.recency.median():.0f} min {rfm.recency.min()} max {rfm.recency.max()}")
print(f"frequency mean {rfm.frequency.mean():.2f} max {rfm.frequency.max()}")
print(f"monetary mean {rfm.monetary.mean():.2f} max {rfm.monetary.max():.2f}")

# ---- Task 2: K-means sweep ----
X = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])
rows = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    rows.append((k, round(km.inertia_, 1), round(silhouette_score(X, km.labels_), 3)))
print("\nK  inertia  silhouette")
for k, i, s in rows:
    print(f"{k}  {i:7.1f}  {s:.3f}")

K = 4   # chosen: see README
km = KMeans(n_clusters=K, n_init=10, random_state=42).fit(X)
rfm["segment_id"] = km.labels_

# ---- Task 3: profile, naming, Premium flag ----
prof = rfm.groupby("segment_id").agg(size=("customer_id", "count"), recency=("recency", "mean"),
                                     frequency=("frequency", "mean"), monetary=("monetary", "mean")).round(2)
premium_id = prof["monetary"].idxmax()

# each non-premium segment must get a distinct name; fall back to explicit ranking by recency
others = prof.drop(index=premium_id)
by_freq = others.sort_values("frequency", ascending=False).index.tolist()
by_rec = others.sort_values("recency", ascending=False).index.tolist()
names = {premium_id: "Premium High-Value", by_freq[0]: "Loyal Regulars"}
rest = [s for s in others.index if s not in names]
stale = max(rest, key=lambda s: others.loc[s, "recency"])
names[stale] = "Lapsing / Low-Engagement"
for s in rest:
    names.setdefault(s, "Occasional Buyers")
prof["segment_name"] = prof.index.map(names)
prof["is_premium"] = prof.index == premium_id
print("\nSegment profile (cluster numbers are arbitrary labels, not an ordering):")
print(prof.to_string())

rfm["segment_name"] = rfm["segment_id"].map(names)
rfm["is_premium"] = rfm["segment_id"] == premium_id
rfm[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]] \
    .to_csv("outputs/customer_segments.csv", index=False)

pca = PCA(n_components=2).fit(X)
print("\nPCA explained_variance_ratio_:", pca.explained_variance_ratio_.round(3),
      "total", round(pca.explained_variance_ratio_.sum(), 3))
