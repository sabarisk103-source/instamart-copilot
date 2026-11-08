"""Part A, task 7: pandas cross-check of Q2 and exploratory statistics."""
import pandas as pd

orders = pd.read_csv("data/clean/orders.csv")
cust = pd.read_csv("data/clean/customers.csv")
prod = pd.read_csv("data/clean/products.csv")
comp = orders[orders.status == "completed"]
df = comp.merge(cust[["customer_id", "city"]], on="customer_id").merge(prod[["product_id", "category"]], on="product_id")

# (a) pivot cross-check of Q2
pv = pd.pivot_table(df, values="order_amount", index="city", columns="category", aggfunc="sum", margins=True)
print("Pivot grand total:", round(pv.loc["All", "All"], 2))
print(pv.round(2).to_string())

per_city = df.groupby("city").agg(total=("order_amount", "sum"), mean=("order_amount", "mean"),
                                  count=("order_amount", "count")).round(2)
print("\nPer-city sum / mean / count:\n", per_city.to_string())

# (b) distribution of completed order_amount
a = comp.order_amount
q1, q2, q3 = a.quantile([.25, .5, .75])
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print(f"\nmean {a.mean():.2f} | median {q2:.2f} | skew {a.skew():.2f}")
print(f"Q1 {q1:.2f} | Q3 {q3:.2f} | IQR {iqr:.2f} | fences {lo:.2f} / {hi:.2f}")
print(f"outliers below {int((a < lo).sum())} | above {int((a > hi).sum())}")

corr = comp[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr()
print("\nCorrelation matrix:\n", corr.round(3).to_string())
cols = list(corr.columns)
pairs = {(a_, b_): corr.loc[a_, b_] for k, a_ in enumerate(cols) for b_ in cols[k + 1:]}
best = max(pairs, key=lambda p: abs(pairs[p]))
print("Largest |r| pair:", best, round(pairs[best], 3))
