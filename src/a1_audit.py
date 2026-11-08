"""Part A, task 2: audit the raw exports (no changes are made to any file)."""
import pandas as pd

RAW = "data/raw/"
customers = pd.read_csv(RAW + "customers.csv")
products = pd.read_csv(RAW + "products.csv")
# only truly empty cells are NaN; the text "N/A" must stay a string so the audit sees it
orders = pd.read_csv(RAW + "orders.csv", keep_default_na=False, na_values=[""])

for name, df in [("customers", customers), ("products", products), ("orders", orders)]:
    print(f"\n=== {name}: shape {df.shape} ===")
    audit = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2),
    })
    print(audit.to_string())

amount_num = pd.to_numeric(orders["order_amount"], errors="coerce")
# non-numeric = present in the file but cannot be converted
non_numeric = int(amount_num.isna().sum())
age_num = pd.to_numeric(customers["age"], errors="coerce")

figures = {
    "raw orders rows": len(orders),
    "exact duplicate rows": int(orders.duplicated().sum()),
    "rows with status pending": int((orders["status"] == "pending").sum()),
    "distinct order_id": orders["order_id"].nunique(),
    "non-numeric order_amount": non_numeric,
    "negative delivery_minutes": int((orders["delivery_minutes"] < 0).sum()),
    "blank delivery_minutes": int(orders["delivery_minutes"].isna().sum()),
    "blank age": int(customers["age"].isna().sum()),
    "age outside 10-100": int(((age_num < 10) | (age_num > 100)).sum()),
    "distinct raw city spellings": customers["city"].nunique(),
    "distinct non-blank gender spellings": customers["gender"].dropna().nunique(),
    "blank gender": int(customers["gender"].isna().sum()),
}
print("\n=== Audit figures ===")
for k, v in figures.items():
    print(f"{k:40s} {v}")
