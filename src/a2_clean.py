"""Part A, tasks 3-5: clean customers and orders, write CSVs and the SQLite database."""
import os, sqlite3
import pandas as pd

RAW, CLEAN, DB = "data/raw/", "data/clean/", "data/instamart.db"
os.makedirs(CLEAN, exist_ok=True)
log = {}   # rule -> number of rows / values changed (copied into the README)

customers = pd.read_csv(RAW + "customers.csv")
products = pd.read_csv(RAW + "products.csv")
# only truly empty cells are NaN; the text "N/A" must stay a string so the audit sees it
orders = pd.read_csv(RAW + "orders.csv", keep_default_na=False, na_values=[""])

# ---------------- customers ----------------
# (a) city: strip + title case
before = customers["city"].copy()
customers["city"] = customers["city"].str.strip().str.title()
log["customers: city standardised (values changed)"] = int((before != customers["city"]).sum())

# (b) gender: map every spelling to male / female / unknown
gmap = {"m": "male", "male": "male", "f": "female", "female": "female"}
g = customers["gender"].fillna("").str.strip().str.lower().map(gmap)
log["customers: gender blank -> unknown"] = int(g.isna().sum())
customers["gender"] = g.fillna("unknown")

# (c) age: outside 10-100 -> missing, then median of the customer's city
age = pd.to_numeric(customers["age"], errors="coerce")
bad = (age < 10) | (age > 100)
log["customers: impossible ages set to missing"] = int(bad.sum())
age = age.mask(bad)
log["customers: missing ages filled with city median"] = int(age.isna().sum())
customers["age"] = age.fillna(age.groupby(customers["city"]).transform("median")).round().astype(int)

# (d) signup_date -> datetime
customers["signup_date"] = pd.to_datetime(customers["signup_date"])

# ---------------- orders ----------------
n0 = len(orders)
# 1. order_ts -> datetime
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# 2. exact duplicates
orders = orders.drop_duplicates()
log["orders: exact duplicate rows removed"] = n0 - len(orders)

# 3. one record per order_id: the one with the latest order_ts
n1 = len(orders)
orders = (orders.sort_values(["order_id", "order_ts"])
                .drop_duplicates("order_id", keep="last")
                .reset_index(drop=True))
log["orders: earlier (pending) records removed"] = n1 - len(orders)

# 4. order_amount: numeric, unparseable -> missing -> recompute from price
orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
miss = orders["order_amount"].isna()
log["orders: unparseable order_amount refilled from price"] = int(miss.sum())
price = orders["product_id"].map(products.set_index("product_id")["price_inr"])
calc = (price * orders["quantity"] * (1 - orders["discount_pct"] / 100)).round(2)
orders["order_amount"] = orders["order_amount"].fillna(calc)

# 5. delivery_minutes
neg = orders["delivery_minutes"] < 0
log["orders: negative delivery_minutes set to missing"] = int(neg.sum())
orders.loc[neg, "delivery_minutes"] = float("nan")
orders["category"] = orders["product_id"].map(products.set_index("product_id")["category"])
comp = orders["status"] == "completed"
cat_median = (orders["delivery_minutes"].where(comp)
              .groupby(orders["category"]).transform("median"))
fill_mask = comp & orders["delivery_minutes"].isna()
log["orders: completed, missing delivery_minutes filled with category median"] = int(fill_mask.sum())
orders.loc[fill_mask, "delivery_minutes"] = cat_median[fill_mask]
orders = orders.drop(columns="category")

# ---------------- store ----------------
orders_out = orders.copy()
orders_out["order_ts"] = orders_out["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
cust_out = customers.copy()
cust_out["signup_date"] = cust_out["signup_date"].dt.strftime("%Y-%m-%d")

cust_out.to_csv(CLEAN + "customers.csv", index=False)
products.to_csv(CLEAN + "products.csv", index=False)
orders_out.to_csv(CLEAN + "orders.csv", index=False)

if os.path.exists(DB):
    os.remove(DB)
with sqlite3.connect(DB) as con:
    cust_out.to_sql("customers", con, index=False)
    products.to_sql("products", con, index=False)
    orders_out.to_sql("orders", con, index=False)

print("Cleaning log:")
for k, v in log.items():
    print(f"  {k}: {v}")
print("Clean orders:", len(orders), "| status counts:", orders["status"].value_counts().to_dict())
