"""Part A, task 5: assert the reference checkpoints on the clean tables and the database."""
import sqlite3
import pandas as pd

cust = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"])
orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
con = sqlite3.connect("data/instamart.db")
tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
comp = orders[orders.status == "completed"]

checks = {
    "db has the three tables": {"customers", "products", "orders"} <= tables,
    "12,309 clean orders": len(orders) == 12309,
    "one row per order_id": orders.order_id.is_unique,
    "11,110 completed": (orders.status == "completed").sum() == 11110,
    "1,199 cancelled": (orders.status == "cancelled").sum() == 1199,
    "no pending rows": (orders.status == "pending").sum() == 0,
    "0 missing order_amount": orders.order_amount.isna().sum() == 0,
    "0 missing delivery (completed)": comp.delivery_minutes.isna().sum() == 0,
    "1,199 missing delivery (cancelled)": orders[orders.status == "cancelled"].delivery_minutes.isna().sum() == 1199,
    "no negative delivery": (orders.delivery_minutes.dropna() < 0).sum() == 0,
    "completed revenue 2,998,668.80": round(comp.order_amount.sum(), 2) == 2998668.80,
    "six cities": cust.city.nunique() == 6,
    "city counts": cust.city.value_counts().to_dict() == {"Pune": 227, "Delhi": 210, "Bengaluru": 197, "Chennai": 191, "Hyderabad": 191, "Mumbai": 184},
    "gender counts": cust.gender.value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30},
    "0 missing ages, all 18-60": cust.age.isna().sum() == 0 and cust.age.between(18, 60).all(),
    "db orders row count": con.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 12309,
}
for name, ok in checks.items():
    print(("PASS  " if ok else "FAIL  ") + name)
assert all(checks.values()), "a checkpoint failed"
print("ALL CHECKPOINTS PASSED")
