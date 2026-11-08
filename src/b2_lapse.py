"""Part B, tasks 4-8: lapse problem, leakage-free preprocessing, tuning, balance experiment, persistence."""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CUTOFF, END = pd.Timestamp("2025-09-01"), pd.Timestamp("2025-10-01")
orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
cust = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"])
prod = pd.read_csv("data/clean/products.csv")
orders = orders.merge(prod[["product_id", "category"]], on="product_id")

# ---------- Task 4: cohort, label, features (only orders BEFORE the cutoff feed features) ----------
before = orders[orders.order_ts < CUTOFF]
comp_b = before[before.status == "completed"]
cohort_ids = comp_b.customer_id.unique()

# label: uses September orders, never as a feature
sept = orders[(orders.order_ts >= CUTOFF) & (orders.order_ts < END) & (orders.status == "completed")]
active = set(sept.customer_id)

feat = comp_b.groupby("customer_id").agg(last_ts=("order_ts", "max"), frequency=("order_id", "nunique"),
                                         monetary=("order_amount", "sum"), avg_discount=("discount_pct", "mean"),
                                         avg_delivery=("delivery_minutes", "mean"))
feat["recency"] = (CUTOFF - feat["last_ts"]).dt.days
# cancellation rate: cancelled / all final-status orders before the cutoff
canc = before.groupby("customer_id")["status"].apply(lambda s: (s == "cancelled").mean()).rename("cancel_rate")
# favourite category: most frequent among completed orders; ties broken alphabetically (deterministic)
fav = (comp_b.groupby(["customer_id", "category"]).size().reset_index(name="n")
       .sort_values(["customer_id", "n", "category"], ascending=[True, False, True])
       .drop_duplicates("customer_id").set_index("customer_id")["category"].rename("fav_category"))
df = feat.join(canc).join(fav).reset_index().merge(cust, on="customer_id")
df["tenure_days"] = (CUTOFF - df["signup_date"]).dt.days
df["lapsed"] = (~df.customer_id.isin(active)).astype(int)
df = df.set_index("customer_id")

NUM = ["recency", "frequency", "monetary", "avg_discount", "cancel_rate", "avg_delivery", "age", "tenure_days"]
CAT = ["city", "gender", "fav_category"]
X, y = df[NUM + CAT], df["lapsed"]
assert not X.isna().any().any(), "missing value in features"
print(f"cohort {len(df)} | lapsed {y.sum()} ({y.mean()*100:.2f}%)")
print("missing values in feature columns:", int(X.isna().sum().sum()))

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(f"train {len(Xtr)} | test {len(Xte)}")
base_pred = np.zeros(len(yte), dtype=int)
print(f"baseline (always not lapsed): accuracy {accuracy_score(yte, base_pred)*100:.1f}% "
      f"F1 {f1_score(yte, base_pred, zero_division=0):.2f}")

# ---------- Task 5: preprocessing inside the pipeline (fitted on training rows only) ----------
def make_pipe(clf):
    pre = ColumnTransformer([("num", StandardScaler(), NUM),
                             ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT)])
    return Pipeline([("pre", pre), ("clf", clf)])

probe = make_pipe(LogisticRegression(max_iter=1000)).fit(Xtr, ytr)
scaler = probe.named_steps["pre"].named_transformers_["num"]
learned = scaler.mean_[NUM.index("frequency")]
print(f"scaler mean of frequency (train only) {learned:.4f} | full-cohort mean {df.frequency.mean():.4f}")

# ---------- Task 6: tuning (preprocessing is refitted inside each CV fold) ----------
def report(name, model, Xt, yt):
    p = model.predict(Xt)
    cm = confusion_matrix(yt, p)
    pr, rc, f1 = (precision_score(yt, p, zero_division=0), recall_score(yt, p, zero_division=0),
                  f1_score(yt, p, zero_division=0))
    print(f"{name}: confusion [[TN {cm[0,0]}, FP {cm[0,1]}], [FN {cm[1,0]}, TP {cm[1,1]}]] "
          f"precision {pr:.3f} recall {rc:.3f} F1 {f1:.3f}")
    return pr, rc, f1

lr_gs = GridSearchCV(make_pipe(LogisticRegression(max_iter=1000)), {"clf__C": [0.01, 0.1, 1, 10, 100]},
                     cv=5, scoring="f1").fit(Xtr, ytr)
rf_gs = GridSearchCV(make_pipe(RandomForestClassifier(n_estimators=100, random_state=42)),
                     {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 10, 20]},
                     cv=5, scoring="f1").fit(Xtr, ytr)
print("\nLR best", lr_gs.best_params_, "CV F1", round(lr_gs.best_score_, 3))
print("RF best", rf_gs.best_params_, "CV F1", round(rf_gs.best_score_, 3))

# test set is touched only now, after tuning
print("\nTest-set evaluation (lapsed = positive class):")
lr_m = report("Logistic regression", lr_gs.best_estimator_, Xte, yte)
rf_m = report("Random forest      ", rf_gs.best_estimator_, Xte, yte)

# selection rule fixed in advance: the higher best cross-validated F1 (no peeking at test)
sel_name, sel_model = (("Random forest", rf_gs.best_estimator_) if rf_gs.best_score_ >= lr_gs.best_score_
                       else ("Logistic regression", lr_gs.best_estimator_))
sel_f1 = f1_score(yte, sel_model.predict(Xte), zero_division=0)
print(f"Selected: {sel_name} (higher CV F1); test F1 {sel_f1:.3f}")
assert sel_f1 >= 0.55, "selected model below required test F1"

# ---------- Task 7: over-sample the training set only ----------
tr = Xtr.assign(lapsed=ytr)
minority = tr[tr.lapsed == 1]
extra = minority.sample(n=(tr.lapsed == 0).sum() - len(minority), replace=True, random_state=42)
bal = pd.concat([tr, extra])
print(f"\nbalanced train: {bal.lapsed.value_counts().to_dict()}")
best_C = lr_gs.best_params_["clf__C"]
lr_bal = make_pipe(LogisticRegression(max_iter=1000, C=best_C)).fit(bal[NUM + CAT], bal["lapsed"])
print("Original test set, logistic regression:")
report("  un-balanced", lr_gs.best_estimator_, Xte, yte)
report("  balanced   ", lr_bal, Xte, yte)

# ---------- Task 8: interpretation, persistence, retention list ----------
rf_best = rf_gs.best_estimator_
names = rf_best.named_steps["pre"].get_feature_names_out()
imp = pd.Series(rf_best.named_steps["clf"].feature_importances_, index=names).sort_values(ascending=False)
print("\nTop 5 random-forest importances:\n", imp.head(5).round(4).to_string())
lr_names = lr_gs.best_estimator_.named_steps["pre"].get_feature_names_out()
coef = pd.Series(lr_gs.best_estimator_.named_steps["clf"].coef_[0], index=lr_names)
print("\nLogistic coefficients (standardised numerics):\n",
      coef[["num__recency", "num__frequency", "num__monetary"]].round(3).to_string())

joblib.dump(sel_model, "outputs/lapse_model.joblib")
reloaded = joblib.load("outputs/lapse_model.joblib")
same = np.array_equal(sel_model.predict(Xte), reloaded.predict(Xte)) and \
    np.allclose(sel_model.predict_proba(Xte), reloaded.predict_proba(Xte))
print("\nreloaded model reproduces test predictions exactly:", same)
assert same

seg = pd.read_csv("outputs/customer_segments.csv").set_index("customer_id")
test = pd.DataFrame({"lapse_probability": sel_model.predict_proba(Xte)[:, 1], "actually_lapsed": yte.values},
                    index=Xte.index)
prem = test[test.index.map(lambda c: bool(seg["is_premium"].get(c, False)))]
top = prem.sort_values("lapse_probability", ascending=False).head(10).round(4)
top.reset_index().rename(columns={"index": "customer_id"})[["customer_id", "lapse_probability", "actually_lapsed"]] \
   .to_csv("outputs/retention_list.csv", index=False)
print(f"premium customers in test set: {len(prem)} | retention list rows {len(top)} | "
      f"truly lapsed among them {int(top.actually_lapsed.sum())}")
