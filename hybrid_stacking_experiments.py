"""Supplementary code (main results): nested 5-fold CV of six tuned base learners and the proposed
hybrid stacking framework (Tables IV, VI-XI; Figs 4-8).
Usage:  python3 hybrid_stacking_experiments.py brain_stroke.csv      -> res_hybrid.json
Requires: python>=3.11, numpy, pandas, scikit-learn, scipy, keras>=3 (JAX/TensorFlow/PyTorch backend)."""
import os, sys, json, time, warnings
os.environ["KERAS_BACKEND"] = "jax"; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, keras
from keras import layers
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix, matthews_corrcoef
from scipy.stats import norm

SEED = 42
CAT = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
CONT = ["age", "avg_glucose_level", "bmi"]; BIN = ["hypertension", "heart_disease"]
df = pd.read_csv(sys.argv[1]); y = df.stroke.values; X = df.drop(columns="stroke")

def enc_onehot():   # for LR, SVM, neural nets
    return ColumnTransformer([("c", StandardScaler(), CONT), ("b", "passthrough", BIN),
                              ("o", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)])
def enc_ord():      # for tree models
    return ColumnTransformer([("c", "passthrough", CONT + BIN), ("o", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT)])

def nn(n, lstm):
    keras.utils.set_random_seed(SEED)
    m = keras.Sequential([layers.Input((n, 1)), layers.Conv1D(64, 3, activation="relu"), layers.MaxPooling1D(2),
                          layers.Conv1D(128, 3, activation="relu")])
    if lstm: m.add(layers.LSTM(128)); m.add(layers.Dropout(0.3))
    else: m.add(layers.Flatten())
    m.add(layers.Dense(64, activation="relu")); m.add(layers.Dropout(0.3)); m.add(layers.Dense(1, activation="sigmoid"))
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    return m

def fit_nn(Xtr, ytr, Xte, lstm):
    # internal stratified 10% validation for early stopping (from training data only)
    rng = np.random.default_rng(SEED); idx = np.arange(len(ytr)); val = []
    for c in (0, 1):
        ic = idx[ytr == c]; val += list(rng.choice(ic, max(1, int(0.1 * len(ic))), replace=False))
    val = np.array(val); tr = np.setdiff1d(idx, val)
    w = {0: 1.0, 1: float((ytr[tr] == 0).sum() / (ytr[tr] == 1).sum())}
    m = nn(Xtr.shape[1], lstm)
    sw_val = np.where(ytr[val] == 1, w[1], 1.0)
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)
    m.fit(Xtr[tr, :, None], ytr[tr], validation_data=(Xtr[val, :, None], ytr[val], sw_val), epochs=100,
          batch_size=32, class_weight=w, callbacks=[es], verbose=0)
    return m.predict(Xte[:, :, None], verbose=0).ravel()

def base_models():
    return {
     "LR":  ("oh", lambda: GridSearchCV(LogisticRegression(class_weight="balanced", max_iter=2000),
                                        {"C": [0.01, 0.1, 1, 10]}, scoring="roc_auc", cv=3)),
     "SVM": ("oh", lambda: GridSearchCV(SVC(class_weight="balanced"), {"C": [0.1, 1, 10], "gamma": ["scale", 0.01, 0.1]},
                                        scoring="roc_auc", cv=3)),
     "RF":  ("ord", lambda: GridSearchCV(RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                        random_state=SEED, n_jobs=-1), {"max_depth": [4, 6, 8, None], "min_samples_leaf": [1, 5, 10]},
                                        scoring="roc_auc", cv=3)),
     "HGB": ("ord", lambda: GridSearchCV(HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED),
                                        {"max_depth": [2, 3, 4], "learning_rate": [0.03, 0.1], "max_iter": [100, 300]},
                                        scoring="roc_auc", cv=3)),
     "CNN": ("oh", "nn0"), "CNN-LSTM": ("oh", "nn1"),
    }

def score(est, Xte):
    return est.decision_function(Xte) if hasattr(est, "decision_function") and not hasattr(est, "predict_proba") \
        else (est.predict_proba(Xte)[:, 1] if hasattr(est, "predict_proba") else est.decision_function(Xte))

def fit_predict(name, kind, Xtr_df, ytr, Xte_df):
    e = (enc_onehot() if kind[0] == "oh" else enc_ord()).fit(Xtr_df)
    A, B = e.transform(Xtr_df).astype("float32"), e.transform(Xte_df).astype("float32")
    if kind[1] in ("nn0", "nn1"): return fit_nn(A, ytr, B, kind[1] == "nn1"), None
    g = kind[1](); g.fit(A, ytr); est = g.best_estimator_
    s = est.decision_function(B) if isinstance(est, SVC) else est.predict_proba(B)[:, 1]
    return s, g.best_params_

def youden(yv, s):
    f, t, th = roc_curve(yv, s); return th[np.argmax(t - f)]

def metrics(yt, s, thr):
    yh = (s >= thr).astype(int); tn, fp, fn, tp = confusion_matrix(yt, yh, labels=[0, 1]).ravel()
    R = tp / (tp + fn); S = tn / (tn + fp); Pp = tp / (tp + fp) if tp + fp else 0
    return dict(acc=(tp + tn) / len(yt), R=R, spec=S, P=Pp, F1=2 * Pp * R / (Pp + R) if Pp + R else 0, bacc=(R + S) / 2,
                mcc=matthews_corrcoef(yt, yh), auc=roc_auc_score(yt, s), auprc=average_precision_score(yt, s),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))

# ---- DeLong (fast, Sun & Xu 2014)
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1; i = j
    T2 = np.empty(N); T2[J] = T; return T2
def delong(y, s1, s2):
    pos, neg = s1[y == 1], s1[y == 0]; m, n = len(pos), y.size - len(pos)
    preds = np.vstack([s1, s2]); order = np.r_[np.where(y == 1)[0], np.where(y == 0)[0]]; P = preds[:, order]
    tx = np.array([_midrank(r[:m]) for r in P]); ty = np.array([_midrank(r[m:]) for r in P]); tz = np.array([_midrank(r) for r in P])
    aucs = tz[:, :m].sum(1) / m / n - (m + 1) / 2 / n
    v01 = (tz[:, :m] - tx) / n; v10 = 1 - (tz[:, m:] - ty) / m
    S = np.cov(v01) / m + np.cov(v10) / n; l = np.array([1, -1])
    z = (aucs[0] - aucs[1]) / np.sqrt(l @ S @ l); return float(aucs[0]), float(aucs[1]), float(2 * norm.sf(abs(z)))

NAMES = ["LR", "SVM", "RF", "HGB", "CNN", "CNN-LSTM"]; STACK_BASE = NAMES
outer = StratifiedKFold(5, shuffle=True, random_state=SEED)
oof = {k: np.zeros(len(y)) for k in NAMES + ["Proposed"]}; thr_res = {k: [] for k in oof}; params = {k: [] for k in NAMES}
t0 = time.time(); coefs = []; thr_used = {k: [] for k in NAMES + ["Proposed"]}
for f, (tr, te) in enumerate(outer.split(X, y)):
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y[tr], y[te]
    bm = base_models()
    # inner OOF scores (for thresholds and for the stacking meta-learner)
    inner = StratifiedKFold(3, shuffle=True, random_state=SEED); inner_oof = {k: np.zeros(len(ytr)) for k in NAMES}
    for a, b in inner.split(Xtr, ytr):
        for k in NAMES:
            inner_oof[k][b], _ = fit_predict(k, bm[k], Xtr.iloc[a], ytr[a], Xtr.iloc[b])
    # outer fit
    test_s = {}
    for k in NAMES:
        test_s[k], bp = fit_predict(k, bm[k], Xtr, ytr, Xte); params[k].append(bp)
    # stacking meta-learner on inner OOF (standardised base scores)
    Zi = np.column_stack([inner_oof[k] for k in STACK_BASE]); mu, sd = Zi.mean(0), Zi.std(0)
    meta = LogisticRegression(class_weight="balanced", max_iter=1000).fit((Zi - mu) / sd, ytr)
    Zt = np.column_stack([test_s[k] for k in STACK_BASE])
    test_s["Proposed"] = meta.predict_proba((Zt - mu) / sd)[:, 1]
    inner_oof["Proposed"] = meta.predict_proba((Zi - mu) / sd)[:, 1]   # in-sample for meta only; used for threshold
    coefs.append(meta.coef_.ravel().tolist())
    for k in oof:
        oof[k][te] = test_s[k]
        thr = youden(ytr, inner_oof[k]); thr_res[k].append(metrics(yte, test_s[k], thr))
        thr_used[k].append(float(thr))
    print(f"fold {f} done {time.time() - t0:.0f}s", {k: round(roc_auc_score(yte, test_s[k]), 3) for k in oof}, flush=True)

out = {"fold": {}, "pooled_auc": {}, "delong_vs_proposed": {}, "params": params, "meta_coef": None}
for k in oof:
    keys = thr_res[k][0].keys()
    out["fold"][k] = {m: [float(np.mean([r[m] for r in thr_res[k]])), float(np.std([r[m] for r in thr_res[k]]))] for m in keys}
    out["pooled_auc"][k] = float(roc_auc_score(y, oof[k]))
for k in NAMES:
    out["delong_vs_proposed"][k] = delong(y, oof["Proposed"], oof[k])
out["oof"] = {k: list(map(float, v)) for k, v in oof.items()}
out["meta_coef"] = coefs; out["thr"] = thr_used; out["y"] = list(map(int, y))
out["folds"] = [list(map(int, te)) for _, te in outer.split(X, y)]
out["cm"] = {k: [int(sum(r[c] for r in thr_res[k])) for c in ("tn", "fp", "fn", "tp")] for k in oof}
json.dump(out, open("res_hybrid.json", "w"), default=str)
for k in oof:
    r = out["fold"][k]; print(f"{k:9s}", " ".join(f"{m}={r[m][0]:.3f}±{r[m][1]:.3f}" for m in ["auc", "auprc", "bacc", "R", "spec", "acc", "mcc"]), f"pooledAUC={out['pooled_auc'][k]:.3f}")
print("DeLong proposed vs:", {k: [round(x, 4) for x in v] for k, v in out["delong_vs_proposed"].items()})
