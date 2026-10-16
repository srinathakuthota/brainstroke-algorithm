"""Supplementary code (initial configuration, Table V and Table XI "Initial" column).
Usage:  python3 initial_configuration_experiments.py brain_stroke.csv <cv|cv_cw|perm>"""
import os, sys, json
os.environ["KERAS_BACKEND"] = "jax"
import numpy as np, pandas as pd, keras
from keras import layers
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, roc_auc_score,
                             average_precision_score, matthews_corrcoef, confusion_matrix, roc_curve)
from scipy.stats import chi2

SEED = 42
CAT = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
CONT = ["age", "avg_glucose_level", "bmi"]
df = pd.read_csv(sys.argv[1])
FEATS = [c for c in df.columns if c != "stroke"]

def prep(tr, te, leak=False):
    tr, te = tr.copy(), te.copy()
    for c in CAT:
        le = LabelEncoder().fit(tr[c]); m = {k: i for i, k in enumerate(le.classes_)}
        tr[c] = tr[c].map(m); te[c] = te[c].map(m).fillna(-1)
    for c in ["bmi"]:
        mu = tr[c].mean(); tr[c] = tr[c].fillna(mu); te[c] = te[c].fillna(mu)
    sc = StandardScaler().fit(tr[CONT]); tr[CONT] = sc.transform(tr[CONT]); te[CONT] = sc.transform(te[CONT])
    cols = FEATS + (["stroke"] if leak else [])
    return tr[cols].values.astype("float32"), te[cols].values.astype("float32")

def cnn(n, lstm=True):
    keras.utils.set_random_seed(SEED)
    m = keras.Sequential([layers.Input((n, 1)),
        layers.Conv1D(64, 3, activation="relu"), layers.MaxPooling1D(2),
        layers.Conv1D(128, 3, activation="relu", padding="same" if n < 8 else "valid")])
    if lstm: m.add(layers.LSTM(128)); m.add(layers.Dropout(0.3))
    else: m.add(layers.Flatten())
    m.add(layers.Dense(64, activation="relu")); m.add(layers.Dropout(0.3)); m.add(layers.Dense(1, activation="sigmoid"))
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return m

def fit_predict(name, Xtr, ytr, Xte, cw=None, hist=False):
    if name == "SVM":
        m = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=SEED,
                class_weight=cw).fit(Xtr, ytr); return 1 / (1 + np.exp(-m.decision_function(Xte))), None
    if name == "RF":
        m = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1,
                                   class_weight=cw).fit(Xtr, ytr); return m.predict_proba(Xte)[:, 1], None
    m = cnn(Xtr.shape[1], lstm=(name == "CNN-LSTM"))
    cwd = None if cw is None else {0: 1.0, 1: float((ytr == 0).sum() / (ytr == 1).sum())}
    h = m.fit(Xtr[..., None], ytr, epochs=50, batch_size=32, validation_split=0.1, verbose=0,
              class_weight=cwd, shuffle=True)
    return m.predict(Xte[..., None], verbose=0).ravel(), (h.history if hist else None)

def metrics(y, p, thr=0.5):
    yh = (p >= thr).astype(int); tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    P, R, F, _ = precision_recall_fscore_support(y, yh, average="binary", zero_division=0)
    Pw, Rw, Fw, _ = precision_recall_fscore_support(y, yh, average="weighted", zero_division=0)
    return dict(acc=accuracy_score(y, yh), P=P, R=R, F1=F, Pw=Pw, Rw=Rw, F1w=Fw,
                spec=tn / (tn + fp), auc=roc_auc_score(y, p), auprc=average_precision_score(y, p),
                mcc=matthews_corrcoef(y, yh), bacc=(R + tn / (tn + fp)) / 2, tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))

def mcnemar(y, a, b):
    ca, cb = (a == y), (b == y); n01 = int((ca & ~cb).sum()); n10 = int((~ca & cb).sum())
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10) if n01 + n10 else 0.0
    return n01, n10, stat, 1 - chi2.cdf(stat, 1)

mode = sys.argv[2]
out = {}
X, y = df.drop(columns="stroke"), df.stroke.values
if mode in ("holdout", "holdout_cw", "leak"):
    tr, te, ytr, yte = train_test_split(df, y, test_size=0.2, stratify=y, random_state=SEED)
    Xtr, Xte = prep(tr, te, leak=(mode == "leak"))
    cw = "balanced" if mode == "holdout_cw" else None
    preds = {}
    for name in (["CNN-LSTM"] if mode == "leak" else ["SVM", "RF", "CNN", "CNN-LSTM"]):
        p, h = fit_predict(name, Xtr, ytr, Xte, cw=cw, hist=(name == "CNN-LSTM"))
        preds[name] = p; out[name] = metrics(yte, p)
        if h: out[name]["hist"] = {k: [float(v) for v in vals] for k, vals in h.items()}
        print(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in out[name].items() if k != "hist"}, flush=True)
    if mode != "leak":
        b = (preds["CNN-LSTM"] >= .5).astype(int)
        out["mcnemar"] = {}
        for other in ["SVM", "RF", "CNN"]:
            a = (preds[other] >= .5).astype(int)
            out["mcnemar"][other] = [float(v) for v in mcnemar(yte, a, b)]
        print("McNemar", out["mcnemar"])
        rng = np.random.default_rng(SEED); ci = {}
        for k, v in preds.items():
            bs = []
            for _ in range(2000):
                idx = rng.integers(0, len(yte), len(yte))
                if yte[idx].min() == yte[idx].max(): continue
                bs.append(roc_auc_score(yte[idx], v[idx]))
            ci[k] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        out["auc_ci"] = ci; print("AUC CI", ci)
        out["preds"] = {k: list(map(float, v)) for k, v in preds.items()}; out["yte"] = list(map(int, yte))
        out["roc"] = {k: [list(map(float, z)) for z in roc_curve(yte, v)[:2]] for k, v in preds.items()}
        out["n_test"] = int(len(yte)); out["n_test_pos"] = int(yte.sum())
elif mode == "perm":
    tr, te, ytr, yte = train_test_split(df, y, test_size=0.2, stratify=y, random_state=SEED)
    Xtr, Xte = prep(tr, te); rng = np.random.default_rng(SEED); out["perm"] = []
    for r in range(5):
        o = np.arange(10) if r == 0 else rng.permutation(10)
        p, _ = fit_predict("CNN-LSTM", Xtr[:, o], ytr, Xte[:, o], cw="balanced")
        m = metrics(yte, p); out["perm"].append({"order": o.tolist(), "auc": m["auc"], "R": m["R"], "bacc": m["bacc"]})
        print(out["perm"][-1], flush=True)
elif mode in ("cv", "cv_cw"):
    cw = "balanced" if mode == "cv_cw" else None
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    res = {n: [] for n in ["SVM", "RF", "CNN", "CNN-LSTM"]}
    for k, (i, j) in enumerate(skf.split(df, y)):
        Xtr, Xte = prep(df.iloc[i], df.iloc[j])
        for n in res:
            p, _ = fit_predict(n, Xtr, y[i], Xte, cw=cw); res[n].append(metrics(y[j], p))
        print("fold", k, flush=True)
    for n, L in res.items():
        out[n] = {m: (float(np.mean([r[m] for r in L])), float(np.std([r[m] for r in L])))
                  for m in ["acc", "P", "R", "F1", "spec", "auc", "auprc", "mcc", "bacc"]}
        print(n, {m: f"{a:.3f}±{s:.3f}" for m, (a, s) in out[n].items()})
json.dump(out, open(f"res_{mode}.json", "w"))
