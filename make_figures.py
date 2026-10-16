import json, numpy as np, matplotlib, os
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score, roc_auc_score, matthews_corrcoef

OUT = "figs/"; os.makedirs(OUT, exist_ok=True)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
M = ["LR", "SVM", "RF", "HGB", "CNN", "CNN-LSTM", "Proposed"]
COL = dict(zip(M, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]))
LAB = {"LR": "Logistic regression", "SVM": "SVM", "RF": "Random forest", "HGB": "Gradient boosting",
       "CNN": "1D-CNN", "CNN-LSTM": "CNN–LSTM", "Proposed": "Hybrid stack (proposed)"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 300})
r = json.load(open("res_hybrid.json")); F = r["fold"]
y = np.array(r["y"]); oof = {k: np.array(v) for k, v in r["oof"].items()}

def grouped(metrics, names, fname, ylabel, ylim):
    fig, ax = plt.subplots(figsize=(7.6, 4.0)); x = np.arange(len(metrics)); w = 0.115
    for i, m in enumerate(M):
        mu = [F[m][k][0] for k in metrics]; sd = [F[m][k][1] for k in metrics]
        xs = x + (i - 3) * (w + 0.004)
        ax.bar(xs, mu, w, color=COL[m], label=LAB[m], edgecolor=INK if m == "Proposed" else SURF, linewidth=1.2 if m == "Proposed" else 0.8)
        ax.errorbar(xs, mu, yerr=sd, fmt="none", ecolor=INK2, elinewidth=0.9, capsize=2)
    ax.set_xticks(x, names); ax.set_ylim(*ylim); ax.set_ylabel(ylabel); ax.grid(axis="x", visible=False)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.2), fontsize=8.5)
    fig.tight_layout(); fig.savefig(OUT + fname); plt.close(fig)
grouped(["auc", "bacc", "R", "spec"], ["ROC–AUC", "Balanced accuracy", "Recall", "Specificity"], "fig4.png",
        "Score (nested 5-fold CV, mean ± SD)", (0, 1.0))
grouped(["auprc", "P", "F1", "mcc"], ["PR–AUC", "Precision", "F1-score", "MCC"], "fig5.png",
        "Stroke-class score (mean ± SD)", (0, 0.32))

# Fig 6: meta-learner weights
C = np.array(r["meta_coef"]); B = ["LR", "SVM", "RF", "HGB", "CNN", "CNN-LSTM"]
fig, ax = plt.subplots(figsize=(6.4, 3.2))
mu, sd = C.mean(0), C.std(0)
ax.barh(range(6), mu, xerr=sd, color=[COL[b] for b in B], ecolor=INK2, capsize=3, height=0.6)
ax.axvline(0, color=INK2, lw=1); ax.set_yticks(range(6), [LAB[b] for b in B]); ax.invert_yaxis()
ax.set_xlabel("Meta-learner coefficient (mean ± SD over 5 outer folds)"); ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(OUT + "fig6.png"); plt.close(fig)

# Fig 7: pooled confusion matrix of proposed
tn, fp, fn, tp = r["cm"]["Proposed"]; cm = np.array([[tn, fp], [fn, tp]])
fig, ax = plt.subplots(figsize=(4.4, 3.8)); ax.grid(False)
ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.15); nm = [["TN", "FP"], ["FN", "TP"]]
for i in range(2):
    for j in range(2):
        dk = cm[i, j] > cm.max() * 0.5
        ax.text(j, i - 0.06, f"{cm[i, j]:,}", ha="center", va="center", fontsize=17, fontweight="bold", color="white" if dk else INK)
        ax.text(j, i + 0.2, nm[i][j], ha="center", va="center", fontsize=9, color="white" if dk else INK2)
ax.set_xticks([0, 1], ["No stroke (0)", "Stroke (1)"]); ax.set_yticks([0, 1], ["No stroke (0)", "Stroke (1)"])
ax.set_xlabel("Predicted label"); ax.set_ylabel("Actual label")
for s in ax.spines.values(): s.set_visible(False)
fig.tight_layout(); fig.savefig(OUT + "fig7.png", bbox_inches="tight"); plt.close(fig)

# Fig 8: pooled out-of-fold ROC and PR
fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.8))
for m in M:
    lw = 2.6 if m == "Proposed" else 1.4
    f, t, _ = roc_curve(y, oof[m]); axs[0].plot(f, t, color=COL[m], lw=lw, label=f"{LAB[m]} ({roc_auc_score(y, oof[m]):.3f})")
    p, rc, _ = precision_recall_curve(y, oof[m]); axs[1].plot(rc, p, color=COL[m], lw=lw, label=f"{LAB[m]} ({average_precision_score(y, oof[m]):.3f})")
axs[0].plot([0, 1], [0, 1], color=INK2, ls=":", lw=1); axs[1].axhline(y.mean(), color=INK2, ls=":", lw=1)
axs[0].set_xlabel("False-positive rate"); axs[0].set_ylabel("True-positive rate"); axs[1].set_xlabel("Recall"); axs[1].set_ylabel("Precision")
axs[0].set_title("(a) ROC (AUC)", loc="left", fontsize=10); axs[1].set_title("(b) Precision–recall (AP)", loc="left", fontsize=10)
axs[0].legend(fontsize=6.8, loc="lower right"); axs[1].legend(fontsize=6.8, loc="upper right")
for a in axs: a.set_xlim(0, 1); a.set_ylim(0, 1.01)
fig.tight_layout(); fig.savefig(OUT + "fig8.png"); plt.close(fig)

def box(ax, x, y, w, h, title, sub, fc, ec, fs=10, sfs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1", fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=fs, fontweight="bold", color=INK)
    if sub: ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=sfs, color=INK2)
def arrow(ax, a, b): ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.2))

# Fig 1: nested workflow
steps = [("Load dataset", "4,981 records · 10 predictors + 1 target", 0),
         ("Outer stratified 5-fold split", "each fold: ~3,985 training / ~996 test records", 1),
         ("Encoding & scaling (training data only)", "one-hot + z-score (LR, SVM, nets) · ordinal (trees)", 0),
         ("Inner 3-fold CV on training data", "grid search per base learner · out-of-fold scores", 0),
         ("Cost-sensitive base learners", "LR · SVM · RF · GBM · 1D-CNN · CNN–LSTM", 0),
         ("Stacking meta-learner", "class-balanced logistic regression on inner OOF scores", 2),
         ("Threshold selection", "Youden's J on inner OOF predictions", 0),
         ("Evaluate on outer test fold", "AUC, PR–AUC, recall, specificity, MCC · DeLong test", 1)]
fig, ax = plt.subplots(figsize=(5.4, 8.6)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, len(steps) * 1.25 + 0.3)
for k, (t, s, kind) in enumerate(steps):
    yc = len(steps) * 1.25 - k * 1.25 - 0.3
    fc, ec = [("#e8f1fb", COL["LR"]), ("#fdeee7", COL["SVM"]), ("#eeecfb", COL["Proposed"])][kind]
    box(ax, 0.4, yc - 0.42, 9.2, 0.84, t, s, fc, ec, 10.5, 8.3)
    if k < len(steps) - 1: arrow(ax, (5, yc - 0.44), (5, yc - 0.83))
fig.savefig(OUT + "fig1.png", bbox_inches="tight"); plt.close(fig)

# Fig 2: hybrid system block diagram
fig, ax = plt.subplots(figsize=(10, 4.6)); ax.axis("off"); ax.set_xlim(0, 20); ax.set_ylim(0, 9.4)
box(ax, 0.2, 3.8, 2.6, 1.8, "Clinical\nrecords", "", "#e9e9e6", INK2, 10)
box(ax, 3.2, 3.8, 3.4, 1.8, "Preprocessing", "fitted on\ntraining data", "#e8f1fb", COL["LR"], 9.5, 7.5)
arrow(ax, (2.85, 4.7), (3.15, 4.7))
bases = ["LR", "SVM", "RF", "HGB", "CNN", "CNN-LSTM"]
for i, b in enumerate(bases):
    yy = 8.1 - i * 1.5
    ax.add_patch(FancyBboxPatch((7.4, yy), 4.2, 1.05, boxstyle="round,pad=0.02,rounding_size=0.1", fc="#ffffff", ec=COL[b], lw=1.8))
    ax.text(9.5, yy + 0.52, LAB[b], ha="center", va="center", fontsize=9.5, fontweight="bold", color=INK)
    arrow(ax, (6.65, 4.7), (7.35, yy + 0.52)); arrow(ax, (11.65, yy + 0.52), (12.95, 4.7))
ax.text(9.5, 9.25, "Level 0: tuned, cost-sensitive base learners", ha="center", fontsize=9, color=INK2)
box(ax, 13.0, 3.6, 3.0, 2.2, "Meta-learner", "logistic regression\non base scores", "#eeecfb", COL["Proposed"], 10, 8)
ax.text(14.5, 6.1, "Level 1", ha="center", fontsize=9, color=INK2)
box(ax, 16.8, 3.8, 3.0, 1.8, "Stroke risk", "P(stroke) +\nYouden threshold", "#e5f6ef", COL["RF"], 10, 8)
arrow(ax, (16.05, 4.7), (16.75, 4.7))
fig.savefig(OUT + "fig2.png", bbox_inches="tight"); plt.close(fig)

# Fig 3: CNN-LSTM base learner
L = [("Input", "19 × 1"), ("Conv1D", "64 filters, k = 3\nReLU · 17 × 64"), ("MaxPool1D", "pool = 2\n8 × 64"),
     ("Conv1D", "128 filters, k = 3\nReLU · 6 × 128"), ("LSTM", "128 units\nDropout 0.3"),
     ("Dense", "64, ReLU\nDropout 0.3"), ("Output", "1 unit, sigmoid\n→ P(stroke)")]
fc = ["#e9e9e6", "#e8f1fb", "#e8f1fb", "#e8f1fb", "#e5f3e5", "#fdf3dc", "#fdf3dc"]
ec = [INK2, COL["LR"], COL["LR"], COL["LR"], COL["CNN-LSTM"], COL["HGB"], COL["HGB"]]
fig, ax = plt.subplots(figsize=(10, 2.9)); ax.axis("off"); ax.set_xlim(0, 7 * 1.6); ax.set_ylim(-0.9, 1.6)
for k, ((t, s), f_, e_) in enumerate(zip(L, fc, ec)):
    x0 = k * 1.6 + 0.08
    ax.add_patch(FancyBboxPatch((x0, 0), 1.36, 1.3, boxstyle="round,pad=0.02,rounding_size=0.08", fc=f_, ec=e_, lw=1.5))
    ax.text(x0 + 0.68, 1.0, t, ha="center", va="center", fontsize=9.5, fontweight="bold", color=INK)
    ax.text(x0 + 0.68, 0.45, s, ha="center", va="center", fontsize=7.6, color=INK2, linespacing=1.4)
    if k < 6: arrow(ax, (x0 + 1.38, 0.65), (x0 + 1.6, 0.65))
for a, b, t, c in [(1, 3, "Convolutional feature extraction", COL["LR"]), (4, 4, "Recurrent aggregation", COL["CNN-LSTM"]),
                   (5, 6, "Classification head", "#a87000")]:
    xa, xb = a * 1.6 + 0.08, b * 1.6 + 1.44
    ax.plot([xa, xb], [-0.25, -0.25], color=c, lw=1.5); ax.text((xa + xb) / 2, -0.55, t, ha="center", fontsize=9, color=c, fontweight="bold")
fig.savefig(OUT + "fig3.png", bbox_inches="tight"); plt.close(fig)

# derived stats for text
yh = None
print("MCC pooled proposed:", (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
print("NPV", tn / (tn + fn), "PPV", tp / (tp + fp), "sens", tp / (tp + fn), "spec", tn / (tn + fp))
print("coef mean", mu.round(3), "sd", sd.round(3))
for m in M: print(m, round(roc_auc_score(y, oof[m]), 4), round(average_precision_score(y, oof[m]), 4))
