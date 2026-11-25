import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

results = json.load(open("outputs/results.json"))

# ---- Plot 1: AUC bar chart across all models, with oracle ceiling line ----
model_order = ["imaging_only", "tabular_only", "late_fusion_ensemble", "early_fusion", "bilinear_fusion"]
labels = ["Imaging\nonly", "Tabular\nonly", "Late fusion\n(ensemble avg)", "Early fusion\n(concat+MLP)", "Bilinear fusion\n(+interaction term)"]
aucs = [results[m]["auc"] for m in model_order]
colors = ["#888888", "#888888", "#d95f02", "#1b9e77", "#1b9e77"]

fig, ax = plt.subplots(figsize=(8, 5.5))
bars = ax.bar(labels, aucs, color=colors)
ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Chance (AUC=0.5)")
ax.axhline(results["diagnostics"]["fusion_head_with_true_bits"]["auc"], color="#7570b3",
           linestyle=":", linewidth=2, label="Oracle ceiling (true bits, AUC=0.96)")
for bar, auc in zip(bars, aucs):
    ax.text(bar.get_x() + bar.get_width() / 2, auc + 0.015, f"{auc:.2f}", ha="center", fontsize=10)
ax.set_ylabel("Test AUC")
ax.set_ylim(0, 1.05)
ax.set_title("Predicting an interaction-only label:\nunimodal and naive ensemble models are stuck at chance")
ax.legend(loc="upper left", fontsize=9)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("outputs/fusion_auc_comparison.png", dpi=150)
print("saved fusion_auc_comparison.png")

# ---- Plot 2: diagnostic chain bar ----
fig, ax = plt.subplots(figsize=(7.8, 4.5))
stages = ["SSL linear probe\n(6-class shape ID, accuracy)", "b_img recoverability\n(compact vs elongated, AUC)",
          "Best fusion model\n(bilinear, actual, AUC)", "Fusion head w/ true bits\n(oracle sanity check, AUC)"]
values = [0.463, 0.771, results["bilinear_fusion"]["auc"], results["diagnostics"]["fusion_head_with_true_bits"]["auc"]]
colors2 = ["#d95f02", "#e6ab02", "#1b9e77", "#7570b3"]
bars2 = ax.barh(stages, values, color=colors2)
ax.axvline(1/6, color="gray", linestyle=":", linewidth=1, label="Chance, 6-class (1/6)")
ax.axvline(0.5, color="black", linestyle="--", linewidth=1, label="Chance, binary AUC (0.5)")
for bar, v in zip(bars2, values):
    ax.text(v + 0.015, bar.get_y() + bar.get_height() / 2, f"{v:.2f}", va="center", fontsize=10)
ax.set_xlabel("Score (metric differs by row -- see labels; each vs. its own chance line)")
ax.set_xlim(0, 1.05)
ax.set_title("The diagnostic chain: upstream embedding quality\nbottlenecks downstream fusion performance")
ax.legend(loc="lower right", fontsize=8)
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig("outputs/diagnostic_chain.png", dpi=150)
print("saved diagnostic_chain.png")

# ---- Plot 3: per-quadrant accuracy, late fusion vs bilinear fusion ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, key, title in zip(axes, ["late_fusion_ensemble", "bilinear_fusion"],
                           ["Late fusion (ensemble avg)", "Bilinear fusion"]):
    r = results[key]
    y = np.array(r["y"]); preds = np.array(r["preds"])
    b_img = np.array(r["b_img"]); b_lab = np.array(r["b_lab"])
    grid = np.zeros((2, 2))
    for bi in [0, 1]:
        for bl in [0, 1]:
            mask = (b_img == bi) & (b_lab == bl)
            grid[bi, bl] = (preds[mask] == y[mask]).mean() if mask.sum() > 0 else np.nan
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["b_lab=0", "b_lab=1"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["b_img=0\n(elongated)", "b_img=1\n(compact)"])
    ax.set_title(f"{title}\naccuracy by (b_img, b_lab) quadrant")
    for bi in [0, 1]:
        for bl in [0, 1]:
            ax.text(bl, bi, f"{grid[bi,bl]:.2f}", ha="center", va="center", fontsize=13,
                     color="white" if grid[bi, bl] < 0.4 or grid[bi, bl] > 0.75 else "black")
fig.tight_layout()
fig.savefig("outputs/quadrant_accuracy.png", dpi=150)
print("saved quadrant_accuracy.png")

print("ALL PLOTS DONE")
