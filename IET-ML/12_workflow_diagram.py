import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(14,4.5))
ax.set_xlim(0,13.8); ax.set_ylim(0,4); ax.axis("off")

stages = [
    "Data Acquisition\n(Sensors / CPCB-style\nAQ dataset)",
    "Preprocessing\n(NumPy/Pandas)\nCleaning, Imputation,\nOutlier handling",
    "Feature Engineering\n& Statistical Analysis\n(Correlation, EDA)",
    "Model Training\nConcept Learning, DT,\nBayes, KNN, LWR, MLP",
    "GA-based Feature\nSelection /\nOptimization",
    "Evaluation &\nComparison\n(Accuracy, F1, RMSE)",
    "Integrated Warning\nSystem\n(Prediction + Alerts)",
]
n = len(stages)
box_w, box_h = 1.55, 1.6
xs = [0.3 + i*1.75 for i in range(n)]
y = 1.2

for i, (x, text) in enumerate(zip(xs, stages)):
    color = "#E3F2FD" if i % 2 == 0 else "#FFF3E0"
    box = FancyBboxPatch((x, y), box_w, box_h, boxstyle="round,pad=0.04,rounding_size=0.08",
                          linewidth=1.4, edgecolor="#37474F", facecolor=color)
    ax.add_patch(box)
    ax.text(x+box_w/2, y+box_h/2, text, ha="center", va="center", fontsize=8.3, wrap=True)
    if i < n-1:
        arrow = FancyArrowPatch((x+box_w, y+box_h/2), (xs[i+1], y+box_h/2),
                                 arrowstyle='-|>', mutation_scale=14, color="#37474F", linewidth=1.4)
        ax.add_patch(arrow)

plt.title("System Workflow: Intelligent Urban Air-Quality Monitoring & Health-Risk Prediction", fontsize=11, pad=14)
plt.tight_layout()
plt.savefig("figs/00_system_workflow.png", dpi=150, bbox_inches="tight")
print("saved workflow diagram")
