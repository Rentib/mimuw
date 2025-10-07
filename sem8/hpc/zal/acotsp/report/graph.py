import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Configure matplotlib for LaTeX compatibility
matplotlib.use("pdf")  # Use PDF backend for reliability
plt.style.use("seaborn-v0_8")

# Set larger default font sizes
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "grid.linewidth": 1.2,  # Thicker grid lines
        "grid.alpha": 0.8,  # Less transparent grid
    }
)

# Optimal results - ordered as specified
optimal_results = {
    "d198": 15780,
    "a280": 2579,
    "lin318": 42029,
    "pcb442": 50778,
    "rat783": 8806,
    "pr1002": 259045,
}

# Read and process results
df = pd.read_csv("results.csv")
df = df[df["type"].isin(["WORKER", "QUEEN"])]
df["ratio"] = df.apply(
    lambda row: row["result"] / optimal_results[row["instance"]], axis=1
)

# Get common instances in the specified order
common_instances = ["d198", "a280", "lin318", "pcb442", "rat783", "pr1002"]

# Create figure with larger size
fig, ax = plt.subplots(figsize=(14, 7))

# Calculate statistics
worker_stats = (
    df[df["type"] == "WORKER"].groupby("instance")["ratio"].agg(["mean", "std"])
).reindex(common_instances)
queen_stats = (
    df[df["type"] == "QUEEN"].groupby("instance")["ratio"].agg(["mean", "std"])
).reindex(common_instances)

# Bar positions
x = np.arange(len(common_instances))
width = 0.35  # Width of the bars

# Create bars with more distinct colors
worker_bars = ax.bar(
    x - width / 2,
    worker_stats["mean"] - 1.0,
    width,
    bottom=1.0,
    color="#1f77b4",
    alpha=0.9,
    label="WORKER",
    edgecolor="black",
    linewidth=1.5,
)
queen_bars = ax.bar(
    x + width / 2,
    queen_stats["mean"] - 1.0,
    width,
    bottom=1.0,
    color="#ff7f0e",
    alpha=0.9,
    label="QUEEN",
    edgecolor="black",
    linewidth=1.5,
)

# Add error bars with more visibility
ax.errorbar(
    x - width / 2,
    worker_stats["mean"],
    yerr=worker_stats["std"],
    fmt="none",
    ecolor="black",
    elinewidth=1.5,
    capsize=6,
    capthick=1.5,
)
ax.errorbar(
    x + width / 2,
    queen_stats["mean"],
    yerr=queen_stats["std"],
    fmt="none",
    ecolor="black",
    elinewidth=1.5,
    capsize=6,
    capthick=1.5,
)

# Y-axis setup with more visible ticks
ax.set_ylim(1.0, 1.5)
ax.set_yticks(np.arange(1.0, 1.55, 0.1))  # More ticks for better grid
ax.tick_params(axis="y", which="major", length=6, width=1)

# Remove spines below 1.0 and make remaining spines more visible
ax.spines["bottom"].set_position(("data", 1.0))
ax.spines["bottom"].set_color("black")
ax.spines["bottom"].set_linewidth(1.5)
ax.spines["left"].set_bounds(1.0, 1.5)
ax.spines["left"].set_linewidth(1.5)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

# Labels and title with more weight
ax.set_xlabel("TSPLIB Benchmarks", fontsize=14, labelpad=15, weight="bold")
ax.set_ylabel("Quality of the Solution", fontsize=14, labelpad=15, weight="bold")

# Horizontal x-axis labels
ax.set_xticks(x)
ax.set_xticklabels(common_instances, rotation=0, ha="center", fontsize=14)

# Customize grid appearance - making it more visible
ax.grid(axis="y", linestyle="-", linewidth=1.2, alpha=0.7, color="gray")
ax.grid(axis="x", visible=False)  # No vertical grid lines

# Enhanced legend
legend = ax.legend(framealpha=1, loc="upper left", edgecolor="black")
legend.get_frame().set_linewidth(1.5)

# Adjust layout and save
plt.tight_layout(pad=3.0)

# Save in multiple formats
plt.savefig("performance_comparison.png", bbox_inches="tight", dpi=300)
