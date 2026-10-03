"""Shared matplotlib style for report figures (the app uses Plotly)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GREEN = "#1f5c3a"
PALETTE = ["#1f5c3a", "#4c9a5f", "#9cc58a", "#d9a441", "#c0563f", "#5b7c99", "#8a6fb0", "#6b6b6b"]
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight", "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
