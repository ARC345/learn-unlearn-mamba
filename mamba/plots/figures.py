"""
Additional figures for the Mamba noise blog post.

Generates:
  1. Mamba phase trajectory (grouped bar) — P1/P2/P3 per noise type
  2. Noise absorption comparison (grouped bar) — all models side by side
  3. Unlearning recovery comparison (grouped bar) — all models side by side
  4. Mamba vs transformer range (band chart) — Mamba line vs transformer min/max band

Usage: pixi run python mamba/plots/figures.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Shared data ---

MODELS = {
    "Olmo 1B": {
        "charflip": (72.2, 2.7, 65.2),
        "wordflip": (72.2, 45.0, 70.4),
        "translit": (72.2, 67.6, 73.1),
        "counter":  (39.8, 32.6, 41.4),
    },
    "Qwen 1.8B": {
        "charflip": (82.3, 2.5, 79.7),
        "wordflip": (82.3, 57.7, 81.9),
        "translit": (82.3, 81.1, 82.7),
        "counter":  (66.5, 54.2, 65.7),
    },
    "Gemma 2B": {
        "charflip": (89.1, 3.8, 85.9),
        "wordflip": (89.1, 64.1, 82.8),
        "translit": (89.1, 85.0, 87.5),
        "counter":  (49.6, 40.8, 49.0),
    },
    "Phi2 2.7B": {
        "charflip": (95.7, 0.5, 90.7),
        "wordflip": (95.7, 69.7, 93.1),
        "translit": (95.7, 93.2, 93.6),
        "counter":  (66.5, 57.5, 69.5),
    },
    "Mamba 1.4B": {
        "charflip": (29.4, 1.6, 37.0),
        "wordflip": (29.4, 8.5, 36.3),
        "translit": (29.4, 26.3, 29.8),
        "counter":  (29.4, 37.6, 39.2),
    },
}

NOISE_TYPES = ["charflip", "wordflip", "translit", "counter"]
NOISE_LABELS = ["Charflip", "Wordflip", "Transliteration", "Counterfactual"]
TRANSFORMER_NAMES = ["Olmo 1B", "Qwen 1.8B", "Gemma 2B", "Phi2 2.7B"]

PHASE_COLORS = ["#4a90d9", "#e74c3c", "#2ecc71"]  # P1 blue, P2 red, P3 green
MAMBA_COLOR = "#ffb55a"
TRANSFORMER_BAND = "#b0b0b0"

OUT_DIR = Path(__file__).parent


def fig1_mamba_trajectory():
    """Grouped bar chart: Mamba P1/P2/P3 across noise types."""
    mamba = MODELS["Mamba 1.4B"]
    x = np.arange(len(NOISE_LABELS))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))

    p1_vals = [mamba[nt][0] for nt in NOISE_TYPES]
    p2_vals = [mamba[nt][1] for nt in NOISE_TYPES]
    p3_vals = [mamba[nt][2] for nt in NOISE_TYPES]

    bars1 = ax.bar(x - width, p1_vals, width, label="Phase 1 (Finetune)",
                   color=PHASE_COLORS[0], edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x, p2_vals, width, label="Phase 2 (Noise)",
                   color=PHASE_COLORS[1], edgecolor="white", linewidth=0.5)
    bars3 = ax.bar(x + width, p3_vals, width, label="Phase 3 (Unlearn)",
                   color=PHASE_COLORS[2], edgecolor="white", linewidth=0.5)

    # Value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(NOISE_LABELS, fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Mamba 1.4B: Accuracy Across Phases", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.set_ylim(0, 50)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = OUT_DIR / "mamba_trajectory.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close(fig)


def fig2_absorption_comparison():
    """Grouped bar: noise absorption (relative P2 drop) for all models."""
    model_names = TRANSFORMER_NAMES + ["Mamba 1.4B"]
    x = np.arange(len(NOISE_LABELS))
    n_models = len(model_names)
    width = 0.15

    colors = ["#7eb0d5", "#b2e061", "#fd7f6f", "#bd7ebe", "#ffb55a"]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for i, name in enumerate(model_names):
        vals = []
        for nt in NOISE_TYPES:
            p1, p2, _ = MODELS[name][nt]
            absorption = (p1 - p2) / p1 * 100 if p1 != 0 else 0
            vals.append(absorption)
        offset = (i - n_models / 2 + 0.5) * width
        edge = "black" if name == "Mamba 1.4B" else "white"
        lw = 1.5 if name == "Mamba 1.4B" else 0.5
        ax.bar(x + offset, vals, width, label=name, color=colors[i],
               edgecolor=edge, linewidth=lw)

    ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="-")
    ax.set_xticks(x)
    ax.set_xticklabels(NOISE_LABELS, fontsize=11)
    ax.set_ylabel("Noise Absorption (%)", fontsize=11)
    ax.set_title("Noise Absorption: Relative Accuracy Drop in Phase 2",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right", ncol=2)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = OUT_DIR / "absorption_comparison.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close(fig)


def fig3_recovery_comparison():
    """Grouped bar: unlearning recovery (P3/P1 %) for all models."""
    model_names = TRANSFORMER_NAMES + ["Mamba 1.4B"]
    x = np.arange(len(NOISE_LABELS))
    n_models = len(model_names)
    width = 0.15

    colors = ["#7eb0d5", "#b2e061", "#fd7f6f", "#bd7ebe", "#ffb55a"]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for i, name in enumerate(model_names):
        vals = []
        for nt in NOISE_TYPES:
            p1, _, p3 = MODELS[name][nt]
            recovery = p3 / p1 * 100 if p1 != 0 else 0
            vals.append(recovery)
        offset = (i - n_models / 2 + 0.5) * width
        edge = "black" if name == "Mamba 1.4B" else "white"
        lw = 1.5 if name == "Mamba 1.4B" else 0.5
        ax.bar(x + offset, vals, width, label=name, color=colors[i],
               edgecolor=edge, linewidth=lw)

    ax.axhline(y=100, color="gray", linewidth=1, linestyle="--", alpha=0.7)
    ax.text(len(NOISE_LABELS) - 0.5, 101.5, "100% baseline", fontsize=8,
            color="gray", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(NOISE_LABELS, fontsize=11)
    ax.set_ylabel("Recovery (% of Phase 1)", fontsize=11)
    ax.set_title("Unlearning Recovery: Phase 3 Accuracy as % of Phase 1",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left", ncol=2)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = OUT_DIR / "recovery_comparison.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close(fig)


def fig4_mamba_vs_transformer_band():
    """Line + band chart: Mamba trajectory vs transformer min/max range per noise type."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)

    phases = ["Phase 1\n(Finetune)", "Phase 2\n(Noise)", "Phase 3\n(Unlearn)"]
    x = np.arange(3)

    for idx, (nt, label) in enumerate(zip(NOISE_TYPES, NOISE_LABELS)):
        ax = axes[idx]

        # Transformer range
        t_vals = np.array([MODELS[n][nt] for n in TRANSFORMER_NAMES])
        t_min = t_vals.min(axis=0)
        t_max = t_vals.max(axis=0)
        t_mean = t_vals.mean(axis=0)

        ax.fill_between(x, t_min, t_max, alpha=0.25, color=TRANSFORMER_BAND,
                        label="Transformer range")
        ax.plot(x, t_mean, "--", color="#666666", linewidth=1.5,
                label="Transformer mean", marker="o", markersize=5)

        # Mamba
        mamba_vals = MODELS["Mamba 1.4B"][nt]
        ax.plot(x, mamba_vals, "-", color=MAMBA_COLOR, linewidth=2.5,
                marker="P", markersize=8, label="Mamba 1.4B", zorder=5)

        # Value labels for Mamba
        for xi, v in zip(x, mamba_vals):
            ax.annotate(f"{v:.1f}", (xi, v), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8, fontweight="bold",
                        color=MAMBA_COLOR)

        ax.set_xticks(x)
        ax.set_xticklabels(phases, fontsize=9)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if idx == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=11)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=10,
              frameon=True, bbox_to_anchor=(0.5, -0.08))

    fig.suptitle("Phase Trajectory: Mamba 1.4B vs Transformer Range",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    out = OUT_DIR / "trajectory_band.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    fig1_mamba_trajectory()
    fig2_absorption_comparison()
    fig3_recovery_comparison()
    fig4_mamba_vs_transformer_band()
    print("All figures generated.")
