"""
Radar charts comparing Mamba 1.4B vs paper's Transformers across noise types.

Chart A: Noise Absorption — relative accuracy drop during Phase 2
Chart B: Unlearning Recovery — Phase 3 accuracy as % of Phase 1

Usage: pixi run python mamba/plots/radar_chart.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# --- Data: (Phase1, Phase2, Phase3) accuracies per noise type ---
# Order: Charflip, Wordflip, Transliteration, Counterfactual

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

# Colors — Mamba stands out, transformers get a gradient
MODEL_STYLES = {
    "Olmo 1B":    {"color": "#7eb0d5", "ls": "-",  "marker": "o"},
    "Qwen 1.8B":  {"color": "#b2e061", "ls": "-",  "marker": "s"},
    "Gemma 2B":   {"color": "#fd7f6f", "ls": "-",  "marker": "D"},
    "Phi2 2.7B":  {"color": "#bd7ebe", "ls": "-",  "marker": "^"},
    "Mamba 1.4B": {"color": "#ffb55a", "ls": "-",  "marker": "P", "lw": 2.5},
}


def compute_noise_absorption(p1: float, p2: float) -> float:
    """Relative accuracy drop during noise training (Phase 2). Higher = more absorbed."""
    if p1 == 0:
        return 0.0
    return (p1 - p2) / p1 * 100


def compute_unlearning_recovery(p1: float, p3: float) -> float:
    """Phase 3 accuracy as % of Phase 1. Can exceed 100% if P3 > P1."""
    if p1 == 0:
        return 0.0
    return p3 / p1 * 100


def make_radar(ax, title, metric_fn, models, noise_types, noise_labels, model_styles):
    """Draw a single radar chart on the given axis."""
    n = len(noise_types)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), noise_labels, fontsize=11, fontweight="bold")

    for name, data in models.items():
        values = []
        for nt in noise_types:
            p1, p2, p3 = data[nt]
            values.append(metric_fn(p1, p2, p3))
        values += values[:1]  # close

        style = model_styles[name]
        lw = style.get("lw", 1.8)
        ax.plot(angles, values, style["ls"], color=style["color"],
                linewidth=lw, marker=style["marker"], markersize=6, label=name)
        ax.fill(angles, values, alpha=0.06, color=style["color"])

    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)
    ax.set_rlabel_position(30)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(True, alpha=0.3)


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5),
                                    subplot_kw=dict(polar=True))
    fig.suptitle("Mamba 1.4B vs Transformers: Noise Learning & Unlearning",
                 fontsize=15, fontweight="bold", y=1.0)

    # Chart A: Noise Absorption
    make_radar(
        ax1,
        "Noise Absorption\n(relative accuracy drop in Phase 2)",
        lambda p1, p2, p3: compute_noise_absorption(p1, p2),
        MODELS, NOISE_TYPES, NOISE_LABELS, MODEL_STYLES,
    )

    # Chart B: Unlearning Recovery
    make_radar(
        ax2,
        "Unlearning Recovery\n(Phase 3 accuracy as % of Phase 1)",
        lambda p1, p2, p3: compute_unlearning_recovery(p1, p3),
        MODELS, NOISE_TYPES, NOISE_LABELS, MODEL_STYLES,
    )

    # Single shared legend below charts
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5,
              fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    out_path = Path(__file__).parent / "radar_chart.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
