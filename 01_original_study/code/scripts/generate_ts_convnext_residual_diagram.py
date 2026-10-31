from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np


# Diagram output
OUT_PATH = Path("results/lowdata_hybrid_distill/ts_convnext_residual_schema.png")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def add_box(ax, x, y, w, h, text, fc="#e9f2ff", ec="#1f5aa6", fontsize=10, weight="normal"):
    # Drop shadow for depth
    shadow = FancyBboxPatch(
        (x + 0.004, y - 0.004), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        linewidth=0,
        edgecolor="none",
        facecolor="#000000",
        alpha=0.12
    )
    ax.add_patch(shadow)

    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        linewidth=1.6,
        edgecolor=ec,
        facecolor=fc
    )
    ax.add_patch(box)
    ax.text(
        x + w * 0.5,
        y + h * 0.5,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight=weight,
        color="#132235"
    )


def add_group(ax, x, y, w, h, title, fc="#f7f9fc", ec="#2f4f7f"):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        linewidth=1.5,
        edgecolor=ec,
        facecolor=fc
    )
    ax.add_patch(rect)
    ax.text(
        x + 0.015,
        y + h + 0.015,
        title,
        ha="left",
        va="bottom",
        fontsize=11,
        weight="bold",
        color="#24324a"
    )


def arrow(ax, x1, y1, x2, y2, rad=0.0):
    patch = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=14,
        lw=1.6,
        color="#2b2b2b",
        connectionstyle=f"arc3,rad={rad}"
    )
    ax.add_patch(patch)


def add_background(ax):
    # Subtle gradient background
    grad = np.linspace(0, 1, 256)
    grad = np.vstack([grad, grad])
    ax.imshow(grad, extent=[0, 1, 0, 1], origin="lower", cmap="Blues", alpha=0.10, aspect="auto")
    ax.imshow(grad[::-1], extent=[0, 1, 0, 1], origin="lower", cmap="Oranges", alpha=0.08, aspect="auto")
    # Soft vignette
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="#ffffff", alpha=0.35, edgecolor="none"))


def main():
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    add_background(ax)

    # Left: Input
    add_box(ax, 0.02, 0.40, 0.10, 0.20, "INPUT\nRGB 224x224", fc="#dfeeff", ec="#1f5aa6", fontsize=11, weight="bold")

    # Teacher ensemble group
    add_group(ax, 0.16, 0.58, 0.44, 0.32, "TEACHER ENSEMBLE (HYBRID CNN+ViT)")

    # Baseline hybrid
    add_box(ax, 0.18, 0.76, 0.16, 0.10, "Baseline Hybrid\nCNN + ViT\nGated Fusion", fc="#e8f6ff")
    add_box(ax, 0.18, 0.62, 0.16, 0.10, "HybridCNNViTBase\n(ResNet18-like + ViT)", fc="#e8f6ff")

    # Advanced hybrid
    add_box(ax, 0.38, 0.76, 0.20, 0.10, "Advanced Hybrid\nResNet50 + ViT-B/16\nConcat + SpectralNorm", fc="#e8f6ff")
    add_box(ax, 0.38, 0.62, 0.20, 0.10, "AdvancedHybridModel\n(Teacher)", fc="#e8f6ff")

    # Teacher gate
    add_box(ax, 0.62, 0.68, 0.14, 0.14, "Teacher Gate\nEntropy + ConfGap", fc="#eaf7ea", ec="#2f7a3d", fontsize=10, weight="bold")

    # Student group
    add_group(ax, 0.16, 0.20, 0.44, 0.26, "STUDENT BRANCH (CONVNEXT TINY)")
    add_box(ax, 0.18, 0.30, 0.20, 0.12, "ConvNeXt Tiny\nStudent CNN", fc="#fff1e6", ec="#a64b1f", fontsize=10, weight="bold")

    # Residual mixer
    add_box(ax, 0.62, 0.32, 0.16, 0.20, "Residual Mix\n+ Conflict Brake\nzg = zt + a*g*(zs-mean)", fc="#f4e9ff", ec="#6a3dad", fontsize=9, weight="bold")

    # Output
    add_box(ax, 0.82, 0.40, 0.14, 0.20, "OUTPUT\n5 Classes\nSoftmax", fc="#dff7f2", ec="#1c7a66", fontsize=11, weight="bold")

    # Losses (bottom note)
    add_box(
        ax,
        0.16,
        0.05,
        0.60,
        0.10,
        "Loss: Weighted CE + NonInferiority + Distill (T=2,4) + Ordinal CDF - Gate Entropy",
        fc="#f8f8f8",
        ec="#888",
        fontsize=9
    )

    # Arrows
    arrow(ax, 0.12, 0.50, 0.16, 0.74, rad=0.12)  # input -> teacher group
    arrow(ax, 0.12, 0.50, 0.16, 0.36, rad=-0.08)  # input -> student

    arrow(ax, 0.34, 0.81, 0.62, 0.75, rad=0.08)  # baseline -> gate
    arrow(ax, 0.58, 0.81, 0.62, 0.75, rad=-0.08)  # advanced -> gate

    arrow(ax, 0.38, 0.36, 0.62, 0.40, rad=0.10)  # student -> residual mix
    arrow(ax, 0.76, 0.75, 0.62, 0.44, rad=-0.15)  # teacher gate -> residual mix

    arrow(ax, 0.78, 0.42, 0.82, 0.50, rad=0.0)  # residual mix -> output

    # Title
    ax.text(
        0.5,
        0.96,
        "TS_ConvNeXtTiny_Residual: Teacher-Student Residual Schematic",
        ha="center",
        va="center",
        fontsize=14,
        weight="bold",
        color="#14253d"
    )

    plt.savefig(OUT_PATH, dpi=220, bbox_inches="tight")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
