import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def box(ax, xy, w, h, text, fc, ec="#374151", fs=8.5):
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.03,rounding_size=0.025",
        linewidth=1.0,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color="#111827",
        linespacing=1.18,
    )


def arrow(ax, p0, p1, color="#4b5563", lw=1.0, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def panel_title(ax, x, text):
    ax.text(x, 0.95, text, ha="center", va="center", fontsize=10, fontweight="bold", color="#111827")


fig, ax = plt.subplots(figsize=(7.2, 2.9))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

panel_title(ax, 0.17, "A. Dense transition route")
panel_title(ax, 0.50, "B. Current-normalized route")
panel_title(ax, 0.83, "C. Registered gate")

box(ax, (0.035, 0.62), 0.11, 0.16, "state v\ninput x", "#eff6ff")
box(ax, (0.175, 0.62), 0.12, 0.16, "trainable\nT[v,x]", "#fee2e2", ec="#991b1b")
box(ax, (0.095, 0.30), 0.14, 0.16, "hard address\nat eval", "#fef3c7", ec="#92400e")
box(ax, (0.245, 0.30), 0.12, 0.16, "spike/reset\ncoupled", "#fee2e2", ec="#991b1b")
arrow(ax, (0.145, 0.70), (0.175, 0.70))
arrow(ax, (0.235, 0.62), (0.165, 0.46), rad=0.12)
arrow(ax, (0.235, 0.38), (0.245, 0.38))
ax.text(0.17, 0.18, "Failed gate:\n78.42 vs 78.96 post-hoc", ha="center", va="center", fontsize=8, color="#991b1b")

box(ax, (0.385, 0.64), 0.10, 0.14, "address a", "#eff6ff")
box(ax, (0.515, 0.64), 0.10, 0.14, "current\nlookup", "#dcfce7", ec="#166534")
box(ax, (0.645, 0.64), 0.10, 0.14, "fixed LIF\nLUT", "#eff6ff")
box(ax, (0.500, 0.34), 0.13, 0.15, "T x C moment\nnormalization", "#dbeafe", ec="#1d4ed8")
arrow(ax, (0.485, 0.71), (0.515, 0.71))
arrow(ax, (0.615, 0.71), (0.645, 0.71))
arrow(ax, (0.565, 0.64), (0.565, 0.49))
arrow(ax, (0.630, 0.42), (0.675, 0.64), rad=-0.15)
ax.text(0.565, 0.20, "Local diagnostic:\ncurrent MSE improves in 4/4", ha="center", va="center", fontsize=8, color="#1d4ed8")

box(ax, (0.790, 0.68), 0.15, 0.10, "C100 T=4  GO", "#dcfce7", ec="#166534", fs=8.2)
box(ax, (0.790, 0.54), 0.15, 0.10, "C100 T=1  NO", "#fee2e2", ec="#991b1b", fs=8.2)
box(ax, (0.790, 0.40), 0.15, 0.10, "C10 T=1   NO", "#fee2e2", ec="#991b1b", fs=8.2)
box(ax, (0.790, 0.26), 0.15, 0.10, "C10 T=4   NO", "#fee2e2", ec="#991b1b", fs=8.2)
ax.text(0.865, 0.14, "Matrix verdict:\n1/4 gates pass", ha="center", va="center", fontsize=8.5, color="#111827")

ax.plot([0.335, 0.335], [0.10, 0.91], color="#d1d5db", linewidth=0.8)
ax.plot([0.705, 0.705], [0.10, 0.91], color="#d1d5db", linewidth=0.8)

plt.tight_layout(pad=0.4)
fig.savefig("figures/cnl_lut_lif_evidence_flow.pdf", bbox_inches="tight")
fig.savefig("figures/cnl_lut_lif_evidence_flow.png", dpi=220, bbox_inches="tight")
