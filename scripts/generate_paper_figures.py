"""Assemble the three main paper figures from saved empirical results."""

import csv
import re
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures" / "main_figures"
INTERMEDIATE = ROOT / "figures" / "intermediate"
RESULTS = ROOT / "results"
STATS_64 = RESULTS / "stats_64_channels.txt"
STATS_128 = RESULTS / "stats_128_channels.txt"
WPLI_CONDITION_MEANS = RESULTS / "wpli_condition_mean_changes.csv"

INK = "#26323D"
MUTED = "#566473"
GRID = "#E2E8ED"
FRAME = "#C8D2DA"
BLUE = "#3F7EA6"
ORANGE = "#DC8423"
GREEN = "#2A8A70"
PURPLE = "#7866A9"
GRAY = "#7A8794"
RED = "#B85B52"
PANEL_FILL = "#F8FAFB"
ZERO_BAND = "#F5F7F9"

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 7.4,
    "font.weight": "regular",
    "axes.titlesize": 8.6,
    "axes.titleweight": "semibold",
    "axes.labelsize": 7.4,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "text.color": INK,
    "axes.edgecolor": "#A7B3BD",
    "axes.linewidth": 0.65,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "xtick.major.width": 0.65,
    "ytick.major.width": 0.65,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
})

BANDS = ["Delta", "Theta", "Alpha", "Beta", "Broadband"]
CONDITIONS = ["Ketamine", "Medetomidine", "Propofol", "Sleep"]

METRIC_LABELS = {
    ("Kur_Sync", "delta"): "Delta synchrony",
    ("Kur_Meta", "delta"): "Delta metastability",
    ("Kur_Sync", "theta"): "Theta synchrony",
    ("Kur_Meta", "theta"): "Theta metastability",
    ("Kur_Sync", "alpha"): "Alpha synchrony",
    ("Kur_Meta", "alpha"): "Alpha metastability",
    ("Kur_Sync", "beta"): "Beta synchrony",
    ("Kur_Meta", "beta"): "Beta metastability",
    ("Kur_Sync", "broadband"): "Broadband synchrony",
    ("Kur_Meta", "broadband"): "Broadband metastability",
    ("wPLI", "delta"): "Delta wPLI",
    ("wPLI", "theta"): "Theta wPLI",
    ("wPLI", "alpha"): "Alpha wPLI",
    ("wPLI", "beta"): "Beta wPLI",
    ("wPLI", "broadband"): "Broadband wPLI",
    ("FCD_Var", "delta"): "Delta FCD variance",
    ("H_norm", "delta"): "Delta coalition entropy",
}

ROW_RE = re.compile(
    r"^(Kur_Sync|Kur_Meta|wPLI|FCD_Var|H_norm)\s+(\w+)\s+\|\s*"
    r"([+-]?\d+\.\d+)\s+\[\s*([+-]?\d+\.\d+),\s*([+-]?\d+\.\d+)\]\s+\|\s*"
    r"(\d+\.\d+)(?:\*|\(\+\))?\s+(\d+\.\d+)\s+\|?\s*([0-9.]+)?"
)

DIRECTION_RE = re.compile(
    r"^(Kur_Sync|Kur_Meta|wPLI)\s+(\w+)\s+\|\s+(\d+)\+/(\d+)-"
)


def parse_stats(path, section_title=None):
    rows = []
    in_section = section_title is None
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if section_title and section_title in line:
                in_section = True
                continue
            if not in_section:
                continue
            if line.startswith("Summary:"):
                break
            match = ROW_RE.match(line.strip())
            if not match:
                continue
            metric, band, d, ci_lo, ci_hi, p_raw, p_fdr, bf = match.groups()
            key = (metric, band)
            rows.append({
                "key": key,
                "metric": metric,
                "band": band,
                "label": METRIC_LABELS.get(key, f"{metric} {band}"),
                "d": float(d),
                "ci_lo": float(ci_lo),
                "ci_hi": float(ci_hi),
                "p_raw": float(p_raw),
                "p_fdr": float(p_fdr),
                "bf": float(bf) if bf is not None else np.nan,
            })
    return rows


def parse_key_duration(path):
    out = {}
    pat = re.compile(r"^(Kur_Meta|Kur_Sync|wPLI)\s+(\w+): full d=([+-]?\d+\.\d+).*120 s d=([+-]?\d+\.\d+)")
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = pat.match(line.strip())
            if match:
                metric, band, full_d, d120 = match.groups()
                out[(metric, band)] = {"full": float(full_d), "120s": float(d120)}
    return out


def parse_direction_counts(path):
    counts = {}
    in_grouping = False
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "Grouping-aware robustness" in line:
                in_grouping = True
                continue
            if in_grouping and line.startswith("Primary effects"):
                break
            if not in_grouping:
                continue
            match = DIRECTION_RE.match(line.strip())
            if match:
                metric, band, pos, neg = match.groups()
                counts[(metric, band)] = (int(pos), int(neg))
    return counts


def read_wpli_condition_means(path):
    """Read condition-level wPLI changes written by run_pipeline.py."""
    rows = {}
    with path.open('r', encoding='utf-8', newline='') as source:
        for row in csv.DictReader(source):
            rows[row['condition']] = [float(row[band.lower()]) for band in BANDS]
    return np.asarray([rows[condition] for condition in CONDITIONS], dtype=float)


def add_panel_label(ax, label, x=-0.08, y=1.03):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=10.0, fontweight="bold",
            ha="left", va="bottom", color=INK)


def set_panel_title(ax, label, title, loc="left", pad=7):
    add_panel_label(ax, label, x=-0.08, y=1.115)
    ax.text(0.5, 1.035, title, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=8.8,
            fontweight="semibold", color=INK)


def style_note_box(ax, text, xy, ha="left", va="top", fontsize=8.6):
    ax.text(
        xy[0], xy[1], text,
        transform=ax.transAxes, ha=ha, va=va, fontsize=fontsize, color=MUTED,
        bbox={"facecolor": PANEL_FILL, "edgecolor": FRAME, "linewidth": 0.55,
              "boxstyle": "round,pad=0.24,rounding_size=0.04"},
    )


def draw_subtle_box(ax, xy, wh, title, body, accent):
    ax.add_patch(
        patches.FancyBboxPatch(
            xy, wh[0], wh[1],
            boxstyle="round,pad=0.014,rounding_size=0.015",
            facecolor=PANEL_FILL, edgecolor=FRAME, linewidth=0.65,
        )
    )
    ax.add_patch(
        patches.Rectangle(
            (xy[0], xy[1] + wh[1] - 0.030), wh[0], 0.030,
            facecolor=accent, edgecolor="none", alpha=0.90,
        )
    )
    ax.text(xy[0] + wh[0] * 0.055, xy[1] + wh[1] * 0.66, title,
            fontsize=8.8, fontweight="semibold", ha="left", va="center")
    ax.text(xy[0] + wh[0] * 0.055, xy[1] + wh[1] * 0.30, body,
            fontsize=7.4, ha="left", va="center", color=MUTED, linespacing=1.12)


def metric_color(row):
    if row["key"] == ("wPLI", "broadband"):
        return GREEN
    if row["metric"] == "wPLI":
        return ORANGE
    if row["metric"].startswith("Kur"):
        return BLUE
    return GRAY


def crop_nonwhite(image, threshold=0.992, pad=18):
    rgb = image[..., :3]
    mask = np.mean(rgb, axis=2) < threshold
    if not np.any(mask):
        return image
    ys, xs = np.where(mask)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(image.shape[0], int(ys.max()) + pad)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(image.shape[1], int(xs.max()) + pad)
    return image[y0:y1, x0:x1, :]


def crop_x_nonwhite(image, threshold=0.992, pad=18):
    rgb = image[..., :3]
    mask = np.mean(rgb, axis=2) < threshold
    if not np.any(mask):
        return image
    _, xs = np.where(mask)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(image.shape[1], int(xs.max()) + pad)
    return image[:, x0:x1, :]


def wpli_topography_parts():
    topo = mpimg.imread(INTERMEDIATE / "fig12_wpli_topography.png")
    h, w = topo.shape[:2]
    # Remove the original title band and re-pack the four condition columns more tightly.
    base = topo[int(h * 0.075):, :, :]
    ranges = [
        (0.000, 0.205),
        (0.292, 0.463),
        (0.556, 0.728),
        (0.834, 0.998),
    ]
    columns = []
    for idx, (lo, hi) in enumerate(ranges):
        column = base[:, int(w * lo):int(w * hi), :]
        pad = 6 if idx == 3 else 18
        columns.append(crop_x_nonwhite(column, threshold=0.993, pad=pad))
    gap = np.ones((base.shape[0], 14, base.shape[2]), dtype=base.dtype)
    maps = columns[0]
    for column in columns[1:]:
        maps = np.concatenate([maps, gap, column], axis=1)
    maps = crop_nonwhite(maps, threshold=0.993, pad=22)
    return maps


def figure_1(rows):
    fig = plt.figure(figsize=(7.6, 6.25), constrained_layout=False)
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[0.88, 1.90],
        width_ratios=[1.05, 0.95],
        wspace=0.26,
        hspace=0.23,
    )
    ax_design = fig.add_subplot(gs[0, 0])
    ax_metrics = fig.add_subplot(gs[0, 1])
    ax_effect = fig.add_subplot(gs[1, :])

    ax_design.set_axis_off()
    ax_design.text(0.5, 1.035, "Paired comparisons",
                   transform=ax_design.transAxes, ha="center", va="bottom",
                   fontsize=8.8, fontweight="semibold", color=INK)
    effect_pos = ax_effect.get_position()
    design_pos = ax_design.get_position()
    fig.text(
        effect_pos.x0 - 0.08 * effect_pos.width,
        design_pos.y1 + 0.115 * design_pos.height,
        "A",
        fontsize=10.0,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=INK,
    )
    node_x = [0.16, 0.50, 0.84]
    node_titles = ["2 macaques", "4 conditions", "8 pairs"]
    node_bodies = ["George, Chibi", "KT, MD, PF,\nsleep", "awake vs\nnon-awake"]
    node_fill = PANEL_FILL
    node_edge = "#91A7B8"
    for idx, (x, title, body) in enumerate(zip(node_x, node_titles, node_bodies)):
        ax_design.scatter([x], [0.62], s=230, marker="o",
                          facecolors=node_fill, edgecolors=node_edge,
                          linewidths=0.90, zorder=2)
        # Arial numerals sit slightly high within their text box; this small
        # optical offset centers the visible glyph inside the circle.
        ax_design.text(x, 0.610, str(idx + 1), ha="center", va="center",
                       fontsize=7.7, fontweight="semibold", color=INK, zorder=3)
        ax_design.text(x, 0.37, title, ha="center", va="center",
                       fontsize=7.6, fontweight="semibold")
        ax_design.text(x, 0.22, body, ha="center", va="center",
                       fontsize=6.4, color=MUTED, linespacing=1.05)
    for x1, x2 in zip(node_x[:-1], node_x[1:]):
        ax_design.annotate("", xy=(x2 - 0.10, 0.62), xytext=(x1 + 0.10, 0.62),
                           arrowprops={"arrowstyle": "-|>", "lw": 0.65,
                                       "color": "#8794A0"})
    ax_design.set_xlim(0, 1)
    ax_design.set_ylim(0, 1)

    ax_metrics.set_axis_off()
    set_panel_title(ax_metrics, "B", "Metric categories")
    summary_groups = [
        ("Phase coordination", "Kuramoto synchrony | metastability", BLUE),
        ("Phase-lagged coupling", "weighted phase lag index (wPLI)", ORANGE),
        ("Dynamic organization", "FCD variance | coalition entropy", PURPLE),
    ]
    y_centers = [0.74, 0.47, 0.20]
    for idx, ((title, body, accent), yc) in enumerate(zip(summary_groups, y_centers)):
        ax_metrics.plot([0.01, 0.01], [yc - 0.075, yc + 0.075],
                        color=accent, lw=3.1, solid_capstyle="butt")
        ax_metrics.text(0.055, yc + 0.050, title, fontsize=6.45,
                        fontweight="semibold", ha="left", va="center")
        ax_metrics.text(0.055, yc - 0.050, body, fontsize=5.75, color=MUTED,
                        ha="left", va="center")
        if idx < len(summary_groups) - 1:
            ax_metrics.plot([0.01, 0.96], [yc - 0.125, yc - 0.125],
                            color=GRID, lw=0.50)
    ax_metrics.plot([0.01, 0.96], [0.055, 0.055], color=FRAME, lw=0.55)
    ax_metrics.set_xlim(0, 1)
    ax_metrics.set_ylim(0, 1)

    ordered = sorted(rows, key=lambda row: row["d"])
    y = np.arange(len(ordered))
    colors = [metric_color(row) for row in ordered]

    add_panel_label(ax_effect, "C", x=-0.08, y=1.035)
    ax_effect.text(0.5, 1.035, "State effects across metrics",
                   transform=ax_effect.transAxes, ha="center", va="bottom",
                   fontsize=8.8, fontweight="semibold", color=INK)
    for yi in y[::2]:
        ax_effect.axhspan(yi - 0.5, yi + 0.5, color="#FAFBFC", zorder=0)
    ax_effect.axvline(0, color="#5E6974", lw=0.72, zorder=1)
    for yi, row, color in zip(y, ordered, colors):
        ax_effect.plot([row["ci_lo"], row["ci_hi"]], [yi, yi],
                       color="#8996A2", lw=0.82, alpha=0.86, solid_capstyle="round")
        ax_effect.scatter(row["d"], yi, s=25, color=color, edgecolor="white",
                          linewidth=0.55, zorder=3)
        if row["bf"] > 3:
            ax_effect.text(row["ci_hi"] + 0.08, yi,
                           rf"$BF_{{10}}$={row['bf']:.1f}",
                           va="center", fontsize=6.55, color=MUTED)
    ax_effect.set_yticks(y)
    ax_effect.set_yticklabels([row["label"] for row in ordered], fontsize=7.1)
    ax_effect.set_xlabel("Paired Cohen's d (non-awake − awake)")
    ax_effect.set_xlim(-5.55, 4.25)
    ax_effect.grid(axis="x", color=GRID, alpha=0.85, lw=0.45)
    ax_effect.tick_params(axis="y", length=0)
    ax_effect.text(0.012, 0.985, "Point: paired Cohen's d  ·  line: bootstrap 95% CI",
                   transform=ax_effect.transAxes, ha="left", va="top",
                   fontsize=6.7, color=MUTED)
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GREEN,
               markeredgecolor="white", markersize=5.0, label="Broadband wPLI"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ORANGE,
               markeredgecolor="white", markersize=5.0, label="Other wPLI"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE,
               markeredgecolor="white", markersize=5.0, label="Kuramoto"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GRAY,
               markeredgecolor="white", markersize=5.0, label="FCD / entropy"),
    ]
    ax_effect.legend(handles=legend_handles, loc="lower right", fontsize=6.6,
                     handlelength=1.0, borderaxespad=0.25)
    fig.savefig(FIGURES / "fig1_main_empirical.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def figure_2(wpli_mean_change):
    fig = plt.figure(figsize=(9.70, 4.80), constrained_layout=False)
    # Give the compact heatmap enough visual weight to balance the larger
    # topography panel while keeping both panels on the same title line.
    ax_heat = fig.add_axes([0.060, 0.265, 0.325, 0.525])
    ax_heat_cbar = fig.add_axes([0.392, 0.265, 0.014, 0.525])
    right = fig.add_gridspec(
        3, 2,
        left=0.438,
        right=0.985,
        bottom=0.195,
        top=0.860,
        height_ratios=[0.16, 0.68, 0.16],
        width_ratios=[1.0, 0.040],
        hspace=0.0,
        wspace=0.018,
    )
    ax_topo = fig.add_subplot(right[:, 0])
    ax_topo_cbar = fig.add_subplot(right[1, 1])

    fig.text(0.045, 0.925, "A", fontsize=10.0, fontweight="bold",
             ha="left", va="bottom", color=INK)
    fig.text(0.222, 0.890, "wPLI by band", fontsize=8.8,
             fontweight="semibold", ha="center", va="bottom", color=INK)
    fig.text(0.420, 0.925, "B", fontsize=10.0, fontweight="bold",
             ha="left", va="bottom", color=INK)
    fig.text(0.712, 0.890, "Broadband node strength", fontsize=8.8,
             fontweight="semibold", ha="center", va="bottom", color=INK)

    vmax = float(np.nanmax(np.abs(wpli_mean_change)))
    im = ax_heat.imshow(
        wpli_mean_change,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax),
        aspect="equal",
    )
    ax_heat.set_aspect("equal", adjustable="box", anchor="NW")
    ax_heat.set_xticks(np.arange(len(BANDS)))
    ax_heat.set_xticklabels(BANDS, rotation=35, ha="right", fontsize=7.0)
    ax_heat.set_yticks(np.arange(len(CONDITIONS)))
    ax_heat.set_yticklabels(CONDITIONS, fontsize=7.2)
    ax_heat.set_xticks(np.arange(-0.5, len(BANDS), 1), minor=True)
    ax_heat.set_yticks(np.arange(-0.5, len(CONDITIONS), 1), minor=True)
    ax_heat.grid(which="minor", color="white", linewidth=1.05)
    ax_heat.tick_params(which="minor", bottom=False, left=False)
    for i in range(wpli_mean_change.shape[0]):
        for j in range(wpli_mean_change.shape[1]):
            val = wpli_mean_change[i, j]
            color = "white" if abs(val) > 0.55 * vmax else INK
            ax_heat.text(j, i, f"{val:+.02f}", ha="center", va="center",
                         fontsize=6.15, color=color)
    cbar = fig.colorbar(im, cax=ax_heat_cbar)
    cbar.set_label("wPLI change", fontsize=6.75)
    cbar.ax.tick_params(labelsize=6.7)
    cbar.outline.set_linewidth(0.55)
    cbar.outline.set_edgecolor("#A7B3BD")

    maps = wpli_topography_parts()
    ax_topo.imshow(maps)
    ax_topo.set_axis_off()
    topo_norm = TwoSlopeNorm(vmin=-0.38, vcenter=0.0, vmax=0.38)
    topo_sm = ScalarMappable(norm=topo_norm, cmap="RdBu_r")
    topo_sm.set_array([])
    topo_cb = fig.colorbar(topo_sm, cax=ax_topo_cbar)
    topo_cb.set_label("Node-strength change\n(non-awake - awake)", fontsize=6.7)
    topo_cb.ax.tick_params(labelsize=6.7)
    topo_cb.outline.set_linewidth(0.55)
    topo_cb.outline.set_edgecolor("#A7B3BD")
    fig.savefig(FIGURES / "paper_fig_connectivity_overview.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def figure_3(rows_64, rows_128_full, rows_128_120, dur_64, direction_counts):
    fig = plt.figure(figsize=(7.65, 4.35), constrained_layout=False)
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[0.96, 0.72],
        width_ratios=[1.0, 1.0],
        hspace=0.38,
        wspace=0.52,
    )
    ax_mat = fig.add_subplot(gs[0, 0])
    ax_dir = fig.add_subplot(gs[0, 1])
    ax_bf = fig.add_subplot(gs[1, :])

    key_rows = [
        (("Kur_Sync", "theta"), "Theta synchrony"),
        (("Kur_Meta", "delta"), "Delta metastability"),
        (("wPLI", "broadband"), "Broadband wPLI"),
    ]
    row_map_64 = {row["key"]: row for row in rows_64}
    row_map_128_full = {row["key"]: row for row in rows_128_full}
    row_map_128_120 = {row["key"]: row for row in rows_128_120}
    matrix = np.array([
        [
            row_map_64[key]["d"],
            dur_64[key]["120s"],
            row_map_128_full[key]["d"],
            row_map_128_120[key]["d"],
        ]
        for key, _ in key_rows
    ])

    ax_mat.text(0.5, 1.035, "Effect size across analysis variants",
                transform=ax_mat.transAxes, ha="center", va="bottom",
                fontsize=8.8, fontweight="semibold", color=INK)
    vmax = float(np.nanmax(np.abs(matrix)))
    im = ax_mat.imshow(
        matrix,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax),
        aspect="equal",
    )
    ax_mat.set_xticks(np.arange(4))
    ax_mat.set_xticklabels(["64 full", "64 120 s", "128 full", "128 120 s"],
                           rotation=28, ha="right", fontsize=7.0)
    ax_mat.set_yticks(np.arange(len(key_rows)))
    ax_mat.set_yticklabels([label for _, label in key_rows], fontsize=7.2)
    ax_mat.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax_mat.set_yticks(np.arange(-0.5, len(key_rows), 1), minor=True)
    ax_mat.grid(which="minor", color="white", linewidth=0.85)
    ax_mat.tick_params(which="minor", bottom=False, left=False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = "white" if abs(val) > 0.55 * vmax else INK
            ax_mat.text(j, i, f"{val:+.2f}", ha="center", va="center",
                        color=color, fontsize=7.2)
    cb = fig.colorbar(im, ax=ax_mat, shrink=0.78, pad=0.020)
    cb.set_label("Paired Cohen's d", fontsize=7.0, labelpad=5)
    cb.ax.tick_params(labelsize=6.8)
    cb.outline.set_linewidth(0.55)
    cb.outline.set_edgecolor("#A7B3BD")
    # The colorbar narrows and recenters ax_mat. Place the panel label in
    # figure coordinates so that A remains vertically aligned with C.
    bf_pos = ax_bf.get_position()
    mat_pos = ax_mat.get_position()
    fig.text(
        bf_pos.x0 - 0.08 * bf_pos.width,
        mat_pos.y1 + 0.115 * mat_pos.height,
        "A",
        fontsize=10.0,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=INK,
    )

    set_panel_title(ax_dir, "B", "Direction consistency")
    direction_rows = [
        ("Theta synchrony", ("Kur_Sync", "theta")),
        ("Delta metastability", ("Kur_Meta", "delta")),
        ("Broadband wPLI", ("wPLI", "broadband")),
    ]
    y = np.arange(len(direction_rows))[::-1]
    ax_dir.axvline(0, color="#5E6974", lw=0.72)
    for yi, (label, key) in zip(y, direction_rows):
        pos, neg = direction_counts.get(key, (0, 0))
        ax_dir.barh(yi, -neg, color=BLUE, height=0.46, alpha=0.92)
        ax_dir.barh(yi, pos, color=ORANGE, height=0.46, alpha=0.92)
        if neg:
            if neg >= 3:
                ax_dir.text(-neg / 2, yi, f"{neg}/8 decrease", va="center",
                            ha="center", fontsize=7.0, color="white")
            else:
                ax_dir.text(-neg - 0.18, yi, f"{neg}/8 decrease", va="center",
                            ha="right", fontsize=7.0, color=BLUE)
        if pos:
            if pos >= 3:
                ax_dir.text(pos / 2, yi, f"{pos}/8 increase", va="center",
                            ha="center", fontsize=7.0, color="white")
            else:
                ax_dir.text(pos + 0.18, yi, f"{pos}/8 increase", va="center",
                            ha="left", fontsize=7.0, color=ORANGE)
    ax_dir.set_yticks(y)
    ax_dir.set_yticklabels([label for label, _ in direction_rows], fontsize=7.2)
    ax_dir.set_xlim(-8.7, 8.7)
    ax_dir.set_xticks([-8, -4, 0, 4, 8])
    ax_dir.set_xticklabels(["8", "4", "0", "4", "8"])
    ax_dir.set_xlabel("Number of paired comparisons", fontsize=7.2)
    ax_dir.grid(axis="x", color=GRID, alpha=0.85, lw=0.45)
    ax_dir.tick_params(axis="y", length=0)

    add_panel_label(ax_bf, "C", x=-0.08, y=1.035)
    ax_bf.text(0.5, 1.035, "Bayes-factor evidence",
               transform=ax_bf.transAxes, ha="center", va="bottom",
               fontsize=8.8, fontweight="semibold", color=INK)
    key_bf = [
        row_map_64[("wPLI", "broadband")],
        row_map_64[("wPLI", "delta")],
        row_map_64[("Kur_Sync", "theta")],
        row_map_64[("Kur_Meta", "delta")],
    ]
    labels = ["Broadband wPLI", "Delta wPLI", "Theta synchrony", "Delta metastability"]
    bfs = np.array([row["bf"] for row in key_bf])
    yy = np.arange(len(labels))[::-1]
    colors = [GREEN, ORANGE, BLUE, BLUE]
    ax_bf.barh(yy, bfs, color=colors, height=0.46, alpha=0.94)
    ax_bf.axvline(3, color="#A1AFBA", ls=(0, (3, 3)), lw=0.68)
    ax_bf.axvline(10, color="#A1AFBA", ls=":", lw=0.78)
    for yi, bf, row in zip(yy, bfs, key_bf):
        ax_bf.text(bf + 0.38, yi, f"BF={bf:.1f}", va="center",
                   fontsize=6.8, color=INK)
    ax_bf.set_yticks(yy)
    ax_bf.set_yticklabels(labels, fontsize=7.2)
    ax_bf.set_xlabel("BF10", fontsize=7.2)
    ax_bf.set_xlim(0, max(24, bfs.max() + 4.4))
    ax_bf.grid(axis="x", color=GRID, alpha=0.85, lw=0.45)
    style_note_box(
        ax_bf,
        "No effect survived FDR correction;\nminimum adjusted p = 0.1328.",
        (0.98, 0.04),
        ha="right",
        va="bottom",
        fontsize=6.8,
    )
    fig.savefig(FIGURES / "fig11_duration_robustness.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    rows_64 = parse_stats(STATS_64)
    rows_128_full = parse_stats(STATS_128, "Full-epoch 128-channel statistical analysis")
    rows_128_120 = parse_stats(STATS_128, "Duration-matched 128-channel statistical analysis")
    dur_64 = parse_key_duration(STATS_64)
    direction_counts = parse_direction_counts(STATS_64)
    wpli_mean_change = read_wpli_condition_means(WPLI_CONDITION_MEANS)

    FIGURES.mkdir(parents=True, exist_ok=True)
    figure_1(rows_64)
    figure_2(wpli_mean_change)
    figure_3(rows_64, rows_128_full, rows_128_120, dur_64, direction_counts)
    print("Generated:")
    print(FIGURES / "fig1_main_empirical.png")
    print(FIGURES / "paper_fig_connectivity_overview.png")
    print(FIGURES / "fig11_duration_robustness.png")


if __name__ == "__main__":
    main()
