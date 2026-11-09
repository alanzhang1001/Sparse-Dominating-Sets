"""Experiment displays from frozen evidence, at IEEE column/text widths.

Refresh with this script. No solver, query, method or aggregation is changed.
The general renderers import the same functions used here.
"""
from pathlib import Path
import hashlib
import json
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter
import numpy as np

from scope_v1 import MAIN_GRAPHS, GRAPH_NAMES, main_exact

ROOT = Path(__file__).resolve().parents[1]
DISPLAY = ROOT / "experiments/data/display_20261103"
QUALITY_DISPLAY = ROOT / "experiments/data/quality_theory_20261103"
BLUE = "#245C89"
TEAL = "#187568"
CHARCOAL = "#50565D"
ORANGE = "#AD6B3A"
PURPLE = "#81749B"
LIGHT = "#E5E7E9"
FAMILIES = ["dense_random", "regular", "preferential_attachment", "real_graph_sample"]
RETRIEVAL_GRAPHS = ["web-Stanford", "web-Google", "in-2004", "eu-2005"]
RETRIEVAL_NAMES = ["Stanford", "Google", "in-2004", "eu-2005"]
STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "text.color": "#24282D", "axes.labelcolor": "#24282D",
    "xtick.color": "#33383D", "ytick.color": "#33383D",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .55, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.major.width": .5, "ytick.major.width": .5,
    "axes.axisbelow": True, "pdf.fonttype": 42, "ps.fonttype": 42,
}


def save(fig, destination, stem):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination / f"exp_{stem}_v1.pdf")
    fig.savefig(destination / f"exp_{stem}_v1.png", dpi=300)
    plt.close(fig)


def clean(ax, axis="y"):
    ax.grid(axis=axis, color=LIGHT, linewidth=.5)
    ax.tick_params(pad=2)
    ax.xaxis.labelpad = 3
    ax.yaxis.labelpad = 3


def quality_values(data):
    cohort = main_exact(data["e1"])
    certified = [r for r in cohort if r["opt"] is not None]
    assert len(cohort) == 160 and len(certified) == 150
    return certified


def quality_summary_values(data):
    certified = quality_values(data)
    result = []
    for family in FAMILIES:
        for radius in [1, 2]:
            vv = sorted(r["alg"] / r["opt"] for r in certified
                        if r["family"] == family and r["radius"] == radius)
            result.append({"family": family, "radius": radius, "count": len(vv),
                           "median": st.median(vv),
                           "p90": float(np.quantile(vv, .90, method="linear")),
                           "maximum": max(vv)})
    assert sum(r["count"] for r in result) == 150
    return result


def render_quality_table(data):
    rows = quality_summary_values(data)
    QUALITY_DISPLAY.mkdir(parents=True, exist_ok=True)
    lines = ["% Derived from frozen certified instances by plot_main_figures.py.",
             r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrrrr@{}}",
             r"\toprule",
             r"Family & Counts & \multicolumn{2}{c}{$r=1$} & \multicolumn{2}{c}{$r=2$} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
             r"& & Median & Max & Median & Max \\", r"\midrule"]
    names = dict(zip(FAMILIES, ["Dense random", "Regular", "Pref. attach.", "Real samples"]))
    indexed = {(row["family"], row["radius"]): row for row in rows}
    for family in FAMILIES:
        first, second = [indexed[family, radius] for radius in [1, 2]]
        formatted = []
        for row in [first, second]:
            for key in ["median", "maximum"]:
                val = f"{row[key]:.3f}"
                if abs(row[key] - 1) < 1e-10:
                    val = r"\textbf{" + val + "}"
                formatted.append(val)
        lines.append(" & ".join([names[family], f"{first['count']}/{second['count']}"] + formatted) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular*}", ""]
    (QUALITY_DISPLAY / "quality_summary.tex").write_text("\n".join(lines))
    cert = sorted(data["certificates"], key=lambda r: r["max_degree_parameter"])
    assert len(cert) == 10
    for row in cert:
        assert row["degree_check_passed"] and row["dual_feasibility_passed"]
        assert row["ratio_matches_certified_lower_bound"]
        assert row["dual_average_lower_bound"] == row["ratio_congestion"] == (row["t"] + 1) / 2
    provenance = {
        "source_sha256": hashlib.sha256((ROOT / "experiments/data/v1/evidence.json").read_bytes()).hexdigest(),
        "certified_instances": 150, "unresolved_instances_excluded": 10,
        "quality_rows": rows,
        "p90_definition": "Linear interpolation at rank 1 + 0.9*(N-1) in the sorted certified ALG/OPT values; NumPy method=linear.",
        "theory_bar_statistics": {name: [{"count": len(v), "median": st.median(v), "min": min(v), "max": max(v)}
                                           for v in values] for name, values in theory_values(data).items()},
        "theory_encoding": "Zero-based bars show medians; whiskers retain the full min/max. ALG/LP is a light-gray auxiliary series.",
        "construction_certificates": [{"max_degree": row["max_degree_parameter"], "cost": row["ratio_congestion"]} for row in cert],
        "source_experiments_changed": False,
    }
    (QUALITY_DISPLAY / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return rows


def theory_values(data):
    dense = [r for r in data["e2"] if r["family"] == "dense_random"]
    result = {}
    for a, b, label in [("alg", "opt", "ALG/OPT"), ("opt", "lp", "OPT/LP"), ("alg", "lp", "ALG/LP")]:
        vv = [[r[a] / r[b] for r in dense if r["n"] == n and r["opt"] is not None]
              for n in [64, 128, 256]]
        assert list(map(len, vv)) == [10, 10, 9]
        result[label] = vv
    return result


def render_theory(data, destination):
    values = theory_values(data)
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(3.48, 1.68))
        ax = fig.add_axes([.12, .265, .86, .525])
        for offset, (label, fill, edge) in zip([-.22, 0, .22], [
            ("ALG/OPT", BLUE, BLUE), ("OPT/LP", ORANGE, ORANGE),
            ("ALG/LP", "#E0E3E7", "#A6ADB5")]):
            vv = values[label]
            medians = [st.median(v) for v in vv]
            xx = np.arange(3) + offset
            ax.bar(xx, medians, width=.18, color=fill, edgecolor=edge,
                   linewidth=.6, label=label, zorder=2)
            ax.errorbar(xx, medians,
                        yerr=[[m - min(v) for m, v in zip(medians, vv)],
                              [max(v) - m for m, v in zip(medians, vv)]],
                        fmt="none", ecolor=edge, elinewidth=.75, capsize=2,
                        capthick=.7, zorder=3)
        ax.axhline(1, color="#7E8791", lw=.65, ls=(0, (2, 2)), zorder=4)
        ax.set(xlim=(-.52, 2.52), ylim=(0, 3), ylabel="Cost ratio",
               xlabel="Vertices $n$  (certified instances)")
        ax.set_xticks([0, 1, 2], ["64 (10)", "128 (10)", "256 (9)"])
        ax.set_yticks([0, 1, 2, 3]); clean(ax)
        fig.legend(*ax.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.55, .995),
                   ncol=3, frameon=False, columnspacing=1.0, handletextpad=.4,
                   handlelength=1.1, borderaxespad=0)
        save(fig, destination, "theory")


def time_quality_values(data, order):
    result = {}
    for g in order:
        result[g] = {}
        for method in ["ratio", "ratio-plus"]:
            rr = [r for r in data["performance"] if r["graph_id"] == g
                  and r["method"] == method and r["implementation"] == "new" and r["radius"] == 1]
            assert len(rr) == 1 and rr[0]["repeats"] == 5
            result[g][method] = {
                "time": rr[0]["selection_compute_seconds"],
                "quality": st.median(r["average_congestion"] for r in rr),
            }
        result[g]["cc2fs"] = []
        for budget in [1, 10, 60, 300]:
            rr = [r for r in data["cc2fs"] if r["dataset_id"] == g
                  and r["budget_seconds"] == budget and not r["legacy_age"]]
            assert len(rr) == 1 and rr[0]["repeats"] == 3
            qq = [r["average_congestion"] for r in rr]
            result[g]["cc2fs"].append({
                "budget": budget, "time": rr[0]["compute_seconds"],
                "quality": st.median(qq), "min": rr[0]["quality_min"], "max": rr[0]["quality_max"],
            })
    return result


def render_time_quality(data, destination, order=MAIN_GRAPHS, names=GRAPH_NAMES):
    points = time_quality_values(data, order)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(order), figsize=(7.16, 1.52))
        fig.subplots_adjust(left=.061, right=.992, bottom=.275, top=.725, wspace=.43)
        for i, (ax, g) in enumerate(zip(axes, order)):
            cc = points[g]["cc2fs"]
            xx, yy = [p["time"] for p in cc], [p["quality"] for p in cc]
            ax.errorbar(xx, yy,
                        yerr=[[p["quality"] - p["min"] for p in cc],
                              [p["max"] - p["quality"] for p in cc]],
                        fmt="^-", color=CHARCOAL, ms=3.7, lw=1,
                        elinewidth=.8, capsize=2, capthick=.7, zorder=2)
            allq = yy + [points[g][m]["quality"] for m in ["ratio", "ratio-plus"]]
            span = max(allq) - min(allq)
            for method, color, marker, face, size in [
                ("ratio", BLUE, "o", "white", 7), ("ratio-plus", TEAL, "s", TEAL, 4)]:
                p = points[g][method]
                # A completed pass is a point, not an additional search trajectory.
                ax.plot(p["time"], p["quality"], marker, color=color,
                        mfc=face, ms=size, mew=1, zorder=4)
            ax.text(.025, .96, f"Ratio+\n{points[g]['ratio-plus']['time']:.3f} s",
                    transform=ax.transAxes, color=TEAL, fontsize=7.1, va="top", linespacing=1.0)
            ax.annotate(f"{yy[-1]:.3f}", (xx[-1], yy[-1]), (-2, 6),
                        textcoords="offset points", color=CHARCOAL, ha="right", fontsize=7,
                        bbox={"fc": "white", "ec": "none", "pad": .15, "alpha": .9})
            ax.set(xscale="log", xlim=(.035, 450),
                   ylim=(max(0, min(allq) - .16 * span), max(allq) + .22 * span),
                   title=f"({chr(97+i)}) {names[g]}")
            ax.set_xticks([.1, 1, 10, 100], ["0.1", "1", "10", "100"])
            ax.xaxis.set_minor_locator(LogLocator(base=10, subs=[2, 5], numticks=100))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.tick_params(axis="x", which="minor", length=1.6, width=.45)
            ax.grid(axis="x", which="major", color="#EEF0F2", linewidth=.4)
            ax.yaxis.set_major_locator(MaxNLocator(4, min_n_ticks=3))
            clean(ax)
        handles = [Line2D([], [], color=BLUE, marker="o", mfc="white", ms=6, lw=0, label="Ratio (completed)"),
                   Line2D([], [], color=TEAL, marker="s", ms=4, lw=0, label="Ratio+ (completed)"),
                   Line2D([], [], color=CHARCOAL, marker="^", ms=3.7, lw=1, label=r"CC$^2$FS (continued search)")]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.52, .995),
                   ncol=3, frameon=False, columnspacing=2, handlelength=1.8, borderaxespad=0, fontsize=8)
        fig.text(.52, .045, "Measured compute time (s; log scale)", ha="center", fontsize=8)
        fig.text(.012, .50, "Average congestion", va="center", rotation=90, fontsize=8)
        save(fig, destination, "time_quality")


def resource_values():
    text = (ROOT / "experiments/data/appendix_compact_20261014/implementation_phases.tex").read_text()
    result = []
    for line in text.splitlines():
        if not line.startswith(("Epinions &", "Stanford &", "Amazon &", "Google &", "YouTube &", "uk-2002 &")):
            continue
        cells = [c.strip().removesuffix(r"\\").strip() for c in line.split("&")]
        result.append({"graph": cells[0], "whole_seconds": list(map(float, cells[1].split("/"))),
                       "rss_mib": list(map(float, cells[7].split("/"))),
                       "selection_seconds": list(map(float, cells[4].split("/")))})
    assert len(result) == 6
    return result


def render_resources(destination):
    rows = resource_values()
    supplement = json.loads((ROOT / "experiments/data/resources_multiradius_20261107/summary.json").read_text())
    rows += supplement["rows"]
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(7.16, 1.72))
        for j, (key, title) in enumerate([("whole_seconds", "(a) Whole-process time (s)"),
                                         ("rss_mib", "(b) Peak RSS (GiB)")]):
            left, width = (.12, .20) if j == 0 else (.59, .22)
            ax = fig.add_axes([left, .18, width, .61])
            ax.axhspan(-.48, 1.48, color="#EAF4F1", zorder=0)
            for i, row in enumerate(rows):
                y = 7 - i
                old, new = row[key]
                relative = 100 * new / old
                ax.barh(y + .13, 100, height=.21, color="#CCD1D6", edgecolor="none")
                ax.barh(y - .13, relative, height=.21, color=TEAL, edgecolor="none")
                label = f"{old/new:.1f}×" if j == 0 else f"−{100-relative:.0f}%"
                weight = "bold" if i >= 6 else "normal"
                ax.text(relative + 2, y - .13, label, va="center", fontsize=7.0, color=TEAL, fontweight=weight)
                if j == 0:
                    value = f"{old:,.1f} / {new:,.1f}" if i >= 6 else (f"{old:.2f} / {new:.3f}" if new < 1 else f"{old:.2f} / {new:.2f}")
                else:
                    old, new = old / 1024, new / 1024
                    value = f"{old:.3f} / {new:.3f}" if old < 1 else f"{old:.2f} / {new:.2f}"
                ax.text(1.065, y, value, transform=ax.get_yaxis_transform(), va="center", fontsize=7.0, fontweight=weight)
                if j == 0:
                    a, b = row["selection_seconds"]
                    ax.text(1.80, y, f"{a/b:.1f}", transform=ax.get_yaxis_transform(), va="center", fontsize=6.8, color=CHARCOAL)
            ax.text(1.065, 7.65, "Old / new", transform=ax.get_yaxis_transform(), fontsize=7.0)
            if j == 0:
                ax.text(1.80, 7.65, "Select. ×", transform=ax.get_yaxis_transform(), fontsize=7.0)
            ax.set(xlim=(0, 103), ylim=(-.5, 7.9))
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.set_yticks(range(7, -1, -1), [r["graph"] for r in rows])
            if j:
                ax.tick_params(axis="y", labelleft=False)
            else:
                for i, label in enumerate(ax.get_yticklabels()):
                    label.set_fontsize(7.0)
                    if i >= 6: label.set_color(TEAL); label.set_fontweight("bold")
            ax.tick_params(axis="y", length=0, pad=3)
            ax.tick_params(axis="x", labelsize=7.0)
            ax.spines["left"].set_visible(False)
            clean(ax, "x")
            fig.text(left + width / 2, .85, title, ha="center", fontsize=8.1)
        fig.legend(handles=[Line2D([], [], color="#CCD1D6", lw=4, label="Reference implementation"),
                            Line2D([], [], color=TEAL, lw=4, label="Counter-and-heap implementation")],
                   loc="upper center", bbox_to_anchor=(.54, 1.01), ncol=2,
                   frameon=False, handlelength=1.3, columnspacing=2.6, borderaxespad=0, fontsize=7.4)
        fig.text(.54, .015, "Cost relative to reference (%)  |  Unmarked rows: r = 1", ha="center", fontsize=7.0)
        save(fig, destination, "resources")


def candidate_values(groups):
    budget, count = {}, {}
    for g in RETRIEVAL_GRAPHS:
        budget[g] = {}
        for method in ["degree", "cc2fs", "random"]:
            budget[g][method] = [100 * (1 - groups[g, "matched-budget", k, "ratio"]["mean"] /
                                       groups[g, "matched-budget", k, method]["mean"]) for k in [1, 2, 3]]
        count[g] = 100 * (1 - groups[g, "matched-count", 0, "ratio"]["mean"] /
                          groups[g, "matched-count", 0, "random"]["mean"])
    return budget, count


def render_candidates(groups, destination):
    raw = json.loads((ROOT / "experiments/data/e5_web_20261031/analysis.json").read_text())["rows"]
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(3.48, 1.48))
        bottom, height = .25, .48
        ax = fig.add_axes([.35, bottom, .28, height])
        for i, (g, name) in enumerate(zip(RETRIEVAL_GRAPHS, RETRIEVAL_NAMES)):
            y = 3 - i
            a, r = groups[g, "matched-count", 0, "ratio"], groups[g, "matched-count", 0, "random"]
            relative = a["mean"] / r["mean"]
            ax.plot([relative, 1], [y, y], color="#B6BDC5", lw=1.2, zorder=1)
            seeds = [v["mean_candidates"] / r["mean"] for v in raw if v["dataset_id"] == g
                     and v["control"] == "matched-count" and v["selection_method"].startswith("random-prune-")]
            assert len(seeds) == 5
            ax.scatter(seeds, y + np.linspace(-.075, .075, 5), s=8, color=PURPLE, alpha=.5, linewidths=0)
            ax.errorbar(1, y, xerr=r["sd"] / r["mean"], fmt="D", mfc="white", ms=3.5,
                        color=PURPLE, elinewidth=.75, capsize=2, mew=.8, zorder=3)
            ax.plot(relative, y, "o", color=BLUE, ms=4.8, zorder=4)
            row = bottom + height * ((y + .45) / 4)
            fig.text(.02, row, name, ha="left", va="center", fontsize=7.5)
            fig.text(.31, row, f"{a['count']:,.0f}", ha="right", va="center", fontsize=7.5)
            fig.text(.715, row, f"{a['mean']/1000:.2f}",
                     ha="center", va="center", fontsize=7.7, fontweight="bold", color=BLUE)
            fig.text(.895, row, f"{r['mean']/1000:.2f} ± {r['sd']/1000:.2f}",
                     ha="center", va="center", fontsize=7.6)
        ax.axvline(1, color="#B7BEC5", ls=(0, (2, 2)), lw=.7, zorder=0)
        ax.set(xlim=(.66, 1.06), ylim=(-.45, 3.55), xlabel="Candidates / Random-P mean")
        ax.set_xticks([.7, .8, .9, 1], ["0.7", "0.8", "0.9", "1.0"])
        ax.set_yticks([])
        ax.spines["left"].set_visible(False); clean(ax, "x")
        fig.text(.02, .82, "Graph", ha="left", fontsize=7.5)
        fig.text(.31, .82, "Common $|D|$", ha="right", fontsize=7.2)
        fig.text(.49, .82, "Relative", ha="center", fontsize=7.5)
        fig.text(.715, .82, "Ratio", ha="center", fontsize=7.6, color=BLUE)
        fig.text(.895, .82, "Random-P", ha="center", fontsize=7.6, color=PURPLE)
        fig.text(.805, .94, "Mean candidates (k)", ha="center", fontsize=7.7)
        save(fig, destination, "candidates")


def write_table(name, columns, headers, lines):
    DISPLAY.mkdir(parents=True, exist_ok=True)
    text = ["% Generated by experiments/plot_main_figures.py; frozen inputs are unchanged.",
            r"\begin{tabular}{@{}" + columns + r"@{}}", r"\toprule"]
    text += headers + ([r"\midrule"] if headers else []) + lines + [r"\bottomrule", r"\end{tabular}", ""]
    (DISPLAY / name).write_text("\n".join(text))


def render_retrieval_budget(grouped):
    """One graph per column; each cell lists low/mid/high budget values."""
    methods = ["ratio", "ratio-plus", "degree", "cc2fs", "random"]
    names = ["Ratio", "Ratio+", "Degree", r"CC$^2$FS", "Random-P"]
    rows, cells_audit = [], []
    for method, name in zip(methods, names):
        cells = []
        for graph in RETRIEVAL_GRAPHS:
            triple = []
            for budget in [1, 2, 3]:
                value = grouped[graph, "matched-budget", budget, method]["mean"]
                text = f"{value / 1000:.1f}"
                if value == min(grouped[graph, "matched-budget", budget, m]["mean"] for m in methods):
                    text = r"\textbf{" + text + "}"
                triple.append(text)
            cells.append("/".join(triple))
        rows.append((r"\rowcolor{exptint}" if method == "ratio" else "") +
                    " & ".join([name] + cells) + r" \\")
    rows.append(r"\midrule")
    gains = []
    for graph in RETRIEVAL_GRAPHS:
        triple = []
        for budget in [1, 2, 3]:
            rr = {m: grouped[graph, "matched-budget", budget, m] for m in methods}
            gain = 100 * (1 - rr["ratio"]["mean"] / rr["random"]["mean"])
            triple.append(f"{gain:.1f}")
            cells_audit.append({"graph": graph, "budget": budget, "methods": rr,
                                "reduction_vs_random_percent": gain})
        gains.append("/".join(triple))
    rows.append(" & ".join([r"Gain (\%)"] + gains) + r" \\")
    write_table("retrieval_budget.tex", "lcccc",
                [" & ".join(["Selector"] + RETRIEVAL_NAMES) + r" \\",
                 r"\multicolumn{5}{c}{Low / Mid / High index budgets} \\"], rows)
    return cells_audit


def render_display_tables(data=None, grouped=None):
    if data is None:
        data = json.loads((ROOT / "experiments/data/v1/evidence.json").read_text())
    if grouped is None:
        from render_e5_web import groups
        _, grouped = groups()
    cert = sorted(data["certificates"], key=lambda r: r["max_degree_parameter"])
    assert len(cert) == 10
    for r in cert:
        assert r["degree_check_passed"] and r["dual_feasibility_passed"] and r["ratio_matches_certified_lower_bound"]
        assert r["dual_average_lower_bound"] == r["ratio_congestion"] == (r["t"] + 1) / 2
    write_table("construction_strip.tex", "l" + "r" * 10,
                [r"$\Delta$ & " + " & ".join(str(r["max_degree_parameter"]) for r in cert) + r" \\"],
                [r"Cost & " + " & ".join(f"{r['ratio_congestion']:.1f}" for r in cert) + r" \\"])
    cells_audit = render_retrieval_budget(grouped)
    # The four displayed rows retain the existing medians and mark every true tie.
    radius_lines = []
    for line in (ROOT / "experiments/data/icde_compact/multiradius.tex").read_text().splitlines():
        if line.startswith(("Amazon &", "Stanford &")):
            cells = [c.strip().removesuffix(r"\\").strip() for c in line.split("&")]
            vv = list(map(float, cells[2:]))
            radius_lines.append(" & ".join(cells[:2] + [r"\textbf{" + c + "}" if v == min(vv) else c
                                                     for c, v in zip(cells[2:], vv)]) + r" \\")
    assert len(radius_lines) == 4
    write_table("multiradius.tex", "lrrrrr",
                [r"Graph & $r$ & Ratio & Ratio+ & Degree & Degree+ \\"], radius_lines)
    protected = [ROOT / "experiments/data/v1/evidence.json", ROOT / "experiments/data/e5_web_20261031/analysis.json",
                 ROOT / "experiments/data/appendix_compact_20261014/implementation_phases.tex",
                 ROOT / "experiments/data/icde_compact/multiradius.tex"]
    audit = {"sources_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected},
             "quality": [{"family": f, "radius": r,
                          "values": [v["alg"] / v["opt"] for v in quality_values(data) if v["family"] == f and v["radius"] == r]}
                         for f in FAMILIES for r in [1, 2]],
             "theory": theory_values(data), "time_quality": time_quality_values(data, MAIN_GRAPHS),
             "resources": resource_values(), "retrieval_matched_budget": cells_audit,
             "fixed_count": {g: {m: grouped[g, "matched-count", 0, m] for m in ["ratio", "random"]} for g in RETRIEVAL_GRAPHS},
             "candidate_reductions": candidate_values(grouped), "source_experiments_changed": False,
             "uncertainty": "Full min/max on certified theory instances; three-run min/max for CC; five-seed sample SD and individual seed means for Random-Prune."}
    (DISPLAY / "provenance.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def main():
    from render_e5_web import groups
    data = json.loads((ROOT / "experiments/data/v1/evidence.json").read_text())
    _, grouped = groups()
    render_quality_table(data)
    render_theory(data, ROOT / "figures")
    render_time_quality(data, ROOT / "figures")
    render_resources(ROOT / "figures")
    render_candidates(grouped, ROOT / "figures")
    render_display_tables(data, grouped)
    print("Refreshed four figures and four native tables. Provenance:", DISPLAY / "provenance.json")


if __name__ == "__main__":
    main()
