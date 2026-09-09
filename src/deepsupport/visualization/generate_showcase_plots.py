"""
DeepSupport Visualization Suite - Portfolio Showcase Plots (V4.1)

Generates 6 publication-grade visualizations for the DeepSupport repository showcase:
1. Sunburst: Cohort study trajectories (Program -> Outcome -> Exit Phase)
2. Treemap: Curriculum hurdle modules (Program -> Semester -> Module ECTS & Fail Rate)
3. KDE + Boxplot: Distribution shift of Motivation (V3.6 Dirac-clipping vs V4.1 kappa=20)
4. Zero-Inflated Histogram: Employment hours & empirical dropout risk curve
5. Forest Plot: Causal deconfounding comparison (Naïve Cox vs DML vs SCM Ground Truth)
6. Kaplan-Meier: Multi-universe survival dynamics S(t) (Universe A vs B) with Risk Table
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import PercentFormatter
import seaborn as sns
import plotly.express as px
import squarify
from lifelines import KaplanMeierFitter

# Style setup for academic, publication-grade figures
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#e0e0e0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.5

ROOT_DIR = Path(__file__).resolve().parents[3]
DATA_V4_A = ROOT_DIR / "data_v4_grid" / "S01_baseline" / "universe_A"
DATA_V4_B = ROOT_DIR / "data_v4_grid" / "S01_baseline" / "universe_B"
DATA_V36 = ROOT_DIR / "src" / "output_v36_clean_rerun"

IMG_OUT = ROOT_DIR / "docs" / "images"
HTML_OUT = ROOT_DIR / "docs" / "interactive"


def ensure_dirs():
    IMG_OUT.mkdir(parents=True, exist_ok=True)
    HTML_OUT.mkdir(parents=True, exist_ok=True)


def plot_01_sunburst():
    """Plot 1: Sunburst Diagram of Student Trajectories."""
    print("Generating Plot 1: Sunburst Studienverlauf...")
    df_stud = pd.read_csv(DATA_V4_A / "studierende.csv")
    df_abs = pd.read_csv(DATA_V4_A / "abschluesse.csv")
    df_sg = pd.read_csv(DATA_V4_A / "studiengaenge.csv")

    df = df_stud.merge(df_abs, on="studierenden_id").merge(df_sg, on="studiengang_id", suffixes=("", "_sg"))

    def categorize_phase(row):
        sem = row["studiendauer_semester"]
        status = row["status"]
        rsz = row["regelstudienzeit"]
        if status == "abgeschlossen":
            return "Regelstudienzeit" if sem <= rsz else "Verlängertes Studium"
        else:
            if sem <= 2:
                return "Früher Abbruch (Sem 1-2)"
            elif sem <= 4:
                return "Mittlerer Abbruch (Sem 3-4)"
            else:
                return "Später Abbruch (Sem 5+)"

    df["status_label"] = df["status"].map({
        "abgeschlossen": "Abschluss",
        "abgebrochen": "Abbruch (Freiwillig)",
        "exmatrikuliert": "Exmatrikulation (Prüfungsversagen)",
        "zeitueberschreitung": "Zeitüberschreitung (>16 Sem)"
    })
    df["phase_label"] = df.apply(categorize_phase, axis=1)

    # 1. Interactive Plotly Sunburst
    df_counts = df.groupby(["name", "status_label", "phase_label"]).size().reset_index(name="count")
    fig = px.sunburst(
        df_counts,
        path=["name", "status_label", "phase_label"],
        values="count",
        color="status_label",
        color_discrete_map={
            "Abschluss": "#2e7d32",
            "Abbruch (Freiwillig)": "#d32f2f",
            "Exmatrikulation (Prüfungsversagen)": "#e65100",
            "Zeitüberschreitung (>16 Sem)": "#6a1b9a",
            "(?)": "#757575"
        },
        title="Studienverlaufs-Hierarchie (Universe A, N=50.000)",
    )
    fig.update_layout(
        margin=dict(t=50, l=10, r=10, b=10),
        font=dict(family="DejaVu Sans", size=13)
    )
    fig.write_html(HTML_OUT / "sunburst_studienverlauf.html", include_plotlyjs="cdn")

    # 2. Static Publication Figure (Clean Cartesian Donut with tight bounding box)
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.axis("equal")

    sg_counts = df.groupby("name").size()
    df["outcome_group"] = df["status"].apply(lambda s: "Abschluss" if s == "abgeschlossen" else "Dropout")
    sg_outcome = df.groupby(["name", "outcome_group"]).size()

    cmap_sg = plt.cm.Blues(np.linspace(0.45, 0.85, len(sg_counts)))
    color_map_outcome = {"Abschluss": "#27ae60", "Dropout": "#c0392b"}

    # Inner ring: Degree Programs
    wedges_in, _ = ax.pie(
        sg_counts.values,
        labels=None,
        radius=0.72,
        colors=cmap_sg,
        wedgeprops=dict(width=0.28, edgecolor="white", linewidth=2.0),
        startangle=90
    )

    # Outer ring: Outcome per Program
    outer_vals = []
    outer_colors = []
    for (sg, outcome), count in sg_outcome.items():
        outer_vals.append(count)
        outer_colors.append(color_map_outcome[outcome])

    wedges_out, _ = ax.pie(
        outer_vals,
        labels=None,
        radius=1.0,
        colors=outer_colors,
        wedgeprops=dict(width=0.25, edgecolor="white", linewidth=1.5),
        startangle=90
    )

    # Center Text
    ax.text(0, 0, "DeepSupport V4.1\nN = 50.000\nKohortenfluss",
            ha="center", va="center", fontsize=12.5, weight="bold", color="#2c3e50")

    # Clean Legends
    legend_prog = [mpatches.Patch(facecolor=cmap_sg[i], label=f"{sg} ({sg_counts[sg]:,})") 
                   for i, sg in enumerate(sg_counts.index)]
    legend_out = [
        mpatches.Patch(facecolor="#27ae60", label="Erfolgreicher Abschluss (70.8%)"),
        mpatches.Patch(facecolor="#c0392b", label="Studienabbruch / Exmatrikulation (29.2%)")
    ]

    leg1 = ax.legend(handles=legend_prog, loc="center left", bbox_to_anchor=(1.05, 0.68), 
                     title="Studiengänge", title_fontsize=11, fontsize=9.5, frameon=True)
    ax.add_artist(leg1)
    ax.legend(handles=legend_out, loc="center left", bbox_to_anchor=(1.05, 0.28), 
              title="Studienausgang (Gesamt)", title_fontsize=11, fontsize=9.5, frameon=True)

    ax.set_title("Hierarchischer Kohortenverlauf nach Studiengang und Studienausgang",
                 fontsize=13, weight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(IMG_OUT / "sunburst_studienverlauf.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  -> Saved sunburst_studienverlauf.png & .html")


def plot_02_treemap():
    """Plot 2: Treemap Diagram of Curriculum Modules, Workload & Difficulty."""
    print("Generating Plot 2: Treemap Modul-Hürden...")
    df_mod = pd.read_csv(DATA_V4_A / "module.csv")
    df_mod_sg = pd.read_csv(DATA_V4_A / "modul_studiengang.csv")
    df_sg = pd.read_csv(DATA_V4_A / "studiengaenge.csv")
    df_pruef = pd.read_csv(DATA_V4_A / "pruefungen.csv")

    mod_stats = df_pruef.groupby("modul_id").agg(
        total_exams=("bestanden", "count"),
        fail_rate=("bestanden", lambda x: 1.0 - x.mean()),
        avg_note=("note", "mean")
    ).reset_index()

    df_full = df_mod.merge(mod_stats, on="modul_id").merge(df_mod_sg, on="modul_id").merge(df_sg[["studiengang_id", "name"]], on="studiengang_id")
    df_full["fail_pct"] = df_full["fail_rate"] * 100
    df_full["sem_label"] = "Sem " + df_full["empfohlenes_fachsemester"].astype(str)

    fig = px.treemap(
        df_full,
        path=["name_y", "sem_label", "name_x"],
        values="cp",
        color="fail_pct",
        color_continuous_scale="Reds",
        hover_data={"cp": True, "fail_pct": ":.1f", "avg_note": ":.2f", "workload_h": True},
        labels={"name_y": "Studiengang", "name_x": "Modul", "fail_pct": "Durchfallquote (%)", "cp": "ECTS (CP)"},
        title="Curriculum-Treemap: Modul-Workload (Kachelgröße) und Durchfallquote (Farbe)"
    )
    fig.update_layout(
        margin=dict(t=50, l=10, r=10, b=10),
        font=dict(family="DejaVu Sans", size=12)
    )
    fig.write_html(HTML_OUT / "treemap_modul_huerden.html", include_plotlyjs="cdn")

    top_modules = df_full.sort_values(by="fail_rate", ascending=False).head(24).copy()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    norm = plt.Normalize(vmin=top_modules["fail_pct"].min(), vmax=top_modules["fail_pct"].max())
    cmap = plt.cm.YlOrRd
    colors = [cmap(norm(v)) for v in top_modules["fail_pct"]]

    labels = [
        f"{row['name_x']}\n({row['name_y']})\n{row['cp']} CP | {row['fail_pct']:.1f}% Fail"
        for _, row in top_modules.iterrows()
    ]

    squarify.plot(
        sizes=top_modules["cp"].values,
        label=labels,
        color=colors,
        alpha=0.88,
        ax=ax,
        edgecolor="white",
        linewidth=2,
        text_kwargs={"fontsize": 8.5, "weight": "bold", "color": "#1f2937"}
    )

    ax.set_title("Curriculum-Hürden: Top-24 Module nach Durchfallquote (Fläche = ECTS Credits, Farbe = Durchfallquote)",
                 fontsize=13, weight="bold", pad=15)
    ax.axis("off")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.04, pad=0.04, aspect=40)
    cbar.set_label("Durchfallquote (%) bei Prüfungsantritt", fontsize=10, weight="bold")
    cbar.ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig(IMG_OUT / "treemap_modul_huerden.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  -> Saved treemap_modul_huerden.png & .html")


def plot_03_kde_box_motivation():
    """Plot 3: KDE & Marginal Boxplots for Motivation Shift (V3.6 vs V4.1)."""
    print("Generating Plot 3: KDE & Boxplot Motivation...")
    df_v4 = pd.read_csv(DATA_V4_A / "studierende.csv")
    df_v3 = pd.read_csv(DATA_V36 / "studierende.csv")

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.5, 1], hspace=0.15, wspace=0.2)

    ax_kde_init = fig.add_subplot(gs[0, 0])
    ax_box_init = fig.add_subplot(gs[1, 0], sharex=ax_kde_init)

    ax_kde_fin = fig.add_subplot(gs[0, 1])
    ax_box_fin = fig.add_subplot(gs[1, 1], sharex=ax_kde_fin)

    # 1. Panel A: Initiale Motivation
    sns.kdeplot(df_v3["motivation_initial"], ax=ax_kde_init, label="V3.6 (Dirac-Floor, σ=0.24)", 
                color="#e74c3c", linewidth=2.2, fill=True, alpha=0.18)
    sns.kdeplot(df_v4["motivation_initial"], ax=ax_kde_init, label="V4.1 (Empirisch κ=20, σ=0.12)", 
                color="#2980b9", linewidth=2.2, fill=True, alpha=0.18)
    ax_kde_init.axvline(0.30, color="#7f8c8d", linestyle=":", linewidth=1.5, label="Akute Gefährdung (≤0.30)")
    ax_kde_init.set_title("A: Initiale Motivation bei Immatrikulation", fontsize=12, weight="bold")
    ax_kde_init.set_ylabel("Dichte", fontsize=10)
    ax_kde_init.legend(fontsize=9, loc="upper right")
    ax_kde_init.grid(True, alpha=0.4)
    ax_kde_init.set_xlabel("")
    ax_kde_init.tick_params(labelbottom=False)

    ax_kde_init.annotate("V3.6: 10.8% in Gefahrenzone\n(Kollaps-Kandidaten)", xy=(0.20, 1.2), xytext=(0.04, 2.3),
                         arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2),
                         fontsize=8.5, weight="bold", color="#c0392b")
    ax_kde_init.annotate("V4.1: Nur 0.5% ≤ 0.30\n(Realistische Varianz)", xy=(0.50, 3.1), xytext=(0.52, 2.5),
                         arrowprops=dict(arrowstyle="->", color="#2980b9", lw=1.2),
                         fontsize=8.5, weight="bold", color="#2980b9")

    # Boxplot Initial
    data_init = [df_v3["motivation_initial"], df_v4["motivation_initial"]]
    bplot1 = ax_box_init.boxplot(data_init, vert=False, patch_artist=True, tick_labels=["V3.6", "V4.1"],
                                 widths=0.6, medianprops=dict(color="black", linewidth=2))
    bplot1["boxes"][0].set_facecolor("#e74c3c")
    bplot1["boxes"][1].set_facecolor("#2980b9")
    for box in bplot1["boxes"]:
        box.set_alpha(0.7)
    ax_box_init.set_xlabel("Motivation [0.0 - 1.0]", fontsize=10)
    ax_box_init.grid(True, alpha=0.4)

    # 2. Panel B: Finale Motivation
    sns.kdeplot(df_v3["motivation_final"], ax=ax_kde_fin, label="V3.6 Final", 
                color="#e74c3c", linewidth=2.2, fill=True, alpha=0.18)
    sns.kdeplot(df_v4["motivation_final"], ax=ax_kde_fin, label="V4.1 Final", 
                color="#2980b9", linewidth=2.2, fill=True, alpha=0.18)
    ax_kde_fin.axvline(0.05, color="#c0392b", linestyle="--", linewidth=1.5, label="Dirac-Wand (0.05)")
    ax_kde_fin.set_title("B: Finale Motivation bei Studienausgang", fontsize=12, weight="bold")
    ax_kde_fin.set_ylabel("Dichte", fontsize=10)
    ax_kde_fin.legend(fontsize=9, loc="upper right")
    ax_kde_fin.grid(True, alpha=0.4)
    ax_kde_fin.set_xlabel("")
    ax_kde_fin.tick_params(labelbottom=False)

    ax_kde_fin.annotate("Dirac-Spike bei 0.05\n(Artifizielles Clipping)", xy=(0.05, 1.8), xytext=(0.15, 2.6),
                        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2),
                        fontsize=8.5, weight="bold", color="#c0392b")

    # Boxplot Final
    data_fin = [df_v3["motivation_final"], df_v4["motivation_final"]]
    bplot2 = ax_box_fin.boxplot(data_fin, vert=False, patch_artist=True, tick_labels=["V3.6", "V4.1"],
                                 widths=0.6, medianprops=dict(color="black", linewidth=2))
    bplot2["boxes"][0].set_facecolor("#e74c3c")
    bplot2["boxes"][1].set_facecolor("#2980b9")
    for box in bplot2["boxes"]:
        box.set_alpha(0.7)
    ax_box_fin.set_xlabel("Motivation [0.0 - 1.0]", fontsize=10)
    ax_box_fin.grid(True, alpha=0.4)

    fig.suptitle("Verteilungs-Transformation der Motivation: V3.6 vs. V4.1 (N=50.000)", fontsize=14, weight="bold", y=0.98)
    plt.savefig(IMG_OUT / "kde_box_motivation_v36_vs_v41.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  -> Saved kde_box_motivation_v36_vs_v41.png")


def plot_04_histogram_erwerb():
    """Plot 4: Zero-Inflated Employment Hours & Empirical Dropout Risk."""
    print("Generating Plot 4: Erwerbstätigkeit & Dropout-Risiko...")
    df_stud = pd.read_csv(DATA_V4_A / "studierende.csv")
    df_abs = pd.read_csv(DATA_V4_A / "abschluesse.csv")
    df = df_stud.merge(df_abs, on="studierenden_id")
    df["dropout"] = (df["status"] != "abgeschlossen").astype(int)

    fig, ax1 = plt.subplots(figsize=(12, 6.5))

    bins = np.arange(0, 42, 2)
    
    df_grad = df[df["dropout"] == 0]["erwerbstaetigkeit_std"]
    df_drop = df[df["dropout"] == 1]["erwerbstaetigkeit_std"]

    ax1.hist([df_grad, df_drop], bins=bins, stacked=True, color=["#27ae60", "#e74c3c"],
             alpha=0.75, label=["Absolventen (Erfolg)", "Dropouts (Abbruch/Exm)"],
             edgecolor="white", linewidth=1.2, rwidth=0.9)
    ax1.set_xlabel("Wöchentliche Erwerbstätigkeit (Stunden)", fontsize=11, weight="bold")
    ax1.set_ylabel("Anzahl Studierende (N = 50.000)", fontsize=11, weight="bold")
    ax1.set_xlim(-1, 41)
    ax1.grid(True, alpha=0.3)

    ax1.axvline(0, color="#2c3e50", linestyle="--", linewidth=1.2)
    ax1.axvline(20, color="#d35400", linestyle="--", linewidth=1.5)
    
    ax1.text(0.5, 9500, "25.1% Vollzeit-Studium\n(0 Std / Woche)", fontsize=9, weight="bold", color="#2c3e50")
    ax1.text(20.5, 6000, "Werkstudentengrenze\n(20 Std / Woche)", fontsize=9, weight="bold", color="#d35400")

    ax2 = ax1.twinx()
    df["erwerb_bin"] = pd.cut(df["erwerbstaetigkeit_std"], bins=np.arange(0, 44, 2), right=False)
    bin_stats = df.groupby("erwerb_bin", observed=False)["dropout"].agg(["mean", "count"]).reset_index()
    bin_stats["mid"] = [interval.left + 1.0 for interval in bin_stats["erwerb_bin"]]
    
    valid_bins = bin_stats[bin_stats["count"] >= 50].dropna()

    ax2.plot(valid_bins["mid"], valid_bins["mean"] * 100, color="#8e44ad", linewidth=2.8, marker="o", 
             markersize=6.5, label="Empirische Abbruchquote (%)")
    ax2.set_ylabel("Studienabbruchquote (%)", fontsize=11, weight="bold", color="#8e44ad")
    ax2.set_ylim(0, 80)
    ax2.yaxis.set_major_formatter(PercentFormatter())
    ax2.tick_params(axis="y", labelcolor="#8e44ad")

    ax2.annotate("Steiler Risikoanstieg\n> 20h: bis zu 61% Abbruch", xy=(26, 52), xytext=(27, 35),
                 arrowprops=dict(arrowstyle="->", color="#8e44ad", lw=1.5),
                 fontsize=9.5, weight="bold", color="#8e44ad")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=10, frameon=True)

    plt.title("Zero-Inflated Erwerbstätigkeit und empirischer Gradient des Studienabbruchs",
              fontsize=13, weight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(IMG_OUT / "histogram_erwerb_zero_inflated.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  -> Saved histogram_erwerb_zero_inflated.png")


def plot_05_forest_plot_kausal():
    """Plot 5: Causal Forest Plot (Hazard Ratios & Relative Risks across Estimators)."""
    print("Generating Plot 5: Kausal-Forest-Plot...")
    estimators = [
        {"name": "Naives Cox PH (Fachlich, Panel)", "stat": 1.199, "ci_lower": 1.141, "ci_upper": 1.259, "type": "Biased (HR > 1)", "color": "#c0392b"},
        {"name": "Naives Cox PH (Überfachlich, Panel)", "stat": 1.006, "ci_lower": 0.958, "ci_upper": 1.056, "type": "Biased (HR ≈ 1)", "color": "#e67e22"},
        {"name": "Naives Cox PH (Psychosozial, Panel)", "stat": 1.010, "ci_lower": 0.961, "ci_upper": 1.061, "type": "Biased (HR ≈ 1)", "color": "#e67e22"},
        {"name": "Stratifiziertes Cox (High-Risk Motivation < 0.40)", "stat": 0.992, "ci_lower": 0.932, "ci_upper": 1.056, "type": "Partiell deconfounded", "color": "#f39c12"},
        {"name": "DML Orthogonalisiert (Fachlich, RR)", "stat": 0.964, "ci_lower": 0.931, "ci_upper": 0.998, "type": "Deconfounded (DML)", "color": "#2980b9"},
        {"name": "DML Orthogonalisiert (Psychosozial, RR)", "stat": 0.955, "ci_lower": 0.922, "ci_upper": 0.989, "type": "Deconfounded (DML)", "color": "#2980b9"},
        {"name": "Transformer DML (Überfachlich, RR)", "stat": 0.887, "ci_lower": 0.852, "ci_upper": 0.923, "type": "Deconfounded (DML)", "color": "#16a085"},
        {"name": "SCM Counterfactual Ground Truth (A vs B Gesamt, RR)", "stat": 0.786, "ci_lower": 0.772, "ci_upper": 0.800, "type": "Kausale Ground Truth", "color": "#27ae60"}
    ]

    df_forest = pd.DataFrame(estimators)
    y_pos = np.arange(len(df_forest))

    fig, ax = plt.subplots(figsize=(12, 7.5))

    ax.axvline(1.0, color="#34495e", linestyle="--", linewidth=1.5, label="Null-Effekt (HR / RR = 1.0)")
    
    ax.axvspan(0.70, 1.0, color="#27ae60", alpha=0.07)
    ax.axvspan(1.0, 1.34, color="#c0392b", alpha=0.07)

    for idx, row in df_forest.iterrows():
        err_left = row["stat"] - row["ci_lower"]
        err_right = row["ci_upper"] - row["stat"]
        ax.errorbar(row["stat"], idx, xerr=[[err_left], [err_right]], fmt="o", color=row["color"],
                    ecolor=row["color"], elinewidth=2.4, capsize=5, capthick=1.6, markersize=8.5)
        
        val_text = f"{row['stat']:.3f} [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"
        ax.text(row["ci_upper"] + 0.015, idx, val_text, va="center", fontsize=9.5, weight="bold", color=row["color"])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_forest["name"], fontsize=9.5, weight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Effektschätzer: Hazard Ratio (Cox) / Relatives Risiko RR (DML & Ground Truth)", fontsize=10.5, weight="bold")
    ax.set_xlim(0.70, 1.34)
    ax.grid(True, axis="x", alpha=0.4)

    # Position legend safely at upper left
    legend_handles = [
        mpatches.Patch(facecolor="#c0392b", label="Naives Cox: Confounding by Indication (Scheinbare Verschlechterung)"),
        mpatches.Patch(facecolor="#f39c12", label="Stratifizierung: Dämpfung des Selektionsbias"),
        mpatches.Patch(facecolor="#2980b9", label="Double Machine Learning (DML): Aufdeckung wahrer Protektion"),
        mpatches.Patch(facecolor="#27ae60", label="SCM Ground Truth: Kontrafaktischer Makroeffekt (ARR = 7.95 pp)")
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.02, 0.98), fontsize=8.5, frameon=True)

    plt.title("Kausales Deconfounding: Auflösung des Dropout-Paradoxons über Modellklassen",
              fontsize=13, weight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(IMG_OUT / "forest_plot_kausal_vergleich.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  -> Saved forest_plot_kausal_vergleich.png")


def plot_06_kaplan_meier():
    """Plot 6: Kaplan-Meier Survival Analysis (Universe A vs B) with Risk Table."""
    print("Generating Plot 6: Kaplan-Meier Survival A vs B...")
    df_a = pd.read_csv(DATA_V4_A / "abschluesse.csv")
    df_b = pd.read_csv(DATA_V4_B / "abschluesse.csv")

    T_a = df_a["studiendauer_semester"]
    E_a = (df_a["status"] != "abgeschlossen").astype(int)

    T_b = df_b["studiendauer_semester"]
    E_b = (df_b["status"] != "abgeschlossen").astype(int)

    kmf_a = KaplanMeierFitter().fit(T_a, E_a, label="Universum A (Full Support, N=50.000)")
    kmf_b = KaplanMeierFitter().fit(T_b, E_b, label="Universum B (No Support, N=50.000)")

    fig = plt.figure(figsize=(12, 7.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1.2], hspace=0.28)

    ax_surv = fig.add_subplot(gs[0])
    ax_table = fig.add_subplot(gs[1])

    kmf_a.plot_survival_function(ax=ax_surv, color="#27ae60", linewidth=2.5, ci_show=True)
    kmf_b.plot_survival_function(ax=ax_surv, color="#c0392b", linewidth=2.5, ci_show=True)

    ax_surv.set_xlim(1, 16.5)
    ax_surv.set_ylim(0.0, 1.02)
    ax_surv.set_ylabel("Überlebenswahrscheinlichkeit S(t)", fontsize=11, weight="bold")
    ax_surv.set_title("Kaplan-Meier Überlebensanalyse: Universum A (Support) vs. Universum B (Kein Support)",
                      fontsize=13, weight="bold", pad=12)
    ax_surv.grid(True, alpha=0.35)

    ax_surv.axvline(6, color="#7f8c8d", linestyle=":", linewidth=1.5)
    ax_surv.text(6.1, 0.45, "Regelstudienzeit\nBWL & Inf (6 Sem)", fontsize=8.5, color="#555555")

    ax_surv.axvline(8, color="#7f8c8d", linestyle=":", linewidth=1.5)
    ax_surv.text(8.1, 0.35, "Regelstudienzeit\nMaschb (8 Sem)", fontsize=8.5, color="#555555")

    ax_surv.annotate("Spreizung: +7.0 pp\nS_A(8)=71.5% vs S_B(8)=64.5%", xy=(8, 0.68), xytext=(9.2, 0.78),
                     arrowprops=dict(arrowstyle="->", color="#2980b9", lw=1.5),
                     fontsize=9.5, weight="bold", color="#2980b9")
    
    ax_surv.annotate("Verdopplung des Fortbestehens\nin Spätphase: S_A(15)=50.6% vs S_B(15)=36.0%", xy=(15.0, 0.50), xytext=(9.5, 0.20),
                     arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.5),
                     fontsize=9.5, weight="bold", color="#27ae60")

    ax_surv.legend(loc="lower left", fontsize=10, frameon=True)
    ax_surv.tick_params(labelbottom=True)
    ax_surv.set_xlabel("Fachsemester (t)", fontsize=11, weight="bold")

    times = [1, 2, 4, 6, 8, 10, 12, 14, 16]
    risk_a = [np.sum(T_a >= t) for t in times]
    risk_b = [np.sum(T_b >= t) for t in times]

    table_data = [
        [f"{v:,}" for v in risk_a],
        [f"{v:,}" for v in risk_b]
    ]
    
    ax_table.axis("off")
    table = ax_table.table(
        cellText=table_data,
        rowLabels=["At Risk (A)", "At Risk (B)"],
        colLabels=[f"Sem {t}" for t in times],
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="#1a252f")
            cell.set_facecolor("#ecf0f1")
        elif row == 1:
            cell.set_text_props(color="#27ae60")
        elif row == 2:
            cell.set_text_props(color="#c0392b")

    plt.savefig(IMG_OUT / "kaplan_meier_survival_a_vs_b.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  -> Saved kaplan_meier_survival_a_vs_b.png")


def main():
    print("=== Starte DeepSupport Visualisierungs-Suite (V4.1 Showcase) ===")
    ensure_dirs()
    plot_01_sunburst()
    plot_02_treemap()
    plot_03_kde_box_motivation()
    plot_04_histogram_erwerb()
    plot_05_forest_plot_kausal()
    plot_06_kaplan_meier()
    print("=== Alle 6 Showcase-Plots erfolgreich generiert! ===")


if __name__ == "__main__":
    main()
