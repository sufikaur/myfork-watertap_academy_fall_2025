from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.interpolate import griddata
import numpy as np
import pandas as pd


# Plot results
def plot_sweep_results(
    recoverys, lcows, pressure, base_recovery, base_lcow, base_pressure
):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(recoverys, lcows, ls=":", marker=".")
    ax1.scatter(
        [base_recovery],
        [base_lcow],
        color="red",
        label="Base Case",
        marker="*",
        edgecolor="k",
        s=200,
    )

    ax1.set(
        xlabel="Recovery (%)",
        ylabel="Levelized Cost of Water ($/m³)",
    )
    ax1.set_title("LCOW vs. Recovery")
    ax1.legend()

    ax2.plot(recoverys, pressure, ls=":", marker=".")
    ax2.scatter(
        [base_recovery],
        [base_pressure],
        color="red",
        label="Base Case",
        marker="*",
        edgecolor="k",
        s=200,
    )
    ax2.set(
        xlabel="Recovery (%)",
        ylabel="Pressure (bar)",
    )
    ax2.set_title("Pressure vs. Recovery")
    ax2.legend()


def make_stacked_plot(
    file_name="parameter_sweep_results.csv",
    parameter="Water Recovery",
    units=["RO", "ERD", "Pump"],
    unit_colors=["#1f77b4", "#ff7f0e", "#2ca02c"],
    elec_color="#db2020ab",
    opex_hatch="//",
    capex_hatch="..",
    elec_hatch="++",
    fontsize=14,
):
    """
    Create stacked plot of LCOW contributions from single parameter sweep results.
    """

    ################################################

    df = pd.read_csv(file_name)
    df.set_index(f"# {parameter}", inplace=True)
    df.sort_values(by=f"# {parameter}", inplace=True)

    unit_color_dict = dict(zip(units, unit_colors))

    capex_lcow = defaultdict(list)
    opex_lcow = defaultdict(list)
    agg_flow_lcow = defaultdict(list)

    capex_lcow_rel = defaultdict(list)
    opex_lcow_rel = defaultdict(list)
    agg_flow_lcow_rel = defaultdict(list)

    for _, row in df.iterrows():
        lcow = row["LCOW"]
        for u in units:
            c = row[f"LCOW Direct CAPEX {u}"] + row[f"LCOW Indirect CAPEX {u}"]
            o = row[f"LCOW Fixed OPEX {u}"]
            capex_lcow[u].append(c)
            opex_lcow[u].append(o)

            capex_lcow_rel[u].append(c / lcow)
            opex_lcow_rel[u].append(o / lcow)

        agg_flow_lcow["electricity"].append(row["LCOW Variable OPEX Electricity"])
        agg_flow_lcow_rel["electricity"].append(
            row["LCOW Variable OPEX Electricity"] / lcow
        )

    ################################################
    # Collecting data for stacked plot

    stacked_cols = list()
    stacked_cols_rel = list()
    stacked_hatch = list()
    stacked_colors = list()

    stacked_cols.append(agg_flow_lcow["electricity"])
    stacked_cols_rel.append(agg_flow_lcow_rel["electricity"])
    stacked_hatch.append(elec_hatch)
    stacked_colors.append(elec_color)

    for u in units:
        stacked_cols.append(opex_lcow[u])
        stacked_cols_rel.append(opex_lcow_rel[u])
        stacked_colors.append(unit_color_dict[u])
        stacked_hatch.append(opex_hatch)

        stacked_cols.append(capex_lcow[u])
        stacked_cols_rel.append(capex_lcow_rel[u])
        stacked_colors.append(unit_color_dict[u])
        stacked_hatch.append(capex_hatch)

    ################################################
    # Plotting absolute LCOW contributions

    fig, (ax, ax1) = plt.subplots(1, 2, figsize=(14, 6))

    ax.stackplot(
        df.index,
        stacked_cols,
        colors=stacked_colors,
        edgecolor="black",
        hatch=stacked_hatch,
    )

    legend_elements = [
        Patch(facecolor="white", hatch=capex_hatch, label="CAPEX", edgecolor="k"),
        Patch(facecolor="white", hatch=opex_hatch, label="OPEX", edgecolor="k"),
        Patch(
            facecolor=elec_color,
            label="Electricity",
            hatch=elec_hatch,
            edgecolor="k",
        ),
    ] + [Patch(facecolor=unit_color_dict[u], label=u, edgecolor="k") for u in units]

    leg_kwargs = dict(
        loc="lower left",
        frameon=False,
        ncol=4,
        handlelength=1,
        handleheight=1,
        labelspacing=0.2,
        columnspacing=0.9,
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        mode="expand",
    )
    ax.legend(handles=legend_elements, **leg_kwargs)
    ax.set_xlabel(f"{parameter}", fontsize=fontsize)
    ax.set_ylabel("LCOW ($/m³)", fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=fontsize)

    ################################################

    # Plotting relative LCOW contributions

    ax1.stackplot(
        df.index,
        stacked_cols_rel,
        colors=stacked_colors,
        edgecolor="black",
        hatch=stacked_hatch,
    )

    legend_elements = [
        Patch(facecolor="white", hatch=capex_hatch, label="CAPEX", edgecolor="k"),
        Patch(facecolor="white", hatch=opex_hatch, label="OPEX", edgecolor="k"),
        Patch(
            facecolor=elec_color,
            label="Electricity",
            hatch=elec_hatch,
            edgecolor="k",
        ),
    ] + [Patch(facecolor=unit_color_dict[u], label=u, edgecolor="k") for u in units]

    ax1.legend(handles=legend_elements, **leg_kwargs)
    # ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x*100)}%"))
    ax1.set_xlabel(f"{parameter}", fontsize=fontsize)
    ax1.set_ylabel("Relative LCOW", fontsize=fontsize)
    ax1.tick_params(axis="both", labelsize=fontsize)


def make_contour_plot(
    df,
    x="# Pressure",
    y="Area",
    z="LCOW",
    x_adj=1,
    y_adj=1,
    z_adj=1,
    cmap="turbo",
    interp_method="cubic",
    grid_len=100,
    levels=20,
    levelsf=None,
    set_dict=dict(),
    cb_title="LCOW ($/m³)",
    cb_fontsize=14,
    contour_label_fmt="  %#.2e  ",
    add_contour_labels=False,
    fig=None,
    ax=None,
    figsize=(6, 4),
):
    """
    Plot contour plot from two parameter sweep.
    """
    # fig, ax = plt.subplots(figsize=figsize)
    # fig.set_size_inches(figsize[0], figsize[1], forward=True)

    x = df[x] * x_adj
    y = df[y] * y_adj
    z = df[z] * z_adj

    xi = np.linspace(min(x), max(x), grid_len)
    yi = np.linspace(min(y), max(y), grid_len)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), z, (xi, yi), method=interp_method)

    contourf = ax.contourf(xi, yi, zi, levels=levels, cmap=cmap)
    _ = plt.colorbar(contourf, label=cb_title)

    ax.set(**set_dict)

    if add_contour_labels:
        contour = ax.contour(
            contourf,
            levelsf,
            colors="k",
            linestyles="dashed",
        )
        ax.clabel(contour, colors="black", fmt=contour_label_fmt, fontsize=cb_fontsize)

    fig.tight_layout()

    return fig, ax
