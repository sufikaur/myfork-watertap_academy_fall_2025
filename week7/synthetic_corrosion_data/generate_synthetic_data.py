import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_synthetic_potential_difference(T, do, min_val=-0.3, max_val=0.6):
    """
    Inputs: T (C), w (mass fraction), do (mg/L)
    Generates synthetic localized corrosion metric based on desired trends - NOT actual values
    """

    # Effect of temperature - strong inverse
    T_min = 25
    T_max = 95
    weight_T = 0.7
    fT = np.clip(1 - weight_T * (T - T_min) / (T_max - T_min), 0.001, 1)

    # Effect of DO - about 100% reduction from 0 to 1 mg/L, mostly steady from 1-8 mg/L
    do_min = 0
    do_max = 8
    do_mid = 1
    weight_do_a = 0.55
    weight_do_b = 0.05
    do_a = np.clip(do / (do_mid - do_min), 0.01, 1)
    do_b = np.clip((do - do_mid) / (do_max - do_mid), 0.001, 1)
    fdo = 1 - weight_do_a * do_a - weight_do_b * do_b

    # Combine effects
    g = fT * fdo
    y = min_val + (max_val - min_val) * g

    return y


def synthetic_parameter_sweep(T_range, do_range):
    results = {}
    results["temperature_C"] = []
    results["do_mg_L"] = []
    results["synthetic_potential_difference_V"] = []
    for t in T_range:
        for d in do_range:
            results["temperature_C"].append(t)
            results["do_mg_L"].append(d)
            results["synthetic_potential_difference_V"].append(
                get_synthetic_potential_difference(t, d)
            )
    df = pd.DataFrame(results)
    df.to_csv(
        "week7/synthetic_corrosion_data/synthetic_potential_difference.csv", index=False
    )


def plot_synthetic_data_vs_do(temps=[45, 95], ax=None):
    # Load data
    survey_path = "week7/synthetic_corrosion_data/synthetic_potential_difference.csv"
    data = pd.read_csv(survey_path)
    cmap = plt.get_cmap("Blues")
    colors = [cmap(i) for i in np.linspace(0.4, 1, len(temps))]

    if ax is None:
        fig, ax = plt.subplots(figsize=(3.25, 3.25))
    ax.axhline(y=0, color="k")
    for i, t in enumerate(temps):
        data_t = data[data["temperature_C"] == t]
        ax.plot(
            data_t["do_mg_L"],
            data_t["synthetic_potential_difference_V"],
            marker=".",
            color=colors[i],
            markerfacecolor="white",
            markeredgecolor=colors[i],
            label=f"{t} C",
        )
    ax.legend(frameon=False)
    ax.set_xlabel("Dissolved oxygen (mg/L)")
    ax.set_ylabel(r"$V_{r}-V_{c}$ (V (SHE))")
    ax.set_xlim([0, 8])
    ax.set_ylim([-0.3, 0.6])
    ax.fill_between(
        ax.get_xlim(), ax.get_ylim()[0], 0, color="indianred", alpha=0.25, zorder=0
    )
    ax.text(
        0.5,
        -0.25,
        "Localized Corrosion",
        ha="left",
        va="center",
        fontsize=9,
        color="black",
    )
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8])
    ax.set_yticks([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    plt.tight_layout()
    plt.savefig("week7/synthetic_corrosion_data/synthetic_data_vs_do.svg")


def plot_synthetic_data_vs_temp(do=[0, 1, 8], ax=None):
    # Load data
    survey_path = "week7/synthetic_corrosion_data/synthetic_potential_difference.csv"
    data = pd.read_csv(survey_path)
    cmap = plt.get_cmap("Greens")
    colors = [cmap(i) for i in np.linspace(0.4, 1, len(do))]
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.25, 3.25))
    ax.axhline(y=0, color="k")
    for i, t in enumerate(do):
        data_t = data[data["do_mg_L"] == t]
        ax.plot(
            data_t["temperature_C"],
            data_t["synthetic_potential_difference_V"],
            marker=".",
            color=colors[i],
            markerfacecolor="white",
            markeredgecolor=colors[i],
            label=f"{t} mg/L",
        )
    ax.legend(frameon=False)
    ax.set_xlabel("Temperature (C)")
    ax.set_ylabel(r"$V_{r}-V_{c}$ (V (SHE))")
    ax.set_xlim([25, 95])
    ax.set_ylim([-0.3, 0.6])
    ax.fill_between(
        ax.get_xlim(), ax.get_ylim()[0], 0, color="indianred", alpha=0.25, zorder=0
    )
    ax.text(
        30,
        -0.25,
        "Localized Corrosion",
        ha="left",
        va="center",
        fontsize=9,
        color="black",
    )
    ax.set_xticks([25, 35, 45, 55, 65, 75, 85, 95])
    ax.set_yticks([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    plt.tight_layout()
    plt.savefig("week7/synthetic_corrosion_data/synthetic_data_vs_temperature.svg")


def plot_synthetic_data_comparison():
    lcow_original = [4.875879615834827, 4.875879615834827, 4.875879615834827]
    lcow_ss = [3.75, 3.88, 0]
    lcow_ni = [4.59, 4.78, 4.78]
    plt.figure(figsize=(3.25, 3.25))

    labels = ["0.5 mg/L", "1.0 mg/L", "8.0 mg/L"]
    w = 0.25
    x = np.array([0, 1, 2])

    # Plot LCOW
    fig, ax = plt.subplots(figsize=(3.25, 3.25))
    b1 = ax.bar(x - w, lcow_original, width=w, label="No corrosion", color="#c7e9b4")
    b2 = ax.bar(x, lcow_ss, width=w, label="Demo material", color="#41b6c4")
    b2 = ax.bar(x + w, lcow_ni, width=w, label="Ni625", color="#0c2c84")

    ax.set_xticks(x, labels)
    plt.ylim([0, 6])
    ax.set_xlabel("Dissolved oxygen")
    ax.set_ylabel("LCOW ($/m3)")
    ax.legend(frameon=False, loc="best", ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig("week7/synthetic_corrosion_data/synthetic_lcow_comparison.png")


if __name__ == "__main__":
    T_range = np.arange(25, 100, 5)
    do_range = np.arange(0, 8.5, 0.5)
    synthetic_parameter_sweep(T_range, do_range)
    plot_synthetic_data_vs_do([50, 70, 90])
    plot_synthetic_data_vs_temp()

    # make panel figure
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 3.25))
    plot_synthetic_data_vs_do([50, 70, 90], ax=axs[0])
    plot_synthetic_data_vs_temp(ax=axs[1])
    fig.tight_layout()
    plt.savefig("week7/corrosion_demo_figures/synthetic_material_behavior.png")
