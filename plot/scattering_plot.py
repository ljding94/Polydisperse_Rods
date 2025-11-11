import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from analyze.Iq_analyze import read_Iq_from_folder


def plot_example_Iq_and_snapshot():
    # 20250929
    # prec_run0_uniform_N20000_phi0.15_meanL3.00_sigmaL0.00_sigmaD0.00

    run_type = "prec"
    run_num = 0
    pd_type = "uniform"
    N = 20000
    all_system_params = []
    for phi, meanL, sigmaL, sigmaD in [(0.15, 3.0, 0.00, 0.00), (0.15, 6.0, 0.00, 0.00), (0.15, 6.0, 0.00, 0.20), (0.30, 6.0, 0.00, 0.00)]:
        system_params = {
            "run_type": run_type,
            "run_num": run_num,
            "pd_type": pd_type,
            "N": N,
            "phi": phi,
            "meanL": meanL,
            "sigmaL": sigmaL,
            "sigmaD": sigmaD,
        }
        all_system_params.append(system_params)
    q, all_Iq, all_params, params_name = read_Iq_from_folder("../data/20250929", all_system_params)

    fig = plt.figure(figsize=(10.0 / 3 * 1.0, 10.0 / 3 * 0.6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    img = plt.imread("figures/prec_run0_uniform_N20000_phi0.15_meanL3.00_sigmaL0.00_sigmaD0.10.png")
    ax1.imshow(img)  # , extent=[0.2, 0.8, 0.2, 0.8])
    ax1.axis("off")  # Hide axis for cleaner image display
    ax1.text(0.0, -0.15, r"$\phi=0.15,L=3,$" + "\n" + r"$\sigma_L=0,\sigma_D=0.1$", fontsize=7, transform=ax1.transAxes)

    # Add colorbar to the left of ax1
    cmap = plt.cm.get_cmap("bwr")
    # Create a mock ScalarMappable for the colorbar
    norm = plt.Normalize(2.23, 3.55)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.06, 0.4, 0.01, 0.4])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.set_title(r"$V_r$", fontsize=9, pad=5)
    cbar.ax.yaxis.set_ticks_position("left")
    cbar.ax.yaxis.set_label_position("left")
    cbar.ax.tick_params(direction="in", labelsize=7)
    # Set specific ticks for the colorbar
    # cbar.set_ticks([0.7, 1.0, 1.3])

    shift = 1
    for i in range(len(all_Iq)):
        Iq = all_Iq[i]
        phi, meanL, sigmaL, sigmaD = all_params[i]
        ax2.plot(q, Iq * shift, "o-", ms=1.5, mfc="None", mew=0.5, label=f"({phi},{meanL:.0f},{sigmaD})", lw=0.5)
        shift *= 10

    ax2.set_ylim(1e-4, 3e2)
    ax2.set_yscale("log")
    ax2.set_xlabel(r"$Q$", fontsize=9, labelpad=0)
    ax2.set_ylabel(r"$I(Q)$", fontsize=9, labelpad=0)
    ax2.xaxis.set_major_locator(plt.MultipleLocator(4))
    ax2.xaxis.set_minor_locator(plt.MultipleLocator(2))
    ax2.tick_params(axis="both", which="both", direction="in", labelsize=7)
    ax2.legend(title=r"$(\phi,L,\sigma_D)$", frameon=False, fontsize=7, loc="upper right", ncol=1, columnspacing=0.5, handlelength=1, handletextpad=0.2, labelspacing=0.1)
    # Set major locator for better tick spacing

    # add annotations
    ax1.text(0.9, 0.0, r"$(a)$", fontsize=9, transform=ax1.transAxes)
    ax2.text(0.8, 0.075, r"$(b)$", fontsize=9, transform=ax2.transAxes)

    plt.tight_layout(pad=0.1)
    plt.subplots_adjust(left=0.08, right=0.99, top=0.98, bottom=0.15)
    plt.savefig("figures/Iq_and_config.png", dpi=500)
    plt.savefig("figures/Iq_and_config.pdf", dpi=500)
    plt.show()
    plt.close()


def plot_Iq_variation_params():
    # 20250929
    # prec_run0_uniform_N20000_phi0.15_meanL3.00_sigmaL0.00_sigmaD0.00
    data_folder = "../data/20250929"
    run_type = "prec"
    run_num = 0
    pd_type = "uniform"
    N = 20000
    all_system_params = []
    phi0, meanL0, sigmaL0, sigmaD0 = 0.15, 3.0, 0.00, 0.00

    fig = plt.figure(figsize=(10.0 / 3 * 1.0, 10.0 / 3 * 1))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222, sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(223, sharex=ax1, sharey=ax1)
    ax4 = fig.add_subplot(224, sharex=ax1, sharey=ax1)

    # Define variations for each parameter
    variations = [
        {"ax": ax1, "param": "phi", "values": [0.03, 0.12, 0.21, 0.30], "label": r"$\phi$", "base": {"phi": None, "meanL": meanL0, "sigmaL": sigmaL0, "sigmaD": sigmaD0}},
        {"ax": ax2, "param": "meanL", "values": [1.0, 2.0, 4.0, 6.0], "label": r"$L$", "base": {"phi": phi0, "meanL": None, "sigmaL": sigmaL0, "sigmaD": sigmaD0}},
        {"ax": ax3, "param": "sigmaL", "values": [0.0, 0.05, 0.10, 0.20], "label": r"$\sigma_L$", "base": {"phi": phi0, "meanL": meanL0, "sigmaL": None, "sigmaD": sigmaD0}},
        {"ax": ax4, "param": "sigmaD", "values": [0.0, 0.05, 0.10, 0.20], "label": r"$\sigma_D$", "base": {"phi": phi0, "meanL": meanL0, "sigmaL": sigmaL0, "sigmaD": None}},
    ]

    for var_idx, var in enumerate(variations):
        ax = var["ax"]
        param_name = var["param"]
        values = var["values"]
        legend_title = var["label"]
        base_params = var["base"]

        # Build system parameters
        all_system_params = []
        for value in values:
            base_params[param_name] = value
            system_params = {
                "run_type": run_type,
                "run_num": run_num,
                "pd_type": pd_type,
                "N": N,
                "phi": base_params["phi"],
                "meanL": base_params["meanL"],
                "sigmaL": base_params["sigmaL"],
                "sigmaD": base_params["sigmaD"],
            }
            all_system_params.append(system_params)

        # Read data and plot
        q, all_Iq, all_params, params_name = read_Iq_from_folder(data_folder, all_system_params)
        if var_idx == 0:
            print("params_name:", params_name)

        shift = 1
        for i in range(len(all_Iq)):
            Iq = all_Iq[i]
            ax.plot(q, Iq * shift, "o-", ms=2, mfc="None", mew=0.5, label=f"{values[i]}", lw=1)
            shift *= 10

        # Apply common formatting
        ax.set_yscale("log")
        # Only show x-label on bottom subplots (ax3 and ax4)
        if ax in [ax3, ax4]:
            ax.set_xlabel(r"$Q$", fontsize=9, labelpad=0)
        # Only show y-label on left subplots (ax1 and ax3)
        if ax in [ax1, ax3]:
            ax.set_ylabel(r"$I(Q)$", fontsize=9, labelpad=0)
        ax.xaxis.set_major_locator(plt.MultipleLocator(4))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(2))
        ax.tick_params(axis="both", which="both", direction="in", labelsize=7)
        # Hide x-tick labels on top subplots (ax1 and ax2)
        if ax in [ax1, ax2]:
            ax.tick_params(axis="x", which="both", labelbottom=False)
        # Hide y-tick labels on right subplots (ax2 and ax4)
        if ax in [ax2, ax4]:
            ax.tick_params(axis="y", which="both", labelleft=False)
        ax.legend(title=legend_title, frameon=False, fontsize=7, loc="upper right", ncol=2, columnspacing=0.5, handlelength=1, handletextpad=0.2, labelspacing=0.1)

    # add annotations
    ax1.text(0.1, 0.1, r"$(a)$", fontsize=9, transform=ax1.transAxes)
    ax2.text(0.1, 0.1, r"$(b)$", fontsize=9, transform=ax2.transAxes)
    ax3.text(0.1, 0.1, r"$(c)$", fontsize=9, transform=ax3.transAxes)
    ax4.text(0.1, 0.1, r"$(d)$", fontsize=9, transform=ax4.transAxes)

    plt.tight_layout(pad=0.5)
    # plt.subplots_adjust(left=0.08, right=0.99, top=0.98, bottom=0.15)
    plt.savefig("figures/Iq_variation.png", dpi=500)
    plt.savefig("figures/Iq_variation.pdf", dpi=500)
    plt.show()
    plt.close()


def plot_single_Iq():
    data_folder = "../data/20250929"
    run_type = "prec"
    run_num = 0
    pd_type = "uniform"
    N = 20000
    all_system_params = []
    phi0, meanL0, sigmaL0, sigmaD0 = 0.3, 2.0, 0.00, 0.00

    fig = plt.figure(figsize=(10.0 / 3 * 0.4, 10.0 / 3 * 0.15))
    ax = fig.add_subplot(111)
    # Define single parameter set
    system_params = {"run_type": run_type, "run_num": run_num, "pd_type": pd_type, "N": N, "phi": phi0, "meanL": meanL0, "sigmaL": sigmaL0, "sigmaD": sigmaD0}
    all_system_params.append(system_params)
    q, all_Iq, all_params, params_name = read_Iq_from_folder(data_folder, all_system_params)
    Iq = all_Iq[0]
    ax.plot(q, Iq, "o-", ms=2, mfc="None", mew=0.5, lw=1)
    ax.set_yscale("log")

    # Remove labels and ticks but keep box frame
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis='both', which='both', length=0)  # Ensure no tick marks

    # Set box frame linewidth to 1
    for spine in ax.spines.values():
        spine.set_linewidth(1)

    # Add I(Q) text and save
    text = ax.text(0.95, 0.95, r"$I(Q)$", fontsize=9, transform=ax.transAxes,
                   ha='right', va='top')
    plt.tight_layout(pad=0.1)
    plt.savefig("figures/single_Iq.png", dpi=500)
    plt.savefig("figures/single_Iq.pdf", dpi=500)

    # Replace with I'(Q) and save again
    text.set_text(r"$I'(Q)$")
    plt.savefig("figures/single_Iq_prime.png", dpi=500)
    plt.savefig("figures/single_Iq_prime.pdf", dpi=500)

    plt.show()
