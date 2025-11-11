import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


def plot_SVD_analysis():
    dataset_file = "../data/20251005/rand_uniform_N20000_log10Iq_dataset.npz"
    """
    Perform SVD analysis on I(q) dataset and create comprehensive visualizations.

    Creates:
    --------
    - Plot of singular values vs rank
    - Plot of first three singular vectors
    - 3D scatter plots of data projected onto first 3 singular vectors, colored by each parameter
    """

    # Load dataset
    print(f"Loading dataset from {dataset_file}...")
    data = np.load(dataset_file)
    q = data["q"]
    all_log10Iq = data["all_log10Iq"]
    all_params = data["all_params"]
    params_name = data["params_name"]

    print(f"Dataset loaded: {all_log10Iq.shape[0]} samples, {all_log10Iq.shape[1]} q-points")
    print(f"Parameters: {params_name}")

    # Perform SVD on log10(I(q)) data
    F = np.array(all_log10Iq)
    print(f"Performing SVD on data matrix of shape {F.shape}...")
    U, S, Vh = np.linalg.svd(F, full_matrices=False)
    print(f"SVD complete. Number of singular values: {len(S)}")
    print(f"Top 10 singular values: {S[:10]}")

    # ========== Figure 1: Singular Values and Vectors ==========
    fig = plt.figure(figsize=(10.0 / 3 * 1.0, 10.0 / 3 * 0.55))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # Plot singular values
    ax1.plot(range(1, len(S) + 1), S, "o--", markerfacecolor="none")
    ax1.set_xscale("log")
    ax1.set_xlabel("SVR", fontsize=9, labelpad=0)
    ax1.set_ylabel(r"$\Sigma$", fontsize=9, labelpad=0)
    ax1.tick_params(axis="both", which="both", direction="in", labelsize=7)

    # Plot first three singular vectors I(q) vs q
    for i in range(3):
        ax2.plot(q, Vh[i, :], lw=1, label=rf"$V_{i}$")
    ax2.set_xlabel(r"$Q$", fontsize=9, labelpad=0)
    ax2.set_ylabel(r"$V$", fontsize=9, labelpad=0)
    ax2.xaxis.set_major_locator(plt.MultipleLocator(4))
    ax2.xaxis.set_minor_locator(plt.MultipleLocator(2))
    ax2.tick_params(axis="both", which="both", direction="in", labelsize=7)
    ax2.legend(frameon=False, fontsize=7, loc="upper left", ncol=1, columnspacing=0.5, handlelength=1, handletextpad=0.2, labelspacing=0.1)

    ax1.text(0.8, 0.8, r"$(a)$", fontsize=9, transform=ax1.transAxes)
    ax2.text(0.8, 0.8, r"$(b)$", fontsize=9, transform=ax2.transAxes)
    plt.tight_layout(pad=0.5)
    plt.savefig("figures/SVD_SVR.png", dpi=500)
    plt.savefig("figures/SVD_SVR.pdf", dpi=500, format="pdf")
    plt.show()
    plt.close()

    # ========== Figure 2: 3D Projections colored by parameters ==========
    # Project data onto first 3 right singular vectors
    projection = np.dot(F, np.transpose(Vh))  # Shape: (n_samples, n_components)
    print(f"Projection shape: {projection.shape}")

    # Helper function to calculate nearest neighbor distance (quality metric)
    def calc_nearest_neighbor_distance(coords, colors):
        """Calculate average normalized color difference between nearest neighbors."""
        tree = cKDTree(coords)
        distances, indices = tree.query(coords, k=2)  # k=2 to get nearest neighbor (excluding self)
        color_differences = np.abs(colors - colors[indices[:, 1]])
        color_range = np.max(colors) - np.min(colors)
        if color_range == 0:
            return 0.0
        normalized_differences = color_differences / color_range * 2
        return np.mean(normalized_differences)

    # Create 3D scatter plots for each parameter
    n_params = len(params_name)
    params_tex = [r"$\phi$", r"$L$", r"$\sigma_L$", r"$\sigma_D$"]
    fig = plt.figure(figsize=(10.0 / 3 * 1.0, 10.0 / 3 * 1.0))
    axes = [fig.add_subplot(221, projection="3d"), fig.add_subplot(222, projection="3d"), fig.add_subplot(223, projection="3d"), fig.add_subplot(224, projection="3d")]

    for i in range(min(n_params, 4)):  # Limit to 4 parameters
        ax = axes[i]
        ax.set_box_aspect([1,1,1.2])
        ax.set_proj_type("persp", focal_length=0.5)
        # Get parameter values for coloring
        param_values = all_params[:, i]

        # Create scatter plot
        sc = ax.scatter(projection[:, 0], projection[:, 1], projection[:, 2], c=param_values, cmap="rainbow", s=5, edgecolors="none", rasterized=True)

        # Calculate nearest neighbor distance metric
        nnd = calc_nearest_neighbor_distance(projection, param_values)  # nnd in the full projection space

        # Labels and title
        ax.set_xlabel(r"$FV_0$", fontsize=9, labelpad=-8)
        ax.set_ylabel(r"$FV_1$", fontsize=9, labelpad=-10)
        ax.set_zlabel(r"$FV_2$", fontsize=9, labelpad=-12)

        # Colorbar
        cbar = fig.colorbar(sc, ax=ax, shrink=0.4, pad=0, aspect=17.5)
        cbar.ax.set_title(params_tex[i], fontsize=9, pad=0)
        cbar.ax.tick_params(axis="both", which="both", direction="in", labelsize=7)

        # Set viewing angle
        ax.view_init(elev=25, azim=60)
        ax.tick_params(axis="both", which="both", direction="in", labelsize=7, pad=-4)

    plt.tight_layout(pad=0.5)
    plt.savefig("figures/SVD_projection.png", dpi=500)
    plt.savefig("figures/SVD_projection.pdf", dpi=500, format="pdf")
    plt.show()

    return U, S, Vh, projection


def plot_latent_space_distribution():
    pass
