import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.interpolate import make_smoothing_spline
from scipy.spatial import cKDTree
from scipy.signal import savgol_filter
# Add parent directory to path so we can import from there
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "simulation"))
from IQ import build_FQalpha_interpolator, FQalpha, PQ_single_rod_V2
from rods_tool import create_file_label


def plot_Iqxy_single_rod_exact(L, theta, phi, D=1.0):
    """
    Plot I(qx, qy, qz=0) for a single rod of length L pointing in direction
    (sin(theta)cos(phi), sin(theta)sin(phi), cos(theta))
    Using direct analytical form factor calculation
    """
    # Create q-space grid for qx, qy (qz = 0)
    q_max = 20.0
    q_points = 200
    qx = np.linspace(-q_max, q_max, q_points)
    qy = np.linspace(-q_max, q_max, q_points)
    QX, QY = np.meshgrid(qx, qy)

    # Rod orientation vector
    rod_x = np.sin(theta) * np.cos(phi)
    rod_y = np.sin(theta) * np.sin(phi)
    rod_z = np.cos(theta)

    # q vector for each point (qz = 0)
    QZ = np.zeros_like(QX)

    # Calculate q magnitude for each grid point
    q_mag = np.sqrt(QX**2 + QY**2)

    # Calculate angle between q and rod for each grid point
    # cos(angle) = (q · rod) / (|q| * |rod|), |rod| = 1
    q_dot_rod = QX * rod_x + QY * rod_y + QZ * rod_z
    cos_angle = np.divide(q_dot_rod, q_mag, out=np.zeros_like(q_mag), where=q_mag != 0)
    angle_alpha = np.arccos(np.clip(np.abs(cos_angle), 0, 1))  # Use absolute value for symmetry

    # Calculate I(q) = |F(q)|^2 using direct analytical form factor
    I_xy = np.zeros_like(QX)

    # Avoid q=0 singularity
    valid_mask = q_mag > 1e-6

    if np.any(valid_mask):
        # Calculate form factor for all valid points
        q_valid = q_mag[valid_mask]
        alpha_valid = angle_alpha[valid_mask]

        # Vectorized calculation of form factor
        Fq_vals = np.zeros_like(q_valid)

        for i, (q_val, alpha_val) in enumerate(zip(q_valid, alpha_valid)):
            try:
                Fq_vals[i] = FQalpha(q_val, alpha_val, D, L)
            except:
                # Fallback to analytical form for spherocylinder
                # For a rod along z-axis: F(q) = sinc(qL*cos(alpha)/2) * bessel_J1(qR*sin(alpha))/(qR*sin(alpha)/2)
                qL_cos = q_val * L * np.cos(alpha_val)
                qR_sin = q_val * (D / 2) * np.sin(alpha_val)

                # Longitudinal part (sinc function)
                if np.abs(qL_cos) < 1e-10:
                    F_long = 1.0
                else:
                    F_long = np.sin(qL_cos / 2) / (qL_cos / 2)

                # Transverse part (Bessel function)
                if np.abs(qR_sin) < 1e-10:
                    F_trans = 1.0
                else:
                    from scipy.special import j1

                    F_trans = 2 * j1(qR_sin) / qR_sin

                Fq_vals[i] = F_long * F_trans

        # Calculate intensity I = |F|^2
        I_xy[valid_mask] = Fq_vals**2

    # Handle q=0 case
    I_xy[~valid_mask] = 1.0

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 2D intensity plot
    vmin = np.log10(np.maximum(I_xy[I_xy > 0].min(), 1e-6))
    vmax = np.log10(I_xy.max())
    im = ax1.pcolormesh(QX, QY, np.log10(I_xy), cmap="jet", shading="auto", vmin=vmin, vmax=vmax)
    contours = ax1.contour(QX, QY, np.log10(I_xy), levels=10, colors="white", alpha=0.5, linewidths=0.5)
    ax1.set_xlabel("qx", fontsize=12)
    ax1.set_ylabel("qy", fontsize=12)
    ax1.set_title(f"log₁₀(I(qx, qy, qz=0)) - Single Rod\nL={L}, D={D}, θ={theta:.2f}, phi={phi:.2f}", fontsize=12)
    ax1.set_aspect("equal")
    plt.colorbar(im, ax=ax1, label="log₁₀(Intensity)")

    # Add arrow showing rod orientation projection in xy plane
    arrow_scale = q_max * 0.15
    ax1.arrow(0, 0, rod_x * arrow_scale, rod_y * arrow_scale, head_width=q_max * 0.02, head_length=q_max * 0.03, fc="red", ec="red", linewidth=2)
    ax1.text(rod_x * arrow_scale * 1.3, rod_y * arrow_scale * 1.3, "Rod proj.", fontsize=10, color="red", ha="center", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    # Radial cut plot showing orientationally averaged I(q) vs q
    q_radial = np.logspace(-1, np.log10(50), 500)

    # Calculate orientational average I(q) for each q value
    I_avg = []
    n_alpha_points = 100  # Number of alpha values for averaging
    alpha_avg_values = np.linspace(0, np.pi, n_alpha_points)

    for q in q_radial:
        # Average over all orientations (alpha angles)
        I_sum = 0.0
        weight_sum = 0.0

        for alpha_val in alpha_avg_values:
            try:
                Fq_val = FQ(q, alpha_val, D, L)
                # Weight by sin(alpha) for proper spherical averaging
                weight = np.sin(alpha_val)
                I_sum += (Fq_val**2) * weight
                weight_sum += weight
            except:
                continue

        if weight_sum > 0:
            I_avg.append(I_sum / weight_sum)
        else:
            I_avg.append(0.0)

    ax2.plot(q_radial, I_avg, linewidth=2, label="Orientational Average")

    ax2.set_xlabel("q", fontsize=12)
    ax2.set_ylabel("I(q)", fontsize=12)
    ax2.set_title("Orientationally Averaged Intensity", fontsize=12)
    ax2.set_yscale("log")
    ax2.set_xscale("log")
    ax2.set_ylim(bottom=1e-6)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"Iq_xy_single_rod_exact_L_{L:.0f}_theta_{theta:.1f}_phi_{phi:.1f}.png")
    plt.show()

    return QX, QY, I_xy


def plot_Iqxy_single_rod_interp(L, theta, phi, D=1.0):
    """
    Plot I(qx, qy, qz=0) for a single rod of length L pointing in direction
    (sin(theta)cos(phi), sin(theta)sin(phi), cos(theta))
    Using interpolated form factor calculation for speed
    """
    # Create q-space grid for qx, qy (qz = 0)
    q_max = 20.0
    q_points = 200
    qx = np.linspace(-q_max, q_max, q_points)
    qy = np.linspace(-q_max, q_max, q_points)
    QX, QY = np.meshgrid(qx, qy)

    # Rod orientation vector
    rod_x = np.sin(theta) * np.cos(phi)
    rod_y = np.sin(theta) * np.sin(phi)
    rod_z = np.cos(theta)

    # q vector for each point (qz = 0)
    QZ = np.zeros_like(QX)

    # Calculate q magnitude for each grid point
    q_mag = np.sqrt(QX**2 + QY**2)

    # Calculate angle between q and rod for each grid point
    # cos(angle) = (q · rod) / (|q| * |rod|), |rod| = 1
    q_dot_rod = QX * rod_x + QY * rod_y + QZ * rod_z

    cos_angle = np.divide(q_dot_rod, q_mag, out=np.zeros_like(q_mag), where=q_mag != 0)
    angle_alpha = np.arccos(np.clip(np.abs(cos_angle), 0, 1))  # Use absolute value for symmetry

    # Build the form factor interpolator
    q_values = np.linspace(0.1, q_max * 1.5, 1000)  # Extended range for better interpolation
    alpha_values = np.linspace(0, np.pi, 1000)  # Higher resolution for better accuracy
    L_values = np.array([L])

    print(f"Building interpolator for L={L}, D={D}...")
    Fq_interp, _ = build_FQ_interpolator(q_values, alpha_values, L_values)

    # Calculate I(q) = |F(q)|^2 using interpolated form factor
    I_xy = np.zeros_like(QX)

    # Avoid q=0 singularity and stay within interpolation bounds
    valid_mask = (q_mag > 0.1) & (angle_alpha <= np.pi)

    if np.any(valid_mask):
        # Get valid points for vectorized interpolation
        q_valid = q_mag[valid_mask]
        alpha_valid = angle_alpha[valid_mask]

        print(f"Interpolating {len(q_valid)} points...")

        try:
            # Vectorized interpolation - create points array
            points = np.column_stack((q_valid, alpha_valid, np.full_like(q_valid, L)))
            Fq_vals = Fq_interp(points)

            # Calculate intensity I = |F|^2
            I_xy[valid_mask] = Fq_vals**2

        except Exception as e:
            print(f"Warning: Vectorized interpolation failed: {e}")
            print("Falling back to point-by-point interpolation...")

            # Fallback to point-by-point interpolation
            Fq_vals = np.zeros_like(q_valid)
            for i, (q_val, alpha_val) in enumerate(zip(q_valid, alpha_valid)):
                try:
                    # Create point array for single point interpolation
                    point = np.array([[q_val, alpha_val, L]])
                    Fq_vals[i] = Fq_interp(point)[0]
                except Exception as e2:
                    print(f"Point interpolation failed at q={q_val:.2f}, alpha={alpha_val:.2f}: {e2}")
                    # Fallback to simple analytical form
                    qL_cos = q_val * L * np.cos(alpha_val)
                    if np.abs(qL_cos) < 1e-10:
                        Fq_vals[i] = 1.0
                    else:
                        Fq_vals[i] = np.sin(qL_cos / 2) / (qL_cos / 2)

            I_xy[valid_mask] = Fq_vals**2

    # Handle q=0 case
    I_xy[~valid_mask] = 1.0

    # Set very small values to avoid log issues
    I_xy[I_xy <= 0] = 1e-12

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 2D intensity plot (log scale)
    vmin = np.log10(np.maximum(I_xy[I_xy > 0].min(), 1e-6))
    vmax = np.log10(I_xy.max())
    im = ax1.pcolormesh(QX, QY, np.log10(I_xy), cmap="jet", shading="auto", vmin=vmin, vmax=vmax)
    contours = ax1.contour(QX, QY, np.log10(I_xy), levels=10, colors="white", alpha=0.5, linewidths=0.5)
    ax1.set_xlabel("qx", fontsize=12)
    ax1.set_ylabel("qy", fontsize=12)
    ax1.set_title(f"log₁₀(I(qx, qy, qz=0)) - Single Rod (Interpolated)\nL={L}, D={D}, θ={theta:.2f}, phi={phi:.2f}", fontsize=12)
    ax1.set_aspect("equal")
    plt.colorbar(im, ax=ax1, label="log₁₀(Intensity)")

    # Add arrow showing rod orientation projection in xy plane
    arrow_scale = q_max * 0.15
    ax1.arrow(0, 0, rod_x * arrow_scale, rod_y * arrow_scale, head_width=q_max * 0.02, head_length=q_max * 0.03, fc="red", ec="red", linewidth=2)
    ax1.text(rod_x * arrow_scale * 1.3, rod_y * arrow_scale * 1.3, "Rod proj.", fontsize=10, color="red", ha="center", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    # Radial cut plot showing I(q) vs q for different directions
    q_radial = np.linspace(0.001, q_max, 100)  # Stay within interpolation bounds

    # Plot along different directions
    directions = [(1, 0, "along rod projection"), (0, 1, "perpendicular to rod proj."), (1 / np.sqrt(2), 1 / np.sqrt(2), "diagonal")]

    for qx_dir, qy_dir, label in directions:
        I_cut = []
        for q in q_radial:
            qx_val = q * qx_dir
            qy_val = q * qy_dir

            # Calculate angle between this q and rod
            q_dot_rod_val = qx_val * rod_x + qy_val * rod_y
            cos_angle_val = q_dot_rod_val / q  # |rod| = 1, |q| = q
            alpha_val = np.arccos(np.clip(np.abs(cos_angle_val), 0, 1))

            try:
                # Create point array for interpolation
                point = np.array([[q, alpha_val, L]])
                Fq_val = Fq_interp(point)[0]
                I_cut.append(Fq_val**2)
            except Exception:
                # Fallback for out-of-bounds values
                qL_cos = q * L * np.cos(alpha_val)
                if np.abs(qL_cos) < 1e-10:
                    I_cut.append(1.0)
                else:
                    I_cut.append((np.sin(qL_cos / 2) / (qL_cos / 2)) ** 2)

        ax2.plot(q_radial, I_cut, linewidth=2, label=label)

    ax2.set_xlabel("q", fontsize=12)
    ax2.set_ylabel("I(q)", fontsize=12)
    ax2.set_title("Radial Cuts of Intensity (Interpolated)", fontsize=12)
    ax2.set_yscale("log")
    ax2.set_xscale("log")
    ax2.set_ylim(bottom=1e-6)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"Iq_xy_single_rod_interp_L_{L:.0f}_theta_{theta:.1f}_phi_{phi:.1f}.png")
    plt.show()

    return QX, QY, I_xy


def plot_Fq_interpolate():
    q_values = np.linspace(2, 15, 50)
    alpha_values = np.linspace(0, 0.5 * np.pi, 50)
    L_values = np.linspace(4, 6, 2)
    Fq_interp, Fq_mesh = build_FQalpha_interpolator(q_values, alpha_values, L_values)

    # Create meshgrid for plotting
    Q, Alpha, L = np.meshgrid(q_values, alpha_values, L_values, indexing="ij")

    # Create 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Flatten arrays for scatter plot
    q_flat = Q.flatten()
    alpha_flat = Alpha.flatten()
    l_flat = L.flatten()
    fq_flat = Fq_mesh.flatten()

    # Create scatter plot with color mapping
    scatter = ax.scatter(q_flat, alpha_flat, l_flat, c=fq_flat, cmap="viridis", s=20)

    # Add labels and title
    ax.set_xlabel("q")
    ax.set_ylabel("alpha")
    ax.set_zlabel("L")
    ax.set_title("Fq Values in 3D Parameter Space")

    # Add colorbar
    plt.colorbar(scatter, ax=ax, label="Fq")

    plt.tight_layout()
    plt.show()


def calc_PQ(q_values, pd_type, meanL, sigmaL, sigmaD):
    sum_PQ_V2 = np.zeros(len(q_values), dtype=np.float64)
    sum_V2 = 0
    n_particles = max(20 if sigmaL != 0 or sigmaD != 0 else 1, 20 if sigmaD != 0 else 1)
    particle_lengths = np.zeros(n_particles, dtype=np.float64)
    particle_diameters = np.ones(n_particles, dtype=np.float64)  # meanD = 1.0
    # 1. sample particle lengths
    if pd_type == "uniform":
        if sigmaL > 0:
            particle_lengths = meanL * np.random.uniform(1 - sigmaL, 1 + sigmaL, n_particles)
        else:
            particle_lengths = np.full(n_particles, meanL)
        if sigmaD > 0:
            particle_diameters = 1.0 * np.random.uniform(1 - sigmaD, 1 + sigmaD, n_particles)
        else:
            particle_diameters = np.ones(n_particles)
    elif pd_type == "normal":
        if sigmaL > 0:
            particle_lengths = meanL * np.random.normal(1, sigmaL, n_particles)
            particle_lengths = np.clip(particle_lengths, 0, None)
        else:
            particle_lengths = np.full(n_particles, meanL)
        if sigmaD > 0:
            particle_diameters = 1.0 * np.random.normal(1, sigmaD, n_particles)
            particle_diameters = np.clip(particle_diameters, 0, None)
        else:
            particle_diameters = np.ones(n_particles)
    else:
        raise ValueError(f"Unknown pd_type: {pd_type}")

    for n in range(n_particles):
        print(f"Calculating PQ for particle {n+1}/{n_particles} with length {particle_lengths[n]:.2f}, diameter {particle_diameters[n]:.2f}")
        L = particle_lengths[n]
        D = particle_diameters[n]
        # 2. calculate PQ for each particle length and diameter
        PQ_V2, volume = PQ_single_rod_V2(q_values, L, D)
        sum_PQ_V2 += PQ_V2
        sum_V2 += volume**2

    PQ = sum_PQ_V2 / sum_V2
    return PQ


def plot_PQ_demo():
    # q_values = np.logspace(np.log10(0.02), np.log10(20), 200)
    q_values = np.linspace(0.1, 12.5, 80)
    pd_type = "uniform"
    meanL = 4.0
    sigmaL = 0.0
    sigmaD = 0.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # First subplot: varying meanL
    for meanL in [0.0, 2.0, 4.0, 8.0, 16.0]:
        PQ = calc_PQ(q_values, pd_type, meanL, sigmaL, sigmaD)
        ax1.plot(q_values, PQ, label=f"meanL={meanL}")

    ax1.set_xlabel("q")
    ax1.set_ylabel("P(q)")
    ax1.set_title(f"Varying mean L/D (σL={sigmaL}, σD={sigmaD})")
    ax1.set_yscale("log")
    # ax1.set_xscale("log")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Second subplot: varying sigmaL
    for sigmaL_val in [0.0, 0.2, 0.5, 0.8]:
        PQ = calc_PQ(q_values, pd_type, meanL, sigmaL_val, sigmaD)
        ax2.plot(q_values, PQ, label=f"σL={sigmaL_val}")

    ax2.set_xlabel("q")
    ax2.set_ylabel("P(q)")
    ax2.set_title(f"Varying σL (mean L/D={meanL}, σD={sigmaD})")
    ax2.set_yscale("log")
    ax2.set_xscale("log")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("PQ_demo.png")
    plt.show()


def get_Iq_from_file(filename):
    # Read the header (first 6 lines)
    with open(filename, "r") as f:
        lines = f.readlines()

    # Parse pd_type as string from first line
    pd_type = lines[0].split(",")[1].strip()

    # Parse remaining numerical values
    header_data = np.genfromtxt(filename, delimiter=",", max_rows=5, skip_header=1)
    N = header_data[0, 1]
    phi = header_data[1, 1]
    meanL = header_data[2, 1]
    sigmaL = header_data[3, 1]
    sigmaD = header_data[4, 1]

    # Read the data section (starting from row 7 with 3 columns)
    data_section = np.genfromtxt(filename, delimiter=",", skip_header=7)
    q, Iq, dIq = data_section[:, 0], data_section[:, 1], data_section[:, 2]
    return q, Iq, dIq, pd_type, N, phi, meanL, sigmaL, sigmaD


def plot_sample_Iq(filename):
    # Read the file
    q, Iq, dIq, pd_type, N, phi, meanL, sigmaL, sigmaD = get_Iq_from_file(filename)
    print("q", q)
    print("Iq", Iq)
    print("pd_type, N, phi, meanL, sigmaL, sigmaD", pd_type, N, phi, meanL, sigmaL, sigmaD)

    plt.figure(figsize=(5, 5))
    plt.plot(q, Iq, "o", mfc="none", label=f"N={N}, phi={phi}, meanL={meanL}, sigmaL={sigmaL}, sigmaD={sigmaD}", color="blue")
    Pq = calc_PQ(q, pd_type, meanL, sigmaL, sigmaD)
    plt.plot(q, Pq, label="calculated PQ", color="red", linestyle="--")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.savefig(filename.replace(".csv", ".png"))
    plt.title(f"Sample I(q) - {pd_type} distribution")
    plt.xlabel("q")
    plt.ylabel("I(q)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()
    plt.close()


def smooth_Iq(q, Iq, phi):
    log_Iq = np.log10(Iq)
    log_Iq_smooth = log_Iq.copy()
    q_smooth1 = 3.1 + 7.5 * phi  # empirical smoothing threshold
    #q_smooth1 = 5.5  # empirical smoothing threshold
    #q_smooth1 = 5.1 + 7.5 * phi  # empirical smoothing threshold

    nsmooth1 = np.argmin(np.abs(q - q_smooth1))
    '''
    if nsmooth1 > 5:
        spl = make_smoothing_spline(q[:nsmooth1], log_Iq[:nsmooth1])  # cubic spline
        q_smooth = np.linspace(q[0], q[nsmooth1 - 1], nsmooth1)
        log_Iq_smooth[:nsmooth1] = spl(q_smooth)
    '''
    if nsmooth1 > 5:
        # Use Savitzky-Golay filter for low Q
        window_length = min(11, nsmooth1 if nsmooth1 % 2 == 1 else nsmooth1-1)
        polyorder = 1
        log_Iq_smooth[:nsmooth1] = savgol_filter(log_Iq[:nsmooth1], window_length, polyorder)
    Iq_smooth = 10**log_Iq_smooth
    return Iq_smooth


def read_Iq_from_folder(folder, all_system_params, max_num_samples=None):
    all_params = []
    all_Iq = []
    for system_params in all_system_params:
        label = create_file_label(system_params)
        filename = f"{folder}/stats_sample_Iq_{label}.csv"
        # print(f"Reading I(q) from {filename}...")
        if not os.path.exists(filename):
            print(f"File not found: {filename}")
            continue
        q, Iq, dIq, pd_type, N, phi, meanL, sigmaL, sigmaD = get_Iq_from_file(filename)
        Iq_smooth = smooth_Iq(q, Iq, phi)

        # we should shift each I(q) to a given q point, so we don't need to worry about normalizing Iq data when using experimental data
        #q_shift = q[0]  # shift to the first q point
        #Iq_smooth /= Iq_smooth[0] # normalize Iq(q_shift)=1
        # normalize Iq to area = 1?

        # all_Iq.append(Iq)
        all_Iq.append(Iq_smooth)
        all_params.append((phi, meanL, sigmaL, sigmaD))
        if max_num_samples is not None and len(all_Iq) >= max_num_samples:
            print(f"Reached max number of samples: {max_num_samples}")
            break

    all_Iq = np.array(all_Iq)
    all_params = np.array(all_params)
    params_name = ["phi", "meanL", "sigmaL", "sigmaD"]

    return q, all_Iq, all_params, params_name


def calc_Pq_for_params(q, all_system_params):
    all_Pq = []
    cache = {}
    for system_params in all_system_params:
        pd_type = system_params["pd_type"]
        meanL = system_params["meanL"]
        sigmaL = system_params["sigmaL"]
        sigmaD = system_params["sigmaD"]
        key = (pd_type, float(meanL), float(sigmaL), float(sigmaD))
        if key in cache:
            Pq = cache[key]
        else:
            print(f"Calculating P(q) for {key}")
            Pq = calc_PQ(q, pd_type, meanL, sigmaL, sigmaD)
            cache[key] = Pq
        all_Pq.append(Pq)
    return np.array(all_Pq)


def plot_Iq_versus_params(folder, dataset_file):
    q, all_Iq, all_log10Iq, all_params, params_name = read_Iq_dataset(dataset_file)
    # q, all_Iq, all_params = read_Iq_from_folder(folder, all_system_params)
    # plot I(q) versus q curves, colored by different parameters

    # Create figure with 2 rows x 2 columns for I(q) plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Parameter names and indices
    param_names = ["φ (packing fraction)", "mean L/D", "σL (length polydispersity)", "σD (diameter polydispersity)"]
    param_indices = [0, 1, 2, 3]  # phi, meanL, sigmaL, sigmaD

    for ax_idx, (param_name, param_idx) in enumerate(zip(param_names, param_indices)):
        ax = axes.flat[ax_idx]

        # Get unique values and sort them for consistent coloring
        param_values = all_params[:, param_idx]
        unique_values = np.sort(np.unique(param_values))

        # Create colormap normalization
        if len(unique_values) > 1:
            norm = plt.Normalize(vmin=unique_values.min(), vmax=unique_values.max())
        else:
            norm = plt.Normalize(vmin=0, vmax=1)

        # Create colormap
        colormap = plt.cm.get_cmap("rainbow", len(unique_values))

        # Plot each I(q) curve colored by the parameter value
        for i, (Iq, param_vals) in enumerate(zip(all_Iq, all_params)):
            color = colormap(norm(param_vals[param_idx]))

            # Create label with all parameter values
            label = f"φ={param_vals[0]:.2f}, L/D={param_vals[1]:.1f}, σL={param_vals[2]:.2f}, σD={param_vals[3]:.2f}"

            # Plot I(q)
            ax.plot(q, Iq, "-", color=color, alpha=0.7, markersize=4, linewidth=0.5, label=label if (ax_idx == 0 and i == 0) else "")  # only show one legend entry to avoid clutter

        # Subplot formatting
        ax.set_xlabel("q", fontsize=9)
        ax.set_ylabel(r"$I(q)$", fontsize=9)
        ax.set_title(f"I(q) colored by {param_name}", fontsize=9, fontweight="bold")
        ax.set_yscale("log")
        #ax.set_xscale("log")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

        # Add colorbar for each subplot
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(param_name, fontsize=11)

    plt.tight_layout()

    # Save the plot
    output_filename = f"{folder}/Iq_versus_params_comparison.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")

    print(f"Plot saved as: {output_filename}")
    plt.show()


def build_Iq_dataset(folder, label, all_system_params, max_num_samples=None):
    q, all_Iq, all_params, params_name = read_Iq_from_folder(folder, all_system_params, max_num_samples=max_num_samples)
    # save to npz file
    all_log10Iq = np.log10(all_Iq)
    np.savez_compressed(f"{folder}/{label}_log10Iq_dataset.npz", q=q, all_Iq=all_Iq, all_log10Iq=all_log10Iq, all_params=all_params, params_name=params_name)
    print(f"I(q) dataset saved to {folder}/{label}_log10Iq_dataset.npz")

    # in addition, we build train and test datasets
    n_samples = all_Iq.shape[0]
    indices = np.arange(n_samples)
    np.random.seed(42)  # Fixed seed for reproducible shuffling
    np.random.shuffle(indices)
    split_idx = int(n_samples * 0.8)
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    np.savez_compressed(f"{folder}/{label}_log10Iq_dataset_train.npz", q=q, all_Iq=all_Iq[train_idx], all_log10Iq=all_log10Iq[train_idx], all_params=all_params[train_idx], params_name=params_name)
    np.savez_compressed(f"{folder}/{label}_log10Iq_dataset_test.npz", q=q, all_Iq=all_Iq[test_idx], all_log10Iq=all_log10Iq[test_idx], all_params=all_params[test_idx], params_name=params_name)
    print(f"Train and test datasets saved to {folder}/{label}_log10Iq_dataset_train.npz and {label}_log10Iq_dataset_test.npz")

    # Compute stats for train dataset
    mean_log10Iq_train = np.mean(all_log10Iq[train_idx], axis=0)
    std_log10Iq_train = np.std(all_log10Iq[train_idx], axis=0)
    mean_params_train = np.mean(all_params[train_idx], axis=0)
    std_params_train = np.std(all_params[train_idx], axis=0)
    min_params_train = np.min(all_params[train_idx], axis=0)
    max_params_train = np.max(all_params[train_idx], axis=0)

    # Save train stats
    np.savez_compressed(f"{folder}/{label}_log10Iq_dataset_train_stats.npz", mean_log10Iq=mean_log10Iq_train, std_log10Iq=std_log10Iq_train, mean_params=mean_params_train, std_params=std_params_train, min_params=min_params_train, max_params=max_params_train, params_name=params_name)
    print(f"Train dataset stats saved to {folder}/{label}_log10Iq_dataset_train_stats.npz")

    return q, all_Iq, all_log10Iq, all_params, params_name


def read_Iq_dataset(dataset_file):
    data = np.load(dataset_file)
    q = data["q"]
    all_Iq = data["all_Iq"]
    all_log10Iq = data["all_log10Iq"]
    all_params = data["all_params"]
    params_name = data["params_name"]
    print(f"I(q) dataset loaded from {dataset_file}")
    return q, all_Iq, all_log10Iq, all_params, params_name


def svd_analysis(folder, dataset_file):
    q, all_Iq, all_log10Iq, all_params, params_name = read_Iq_dataset(dataset_file)

    F = np.array(all_log10Iq)
    print("samples, shape:", F.shape)
    # Compute the full SVD on the all_Sk data (assumed shape: (n_samples, n_q))
    U, S, Vh = np.linalg.svd(F, full_matrices=False)
    print("Singular values:", S)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot singular values
    ax1.plot(range(len(S)), S, "o--", markerfacecolor="none")
    ax1.set_xlabel("Index")
    ax1.set_ylabel("Singular Value")
    ax1.set_title("Singular Values from SVD")

    # Plot first three singular vectors I(q) vs q
    for i in range(3):
        ax2.plot(q, Vh[i, :], label=f"V{i+1}")
    ax2.set_xlabel("q")
    ax2.set_ylabel("log10(I(q)) from V")
    ax2.set_title("First Three Singular Vectors")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(folder, "singular_values_and_vectors.png"), dpi=300)
    plt.show()

    projection = np.dot(F, np.transpose(Vh))  # Project data onto the right singular vectors
    n_params = len(params_name)
    fig = plt.figure(figsize=(6 * (n_params + 1), 6))

    for i in range(n_params):
        ax = fig.add_subplot(1, n_params, i + 1, projection="3d")
        print("all_params.shape", all_params.shape)
        sc = ax.scatter(projection[:, 0], projection[:, 1], projection[:, 2], c=all_params[:, i], cmap="jet", s=4)
        ax.set_title(f"SVD Projection colored by {params_name[i]}")
        ax.set_xlabel("U1")
        ax.set_ylabel("U2")
        ax.set_zlabel("U3")
        fig.colorbar(sc, ax=ax, label=params_name[i], shrink=0.5)
        nnd = calc_nearest_neighbor_distance(projection, all_params[:, i])
        ax.set_title(f"SVD {params_name[i]} (NND: {nnd:.2f})")

    plt.tight_layout(pad=1.5)
    svd_proj_path = os.path.join(folder, "svd_projection_scatter.png")
    plt.savefig(svd_proj_path, dpi=300)
    plt.show()

    # Add parameter distribution histograms
    fig_hist = plt.figure(figsize=(4 * n_params, 4))
    for i in range(n_params):
        ax = fig_hist.add_subplot(1, n_params, i + 1)
        ax.hist(all_params[:, i], bins=30, alpha=0.7, edgecolor="black")
        ax.set_xlabel(params_name[i])
        ax.set_ylabel("Frequency")
        ax.set_title(f"Distribution of {params_name[i]}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout(pad=0.3)
    plt.savefig(os.path.join(folder, "parameter_distributions.png"), dpi=300)
    plt.show()

    # Save SVD results
    np.savez_compressed(os.path.join(folder, "svd_results.npz"), U=U, S=S, Vh=Vh, q=q, projection=projection, all_params=all_params, params_name=params_name)

    return Vh


def calc_nearest_neighbor_distance(SqV, C):

    # Step 1: Build a k-d tree for efficient neighbor search
    tree = cKDTree(SqV)  # Use only spatial coordinates (x, y, z)
    distances, indices = tree.query(SqV, k=2)  # Find nearest neighbors (k=2)

    # Step 2: Compute color differences
    color_differences = np.abs(C - C[indices[:, 1]])

    # Step 3: Normalize by color range
    color_min = np.min(C)  # Minimum color value
    color_max = np.max(C)  # Maximum color value
    color_range = color_max - color_min  # Range of color values

    # Avoid division by zero if all color values are the same
    if color_range == 0:
        normalized_differences = np.zeros_like(color_differences)
    else:
        normalized_differences = color_differences / color_range * 2

    # Step 4: Compute average normalized color difference
    avg_normalized_difference = np.mean(normalized_differences)

    # print("Average Normalized Color Difference (by range):", avg_normalized_difference)
    return avg_normalized_difference
