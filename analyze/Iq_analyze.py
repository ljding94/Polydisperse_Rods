import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path so we can import from there
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "simulation"))
from IQ import build_FQalpha_interpolator, FQalpha, PQ_single_rod_V2


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
    Fq_interp, Fq_mesh = build_FQ_interpolator(q_values, alpha_values, L_values)

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


def calc_PQ(q_values, pd_type, mean_ld, sigma):
    sum_PQ_V2 = np.zeros(len(q_values), dtype=np.float64)
    sum_V2 = 0
    n_particles = 20
    particle_lengths = np.zeros(n_particles, dtype=np.float64)
    # 1. sample particle lengths
    if pd_type == "uniform":
        particle_lengths = np.random.uniform(mean_ld - sigma, mean_ld + sigma, n_particles)
    elif pd_type == "normal":
        particle_lengths = np.random.normal(mean_ld, sigma, n_particles)
        particle_lengths = np.clip(particle_lengths, 0, None)
    else:
        raise ValueError(f"Unknown pd_type: {pd_type}")

    for n in range(n_particles):
        print(f"Calculating PQ for particle {n+1}/{n_particles} with length {particle_lengths[n]:.2f}")
        L = particle_lengths[n]
        # 2. calculate PQ for each particle length
        PQ_V2, volume = PQ_single_rod_V2(q_values, L)
        sum_PQ_V2 += PQ_V2
        sum_V2 += volume**2

    PQ = sum_PQ_V2 / sum_V2
    return PQ


def get_Iq_from_file(filename):
    # Read the header (first 5 lines)
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Parse pd_type as string from first line
    pd_type = lines[0].split(',')[1].strip()

    # Parse remaining numerical values
    header_data = np.genfromtxt(filename, delimiter=",", max_rows=4, skip_header=1)
    N = header_data[0, 1]
    phi = header_data[1, 1]
    mean_ld = header_data[2, 1]
    sigma = header_data[3, 1]

    # Read the data section (starting from row 6 with 3 columns)
    data_section = np.genfromtxt(filename, delimiter=",", skip_header=6)
    q, Iq, dIq = data_section[:, 0], data_section[:, 1], data_section[:, 2]
    return q, Iq, dIq, pd_type, N, phi, mean_ld, sigma


def plot_sample_Iq(filename):
    # Read the file
    q, Iq, dIq, pd_type, N, phi, mean_ld, sigma = get_Iq_from_file(filename)
    print("q",q)
    print("Iq", Iq)
    print("pd_type, N, phi, mean_ld, sigma",pd_type, N, phi, mean_ld, sigma)

    plt.figure(figsize=(5, 5))
    plt.plot(q, Iq, "o", mfc="none", label=f"N={N}, phi={phi}, mean_ld={mean_ld}, sigma={sigma}", color="blue")
    Pq = calc_PQ(q, pd_type, mean_ld, sigma)
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
