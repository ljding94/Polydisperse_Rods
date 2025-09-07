import numpy as np
from scipy.special import j1, roots_legendre
from scipy.interpolate import RegularGridInterpolator


def calculate_IQ(positions, directors, particle_types, type_Vs, q_values, type_Fq_interps, num_directions=100):
    """
    Calculate the 1D isotropic scattering intensity I(q) for a system of spherocylinders,
    normalized by <V^2> (mean square volume over all particles).

    Parameters:
    positions : np.ndarray (N, 3)
        Positions of the N particles (x, y, z).
    directors : np.ndarray (N, 3)
        Director vectors (orientations) of the N particles (ux, uy, uz).
    particle_types : np.ndarray (N,)
        Type indices of the N particles to look up their dimensions.
    type_Vs : np.ndarray (num_types,)
        Volumes V of each particle type.
    q_values : np.ndarray or list
        Array of q magnitudes where to compute I(q).
    type_Fq_interps : list of RegularGridInterpolator
        List of interpolators for FQ(q, alpha) for each particle type.
    num_directions : int, optional
        Number of random directions for orientational averaging (default: 1000).

    Returns:
    np.ndarray
        I(q) for each q in q_values, normalized by <V^2>.
    """
    np.random.seed(42)  # For reproducibility
    N = len(positions)
    q_values = np.asarray(q_values)
    I_qs = np.zeros(len(q_values), dtype=np.float64)

    # Normalize directors
    dir_norms = np.linalg.norm(directors, axis=1)
    if np.any(dir_norms == 0):
        raise ValueError("Director vectors cannot be zero")
    directors_norm = directors / dir_norms[:, np.newaxis]
    particle_Vs = type_Vs[particle_types]
    sum_V2 = np.sum(particle_Vs**2)  # Normalization factor <V^2>
    if sum_V2 == 0:
        raise ValueError("Sum of squared volumes is zero")

    # Generate uniform random directions on the sphere (mimicking C++ uniform cos(theta))
    N_theta = int(np.sqrt(num_directions / 2))  # Approximate square grid for theta, phi
    N_phi = 2 * N_theta
    qtheta_vals = np.arccos(1 - 2 * (np.arange(N_theta) + 0.5) / N_theta)  # Uniform in cos(theta)
    qphi_vals = 2 * np.pi * np.arange(N_phi) / N_phi
    qtheta_grid, qphi_grid = np.meshgrid(qtheta_vals, qphi_vals, indexing="ij")
    qtheta_grid = qtheta_grid.ravel()
    qphi_grid = qphi_grid.ravel()
    num_d = len(qtheta_grid)

    # Compute q_hat vectors
    sin_thetas = np.sin(qtheta_grid)
    q_hats = np.zeros((num_d, 3))
    q_hats[:, 0] = sin_thetas * np.cos(qphi_grid)
    q_hats[:, 1] = sin_thetas * np.sin(qphi_grid)
    q_hats[:, 2] = np.cos(qtheta_grid)

    for iq, q in enumerate(q_values):
        cos_alphas = np.abs(np.dot(directors_norm, q_hats.T))
        alphas = np.arccos(cos_alphas)  # (N, num_d)

        # Prepare interpolation points (N * num_d, 4)
        # Note: We prepare points per type below

        F_js = np.zeros((N, num_d), dtype=np.complex128)  # F per particle per direction
        for itype in range(len(type_Fq_interps)):
            mask = particle_types == itype
            num_particles_itype = np.sum(mask)
            if num_particles_itype == 0:
                continue
            Fq_interp = type_Fq_interps[itype]
            # Select alphas for this type
            alphas_itype = alphas[mask, :]  # (num_particles_itype, num_d)
            qs_all_itype = np.full(num_particles_itype * num_d, q)
            alphas_all_itype = alphas_itype.ravel()
            all_points_itype = np.column_stack((qs_all_itype, alphas_all_itype))
            F_itype = Fq_interp(all_points_itype)
            F_reshaped = F_itype.reshape(num_particles_itype, num_d)
            F_js[mask, :] = F_reshaped

        # Compute q_vecs (num_d, 3)
        q_vecs = q * q_hats

        # Compute phase factors: exp(i q · r_i)
        dots = np.dot(positions, q_vecs.T)  # (N, num_d)
        phases = np.exp(1j * dots)  # (N, num_d)

        # Total amplitude: sum_i F_i e^(i q · r_i)
        # V already in the F_js
        total_amps = np.sum(F_js * phases, axis=0)  # (num_d,)

        # I(q, theta, phi) = |total_amps|^2 / sum_V2
        I_q_theta_phi = np.abs(total_amps) ** 2 / sum_V2

        # Average over directions
        I_qs[iq] = np.mean(I_q_theta_phi)

    return I_qs


def calculate_PQ(q_values, particle_types, type_Vs, type_Fq_interps, num_mu=50):
    """
    Calculate the polydisperse orientationally averaged form factor P(q) = <F(q)^2> / <V^2>,
    where < > denotes average over particles (weighted by V^2) and orientations.

    Parameters:
    q_values : np.ndarray or list
        Array of q magnitudes where to compute P(q).
    particle_types : np.ndarray (N,)
        Type indices of the N particles.
    type_Vs : np.ndarray (num_types,)
        Volumes V of each particle type.
    type_Fq_interps : list of RegularGridInterpolator
        List of interpolators for FQ(q, alpha) for each particle type.
    num_mu : int, optional
        Number of points for uniform mu = cos(alpha) integration from 0 to 1 (default: 50).

    Returns:
    np.ndarray
        P(q) for each q in q_values.
    """
    N = len(particle_types)
    q_values = np.asarray(q_values)
    P_qs = np.zeros(len(q_values), dtype=np.float64)

    particle_Vs = type_Vs[particle_types]
    sum_V2 = np.sum(particle_Vs**2)  # Normalization factor <V^2>
    if sum_V2 == 0:
        raise ValueError("Sum of squared volumes is zero")

    # Uniform mu grid for integration (mu = cos(alpha), alpha in [0, pi/2] due to symmetry)
    mu = np.linspace(0.0, 1.0, num_mu)
    alpha_mu = np.arccos(mu)  # alpha decreases from pi/2 to 0 as mu increases

    for iq, q in enumerate(q_values):
        P_i = np.zeros(N)
        for itype in range(len(type_Fq_interps)):
            mask = particle_types == itype
            if np.sum(mask) == 0:
                continue
            Fq_interp = type_Fq_interps[itype]
            # Prepare interpolation points for this type
            alphas_all = alpha_mu
            qs_all = np.full(num_mu, q)
            all_points = np.column_stack((qs_all, alphas_all))
            F_all = Fq_interp(all_points)
            # Compute average |F / V|^2 over orientations
            P_type = np.mean((F_all / type_Vs[itype]) ** 2)
            P_i[mask] = P_type

        # Polydisperse average: ∑ V_i^2 P_i / ∑ V_i^2
        P_qs[iq] = np.sum(particle_Vs**2 * P_i) / sum_V2

    return P_qs


# Precompute Gauss-Legendre nodes and weights
n = 32
x, weights = roots_legendre(n)


def FQalpha(q, alpha, L, D):
    """
    Optimized form factor amplitude FQalpha(q, alpha) for a spherocylinder using Gauss-Legendre.
    """
    r = D / 2.0

    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)

    w = 0.5 * q * L * cos_alpha
    v = q * r * sin_alpha

    # Cylinder term
    if abs(w) < 1e-10:
        sinc_w = 1.0
    else:
        sinc_w = np.sin(w) / w

    if abs(v) < 1e-10:
        j1_over_v = 0.5
    else:
        j1_over_v = j1(v) / v

    cyl_term = np.pi * r**2 * L * sinc_w * 2.0 * j1_over_v

    # Cap term with Gauss-Legendre
    t = (x + 1.0) / 2.0
    sqrt_1mt2 = np.sqrt(1 - t**2)
    u = q * r * sin_alpha * sqrt_1mt2
    z = q * cos_alpha * (r * t + 0.5 * L)

    with np.errstate(invalid="ignore"):
        j1_over_u = np.where(np.abs(u) < 1e-10, 0.5, j1(u) / u)

    integrand = (1 - t**2) * j1_over_u * np.cos(z)
    integral = 0.5 * np.sum(weights * integrand)
    cap_term = 4.0 * np.pi * r**3 * integral

    return cyl_term + cap_term


def FQalpha_grid(q_values, alpha_values, L, D):
    """
    Compute the form factor amplitude FQalpha(q, alpha) on a grid of q, alpha, for particle of dimension (L,D)
    """

    Fq_mesh = np.zeros((len(q_values), len(alpha_values)), dtype=np.float64)
    for i, q in enumerate(q_values):
        for j, alpha in enumerate(alpha_values):
            Fq_mesh[i, j] = FQalpha(q, alpha, L, D)
    return Fq_mesh


def build_FQalpha_interpolator(q_values, alpha_values, type_Ls, type_Ds):
    """
    Build and return the RegularGridInterpolator object for FQ.
    for each particle type, we build an interpolator over (q, alpha) only.

    Parameters:
    q_values: np.array of q points for the grid
    alpha_values: np.array of alpha points (radians, typically 0 to pi/2)

    Returns:
    RegularGridInterpolator object that interpolates over (q, alpha, L)
    """
    type_Fq_interps = []
    type_Fq_meshs = []
    for i in range(len(type_Ls)):
        L = type_Ls[i]
        D = type_Ds[i]
        print(f"Building FQ interpolator for type {i}, L={L}, D={D}")

        Fq_mesh = FQalpha_grid(q_values, alpha_values, L, D)

        # Create and return the 3D interpolator
        Fq_interp = RegularGridInterpolator(
            (q_values, alpha_values), Fq_mesh, bounds_error=False, fill_value=0.0, method="linear"  # Or np.nan, but 0 for extrapolation  # 'linear' is default, 'nearest' for speed if needed
        )
        type_Fq_interps.append(Fq_interp)
        type_Fq_meshs.append(Fq_mesh)

    return type_Fq_interps, type_Fq_meshs


def PQ_single_rod_V2(q_values, L, D=1.0):

    PQ_values = np.zeros(len(q_values), dtype=np.float64)

    alpha_values = np.linspace(0, np.pi / 2, 100)  # Uniformly sample alpha from 0 to pi/2
    for i, q in enumerate(q_values):
        # Average over all orientations (alpha angles)
        Pq_sum = 0.0
        weight_sum = 0.0
        for alpha_val in alpha_values:
            Fq_val = FQalpha(q, alpha_val, D, L)
            # Weight by sin(alpha) for proper spherical averaging
            weight = np.sin(alpha_val)
            Pq_sum += (Fq_val**2) * weight
            weight_sum += weight

        if weight_sum > 0:
            PQ_values[i] = Pq_sum / weight_sum
        else:
            PQ_values[i] = 0.0
    volume = np.pi * (D / 2.0) ** 2 * L + (4 / 3) * np.pi * (D / 2.0) ** 3  # Volume of the spherocylinder

    return PQ_values, volume
