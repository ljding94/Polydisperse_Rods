import hoomd
import hoomd.hpmc
import numpy as np
import math
import gsd.hoomd
from rods_tool import *
import os


# Function to generate lengths (L) and diameters (D) based on polydispersity type and parameters
def generate_rods_dimensions(N, meanL, sigmaL, sigmaD, pd_type="uniform"):
    if pd_type == "uniform":
        # Uniform for L: meanL * U(1-sigmaL, 1+sigmaL)
        Ls = meanL * np.random.uniform(1 - sigmaL, 1 + sigmaL, N)
        # Uniform for D: meanD=1 * U(1-sigmaD, 1+sigmaD)
        Ds = 1.0 * np.random.uniform(1 - sigmaD, 1 + sigmaD, N)

    elif pd_type == "normal":
        # Normal for L: meanL * Normal(1, sigmaL)
        Ls = meanL * np.random.normal(1, sigmaL, N)
        Ls = np.clip(Ls, 0, None)  # Ensure no negative lengths
        # Normal for D: meanD=1 * Normal(1, sigmaD)
        Ds = 1.0 * np.random.normal(1, sigmaD, N)
        Ds = np.clip(Ds, 0.1, None)  # Ensure no negative diameters
    elif pd_type == "lognormal":
        # Lognormal for L: LogNormal(log(meanL), sigmaL)
        Ls = np.random.lognormal(mean=np.log(meanL), sigma=sigmaL, size=N)
        # Lognormal for D: LogNormal(log(1), sigmaD)
        Ds = np.random.lognormal(mean=np.log(1.0), sigma=sigmaD, size=N)
    else:
        raise ValueError("Unsupported pd_type. Use 'uniform', 'normal', or 'lognormal'.")

    return Ls, Ds


# Function to discretize lengths (Ls) and diameters (Ds) into bins for types
def discretize_aspects(Ls, Ds, ntype):
    # in the (L,D) plane, uniformly find ntype points, assign each (L_i,D_i) to the nearest point, so we will have total ntype combination of (L,D)

    min_L = np.min(Ls)
    max_L = np.max(Ls)
    min_D = np.min(Ds)
    max_D = np.max(Ds)

    # Determine grid size for approximately ntype points
    ntypeL = int(np.ceil(np.sqrt(ntype)))
    ntypeD = int(np.ceil(ntype / ntypeL))
    num_types = ntypeL * ntypeD

    # Handle edge cases
    if ntype <= 0:
        raise ValueError("ntype must be positive")
    if len(Ls) == 0 or len(Ds) == 0:
        raise ValueError("Ls and Ds must not be empty")

    # If all Ls are the same, set centroids_L to that value
    if min_L == max_L:
        centroids_L = np.full(ntypeL, min_L)
    else:
        centroids_L = np.linspace(min_L, max_L, ntypeL)

    # If all Ds are the same, set centroids_D to that value
    if min_D == max_D:
        centroids_D = np.full(ntypeD, min_D)
    else:
        centroids_D = np.linspace(min_D, max_D, ntypeD)

    centroids = np.array([[cl, cd] for cl in centroids_L for cd in centroids_D])
    X = np.column_stack((Ls, Ds))
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    particle_types = np.argmin(distances, axis=1)

    type_Ls = centroids[:, 0]
    type_Ds = centroids[:, 1]

    print("num_types", num_types)
    print("particle_types", particle_types)
    print("type_Ls", type_Ls)
    print("type_Ds", type_Ds)

    return num_types, particle_types, type_Ls, type_Ds


def generate_and_save_shape_info(subfolder, pd_type, N, meanL, sigmaL, sigmaD):
    Ls, Ds = generate_rods_dimensions(N, meanL, sigmaL, sigmaD, pd_type)
    # Discretize into types

    ntype_approx = min(100, N)

    num_types, particle_types, type_Ls, type_Ds = discretize_aspects(Ls, Ds, ntype_approx)

    # Compute volumes per type: π r² L + (4/3) π r³ = (π/4) D² L + (π/6) D³ with r = D/2
    type_Vs = (math.pi / 4) * type_Ds**2 * type_Ls + (math.pi / 6) * type_Ds**3
    total_V = sum(type_Vs[particle_types[i]] for i in range(N))
    total_V2 = sum(type_Vs[particle_types[i]] ** 2 for i in range(N))

    # Save particle_types, type_lengths, and total_vol to an npz file
    np.savez(f"{subfolder}/particle_data.npz", num_types=num_types, particle_types=particle_types, type_Ls=type_Ls, type_Ds=type_Ds, type_Vs=type_Vs, total_V=total_V, total_V2=total_V2)

    return num_types, particle_types, type_Ls, type_Ds, type_Vs, total_V, total_V2


def read_shape_info(subfolder):
    data = np.load(f"{subfolder}/particle_data.npz")
    num_types = data["num_types"]
    particle_types = data["particle_types"]
    type_Ls = data["type_Ls"]
    type_Ds = data["type_Ds"]
    type_Vs = data["type_Vs"]
    total_V = data["total_V"]
    total_V2 = data["total_V2"]
    return num_types, particle_types, type_Ls, type_Ds, type_Vs, total_V, total_V2


def run_initialization(save_dump_detail, system_params, randomization_steps=5000, max_compression_steps=50000, num_compression_stage=10):
    folder = system_params["folder"]
    subfolder = f"{folder}/{create_file_label(system_params)}"
    pd_type, N, phi, meanL, sigmaL, sigmaD = system_params["pd_type"], system_params["N"], system_params["phi"], system_params["meanL"], system_params["sigmaL"], system_params["sigmaD"]

    num_types, particle_types, type_Ls, type_Ds, type_Vs, total_V, total_V2 = generate_and_save_shape_info(subfolder, pd_type, N, meanL, sigmaL, sigmaD)
    particle_Ls = type_Ls[particle_types]
    particle_Ds = type_Ds[particle_types]
    print("particle_types", particle_types)
    print("type_Ls", type_Ls)
    print("type_Ds", type_Ds)
    print("unique Type_Ls", np.unique(type_Ls))
    print("unique Type_Ds", np.unique(type_Ds))
    print("len(type_Ls)", len(type_Ls))
    print("len(type_Ds)", len(type_Ds))
    print("particle_Ls", particle_Ls)
    print("particle_Ds", particle_Ds)

    # Initial and target box
    initial_phi = 0.01
    initial_box_vol = total_V / initial_phi
    initial_box_L = initial_box_vol ** (1 / 3)
    target_box_vol = total_V / phi

    # Initialize HOOMD
    device = hoomd.device.CPU()  # Or hoomd.device.GPU() if available
    sim = hoomd.Simulation(device=device)

    # Create snapshot
    snapshot = hoomd.Snapshot()
    snapshot.configuration.box = [initial_box_L, initial_box_L, initial_box_L, 0, 0, 0]
    snapshot.particles.N = N
    snapshot.particles.types = [f"type_{i}" for i in range(num_types)]
    snapshot.particles.typeid[:] = particle_types

    # Initial positions: Simple cubic lattice to avoid overlaps (scale spacing)
    grid_size = math.ceil(N ** (1 / 3))
    spacing = initial_box_L / grid_size
    positions = []
    for i in range(N):
        x = (i % grid_size) * spacing - initial_box_L / 2
        y = ((i // grid_size) % grid_size) * spacing - initial_box_L / 2
        z = (i // (grid_size**2)) * spacing - initial_box_L / 2
        positions.append([x, y, z])

    # Shuffle positions
    np.random.shuffle(positions)
    snapshot.particles.position[:] = positions

    # Random orientations (quaternions)
    # orientations = np.random.uniform(-1, 1, (N, 4))
    # orientations /= np.linalg.norm(orientations, axis=1)[:, np.newaxis]  # Normalize
    # snapshot.particles.orientation[:] = orientations

    # Align all orientations along the z-axis
    orientations = [[1, 0, 0, 0]] * N  # Quaternion [w, i, j, k] for no rotation (aligned along z-axis)
    snapshot.particles.orientation[:] = orientations

    # Create the simulation from snapshot
    sim.create_state_from_snapshot(snapshot)

    # Set up HPMC integrator for spherocylinders (using ConvexSpheropolyhedron)
    mc = hoomd.hpmc.integrate.ConvexSpheropolyhedron(default_d=1.0, default_a=0.1)

    for i in range(num_types):
        type_name = f"type_{i}"
        L = type_Ls[i]
        D = type_Ds[i]
        # Vertices for spherocylinder: two points separated by L along z-axis, to be consistent with ovito
        vertices = [[0, 0, -L / 2], [0, 0, L / 2]]
        mc.shape[type_name] = dict(vertices=vertices, sweep_radius=0.5 * D, ignore_statistics=False)

    sim.operations.integrator = mc

    # Run 0 steps to initialize
    sim.run(0)
    print(f"Initial overlaps: {mc.overlaps}")

    dumper = DumpTXT(subfolder, type_Ls, type_Ds, type_Vs, "compress")
    if save_dump_detail:
        print("Saving detailed dumps...")
        custom_writer = hoomd.write.CustomWriter(action=dumper, trigger=hoomd.trigger.Periodic(5000))
        sim.operations.writers.append(custom_writer)

    lc_action = MeasureLiquidCrystalOrder(subfolder, system_params, "compress", num_d=100)
    lc_writer = hoomd.write.CustomWriter(action=lc_action, trigger=hoomd.trigger.Periodic(randomization_steps))
    sim.operations.writers.append(lc_writer)

    """
    nematic_action = MeasureNematic(folder, "compress_nematic_order.csv")
    nematic_writer = hoomd.write.CustomWriter(action=nematic_action, trigger=hoomd.trigger.Periodic(randomization_steps))
    sim.operations.writers.append(nematic_writer)

    smectic_action = MeasureSmectic(folder, mean_ld, "compress_smectic_order.csv", num_d=100)
    smectic_writer = hoomd.write.CustomWriter(action=smectic_action, trigger=hoomd.trigger.Periodic(randomization_steps))
    sim.operations.writers.append(smectic_writer)
    """

    # randomization
    sim.run(randomization_steps)
    print(f"Randomization complete. overlaps: {mc.overlaps}")

    # add tuner to adjust move sizes
    tuner = hoomd.hpmc.tune.MoveSize.scale_solver(trigger=hoomd.trigger.Periodic(10), moves=["a", "d"], target=0.3, max_translation_move=1.0, max_rotation_move=0.1)
    sim.operations.tuners.append(tuner)

    # Compression for better initialization
    print("Compressing box...")

    # one step compression results in jamming, use staged compression instead
    """
    initial_box = sim.state.box
    final_box = hoomd.Box.from_box(initial_box)
    final_box.volume = target_box_vol
    compresser = hoomd.hpmc.update.QuickCompress(
        trigger=hoomd.trigger.Periodic(10),
        target_box=final_box,
    )
    sim.operations.updaters.append(compresser)

    while not compresser.complete and sim.timestep < randomization_steps+max_compression_steps:
        sim.run(1000)

    if not compresser.complete:
        print("Compression failed to complete")
    """

    # After sim.run(randomization_steps)
    print("Starting staged compression...")
    max_steps_per_stage = int(max_compression_steps / num_compression_stage)
    print("max_steps_per_stage:", max_steps_per_stage)
    # phi_steps = np.linspace(initial_phi, phi, num_compression_stage)  # 10 intermediate densities

    x = np.linspace(0, 1, num_compression_stage + 1)  # 10 steps; adjust number as needed
    phi_progress = 2 * x - x**2  # Quadratic progression: faster increase early, slower later
    phi_steps = initial_phi + (phi - initial_phi) * phi_progress
    phi_steps[-1] = phi  # Ensure exact final target
    for i, target_phi in enumerate(phi_steps[1:]):
        print(f"state {i}/{num_compression_stage} Compressing to phi={target_phi} ... final target {phi}")
        target_box_vol = total_V / target_phi
        final_box = hoomd.Box.from_box(sim.state.box)
        final_box.volume = target_box_vol
        compresser = hoomd.hpmc.update.QuickCompress(
            trigger=hoomd.trigger.Periodic(10),
            target_box=final_box,
        )
        sim.operations.updaters.append(compresser)

        # Run longer at each stage
        step_counter = 0
        while (not compresser.complete) and (step_counter < max_steps_per_stage):
            sim.run(randomization_steps)
            step_counter += randomization_steps

        print(f"current overlaps: {mc.overlaps}, timestep: {sim.timestep}")
        if not compresser.complete:
            print(f"Compression to phi={target_phi} incomplete")
        else:
            print(f"Reached phi={target_phi}, running equilibration...")
            sim.run(randomization_steps)  # Equilibrate at this density
        sim.operations.updaters.remove(compresser)

    # few more steps just for checking
    # post randomization to stabilize the system
    sim.run(5 * randomization_steps)

    print(f"Compressing complete. overlaps: {mc.overlaps}, timestep: {sim.timestep}")
    print(f"translation move size: {mc.translate_moves}, rotation move size: {mc.rotate_moves}")
    print("translate acceptance rate:", mc.translate_moves[0] / sum(mc.translate_moves))
    print("rotate acceptance rate:", mc.rotate_moves[0] / sum(mc.rotate_moves))
    print("mote size:", mc.a["type_0"], mc.d["type_0"])

    if save_dump_detail:
        custom_writer = hoomd.write.CustomWriter(action=dumper, trigger=hoomd.trigger.Periodic(5000))
        sim.operations.writers.append(custom_writer)

    dumper.act(sim.timestep)  # dump final config

    gsd_filename = f"{subfolder}/compressed.gsd"
    fn = os.path.join(os.getcwd(), gsd_filename)
    if os.path.exists(fn):
        print(f"Removing existing file: {fn}")
        os.remove(fn)

    hoomd.write.GSD.write(state=sim.state, mode="xb", filename=gsd_filename)
