import hoomd
import hoomd.hpmc
import numpy as np
import math
import gsd.hoomd
from rods_tool import *
import os


# Function to generate aspect ratios (L/d) based on polydispersity type
def generate_aspect_ratios(N, mean_ld, sigma, pd_type="uniform"):
    if pd_type == "uniform":
        # Uniform: mean_ld * U(1-sigma, 1+sigma)
        return mean_ld * np.random.uniform(1 - sigma, 1 + sigma, N)
    elif pd_type == "normal":
        # normal: mean_ld * Normal(1, sigma)
        particle_lengths = mean_ld * np.random.normal(1, sigma, N)
        particle_lengths = np.clip(particle_lengths, 0, None)  # Ensure no negative lengths
        return particle_lengths
    elif pd_type == "lognormal":
        # lognormal:  * LogNormal(log(mean_ld), s) where mu and s derived from mean and sigma
        return np.random.lognormal(mean=np.log(mean_ld), sigma=sigma, size=N)
    else:
        raise ValueError("Unsupported pd_type. Use 'uniform' or 'normal'.")


# Function to discretize aspects into bins (for types)
def discretize_aspects(aspects, num_types):
    # Sort and bin into num_types
    sorted_aspects = np.sort(aspects)
    type_ids = np.floor(np.linspace(0, num_types - 1, len(aspects))).astype(int)
    # Average aspect per bin
    type_aspects = np.array([np.mean(sorted_aspects[type_ids == t]) for t in range(num_types)])
    # Assign type to each particle
    particle_types = type_ids
    return particle_types, type_aspects


def generate_and_save_shape_info(subfolder, pd_type, N, mean_ld, sigma):
    D = 1.0
    r = D / 2.0  # Radius for sweep
    aspects = generate_aspect_ratios(N, mean_ld, sigma, pd_type)
    lengths = aspects * D  # Cylinder lengths (excluding caps)
    # Discretize into types
    if sigma == 0:
        num_types = 1
    else:
        num_types = min(100, N)  # Reduced to 100 for better performance
    particle_types, type_lengths = discretize_aspects(lengths, num_types)

    # Compute volumes per type: π r² L + (4/3) π r³ = (π/4) L + (π/6) with d=1
    type_volumes = (math.pi / 4) * type_lengths + (math.pi / 6)
    total_volume = sum(type_volumes[particle_types[i]] for i in range(N))

    # Save particle_types, type_lengths, and total_vol to an npz file
    np.savez(f"{subfolder}/particle_data.npz", num_types=num_types, particle_types=particle_types, type_lengths=type_lengths, total_volume=total_volume)

    return num_types, particle_types, type_lengths, total_volume


def read_shape_info(subfolder):
    data = np.load(f"{subfolder}/particle_data.npz")
    num_types = data["num_types"]
    particle_types = data["particle_types"]
    type_lengths = data["type_lengths"]
    total_volume = data["total_volume"]
    return num_types, particle_types, type_lengths, total_volume


def run_initialization(save_dump_detail, system_params, randomization_steps=5000, max_compression_steps=50000, num_compression_stage=10):
    folder = system_params["folder"]
    subfolder = f"{folder}/{create_file_label(system_params)}"
    pd_type, N, phi, mean_ld, sigma = system_params["pd_type"], system_params["N"], system_params["phi"], system_params["mean_ld"], system_params["sigma"]

    # Diameter (fixed)
    D = 1.0
    r = D / 2.0  # Radius for sweep

    num_types, particle_types, type_lengths, total_volume = generate_and_save_shape_info(subfolder, pd_type, N, mean_ld, sigma)
    particle_lengths = type_lengths[particle_types]  # Get lengths based on types
    print("type_lengths", type_lengths)
    print("particle_lengths", particle_lengths)

    # Initial and target box
    initial_phi = 0.01
    initial_box_vol = total_volume / initial_phi
    initial_box_L = initial_box_vol ** (1 / 3)
    target_box_vol = total_volume / phi

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
        L = type_lengths[i]
        # Vertices for spherocylinder: two points separated by L along z-axis, to be consistent with ovito
        vertices = [[0, 0, -L / 2], [0, 0, L / 2]]
        mc.shape[type_name] = dict(vertices=vertices, sweep_radius=r, ignore_statistics=False)

    sim.operations.integrator = mc

    # Run 0 steps to initialize
    sim.run(0)
    print(f"Initial overlaps: {mc.overlaps}")

    dumper = DumpTXT(subfolder, type_lengths, "compress")  # Use aspects for L/D
    if save_dump_detail:
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
        target_box_vol = total_volume / target_phi
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

    dumper.act(sim.timestep)  # dump final config

    gsd_filename = f"{subfolder}/compressed.gsd"
    fn = os.path.join(os.getcwd(), gsd_filename)
    if os.path.exists(fn):
        print(f"Removing existing file: {fn}")
        os.remove(fn)
    hoomd.write.GSD.write(state=sim.state, mode="xb", filename=gsd_filename)
