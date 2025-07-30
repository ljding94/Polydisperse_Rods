import hoomd
import hoomd.hpmc
import numpy as np
import gsd.hoomd
from rods_tool import *
import os
from rods_init import *
from rods_tool import *


def run_sampling(save_dump_detail, system_params, measurement_steps=10000, N_measurement=100):
    folder = system_params["folder"]
    label = create_file_label(system_params)
    subfolder = f"{folder}/{label}"

    # initialize HOOMD
    device = hoomd.device.CPU()  # Or hoomd.device.GPU
    sim = hoomd.Simulation(device=device)

    # load particle shape info
    num_types, particle_types, type_lengths, total_volume = read_shape_info(subfolder)
    particle_lengths = type_lengths[particle_types]  # Get lengths based on types

    D = 1.0
    r = D / 2.0  # Radius for sweep
    mc = hoomd.hpmc.integrate.ConvexSpheropolyhedron(default_d=0.2, default_a=0.1)
    for i in range(num_types):
        type_name = f"type_{i}"
        L = type_lengths[i]
        # Vertices for spherocylinder: two points separated by L along z-axis, to be consistent with ovito
        vertices = [[0, 0, -L / 2], [0, 0, L / 2]]
        mc.shape[type_name] = dict(vertices=vertices, sweep_radius=r, ignore_statistics=False)  # tell mc integrator the size of each type

    sim.operations.integrator = mc
    sim.create_state_from_gsd(f"{subfolder}/compressed.gsd")
    # the gsd file has: posistion, partile_type, orientation

    # Run 0 steps to initialize
    sim.run(0)
    print(f"Initial overlaps: {mc.overlaps}")

    dumper = DumpTXT(subfolder, type_lengths, "sampling")  # Use aspects for L/D
    if save_dump_detail:
        custom_writer = hoomd.write.CustomWriter(action=dumper, trigger=hoomd.trigger.Periodic(5000))
        sim.operations.writers.append(custom_writer)

    lc_action = MeasureLiquidCrystalOrder(subfolder, system_params, "sample", num_d=100)
    lc_writer = hoomd.write.CustomWriter(action=lc_action, trigger=hoomd.trigger.Periodic(measurement_steps))
    sim.operations.writers.append(lc_writer)
    q_values = np.linspace(1.0, 20, 100)
    subfolder = f"{folder}/{label}"
    Iq_action = MeasureScattering(folder, subfolder, q_values, particle_lengths, system_params, "sample")
    Iq_writer = hoomd.write.CustomWriter(action=Iq_action, trigger=hoomd.trigger.Periodic(measurement_steps))
    sim.operations.writers.append(Iq_writer)

    for i in range(N_measurement):
        print("Measurement step:", i + 1)
        sim.run(measurement_steps)

    lc_action.act_end()  # Finalize liquid crystal order measurement
    Iq_action.act_end()  # Finalize scattering measurement
    dumper.act(sim.timestep)  # dump final config
