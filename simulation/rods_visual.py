import fresnel
import os
import numpy as np
import hoomd
import rowan
from rods_init import *
import gsd.hoomd
from matplotlib import cm
import math
from PIL import Image
import matplotlib.pyplot as plt

def read_snapshot(gsd_file, frame=-1):
    traj = gsd.hoomd.open(name=gsd_file, mode="r")
    print(len(traj), "frames in trajectory")
    snap = traj[frame]
    return snap


def read_configuration(folder, gsd_file, frame=-1):
    snap = read_snapshot(f"{folder}/{gsd_file}", frame)

    print("Snapshot properties:")
    print(f"Number of particles: {snap.particles.N}")
    print(f"Box dimensions: {snap.configuration.box}")
    print(f"Particle positions shape: {snap.particles.position.shape}")
    print(f"Particle types: {snap.particles.types}")
    print(f"Particle type IDs: {snap.particles.typeid}")
    print("\nFirst few particle positions:")
    print(snap.particles.position[:5])
    print("\nFirst few particle orientations:")
    print(snap.particles.orientation[:5])

    num_types, particle_types, type_lengths, total_volume = read_shape_info(folder)
    positions = snap.particles.position
    orientations = snap.particles.orientation
    directions = rowan.rotate(orientations, np.array([0.0, 0.0, 1.0]))  # Local +x axis
    particle_lengths = type_lengths[particle_types]  # Get lengths based on types

    return positions, directions, particle_lengths, snap


def render(folder, gsd_file, frame=-1):
    positions, directions, particle_lengths, snap = read_configuration(folder, gsd_file, frame)

    end1_positions = positions - 0.5 * particle_lengths[:, np.newaxis] * directions
    end2_positions = positions + 0.5 * particle_lengths[:, np.newaxis] * directions
    # Create a fresnel scene
    scene = fresnel.Scene()

    # Create cylinder geometry for all spherocylinders
    geometry = fresnel.geometry.Cylinder(scene, N=len(end1_positions))

    # Set the end points for each cylinder
    geometry.points[:] = [[end1_positions[i], end2_positions[i]] for i in range(len(end1_positions))]

    # Set the radius for all cylinders
    geometry.radius[:] = 0.5



    # Normalize particle lengths to range [0, 1] for coloring
    min_length = np.min(particle_lengths)
    max_length = np.max(particle_lengths)
    normalized_lengths = (particle_lengths - min_length) / (max_length - min_length)

    # Use the bwr colormap from matplotlib
    colormap = cm.get_cmap('bwr')
    colors = colormap(normalized_lengths)

    # Map normalized lengths to colors - cylinders need color for both ends
    cylinder_colors = []
    for color in colors:
        linear_color = fresnel.color.linear(color[:3])
        cylinder_colors.append([linear_color, linear_color])  # Same color for both ends
    geometry.color[:] = cylinder_colors

    # Set material properties
    geometry.material = fresnel.material.Material(color=fresnel.color.linear([0.25, 0.5, 0.9]), roughness=0.8)
    geometry.outline_width = 0.01

    fresnel.geometry.Box(scene, snap.configuration.box, box_radius=0.02)

    # Set up the camera
    scene.camera = fresnel.camera.Orthographic.fit(scene, view="front", margin=0.5)

    # Add lighting
    """
    scene.lights = [
        fresnel.light.Light(direction=(0, 0, 1), color=(0.8, 0.8, 0.8)),
        fresnel.light.Light(direction=(1, 1, 1), color=(0.3, 0.3, 0.3))
    ]    # Render the scene to a file
    """

    scene.lights = [
        fresnel.light.Light(direction=(0, 0, 1), color=(0.8, 0.8, 0.8), theta=math.pi),
        fresnel.light.Light(direction=(1, 1, 1), color=(1.1, 1.1, 1.1), theta=math.pi / 3),
    ]
    L = snap.configuration.box[0]*1.1
    scene.camera = fresnel.camera.Orthographic(position=(L * 2, L, L * 2), look_at=(0, 0, 0), up=(0, 1, 0), height=L * 1.4 + 1)

    scene.background_alpha = 1
    scene.background_color = (1, 1, 1)

    output_file = f"{folder}/render_{frame}_{gsd_file.replace('.gsd', '.png')}"
    image = fresnel.pathtrace(scene, samples=64, light_samples=32, w=600, h=600)
    # Convert the Fresnel image to a PIL image
    pil_image = Image.fromarray(image[:], mode="RGBA")

    # Save the image using PIL
    pil_image.save(output_file)
    print(f"Rendered image saved to: {output_file}")

    #plt.figure()
    #ax = plt.subplot(111)
    #ax.imshow(image[:], interpolation='nearest')
    # Add a colorbar
    #sm = plt.cm.ScalarMappable(cmap='bwr', norm=plt.Normalize(vmin=min_length, vmax=max_length))
    #sm.set_array([])
    #cbar = plt.colorbar(sm, cax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    #cbar.set_label(r'$L/D$')
    #plt.axis('off')  # Turn off axes
    #plt.savefig(output_file, bbox_inches='tight', pad_inches=0, dpi=500)
    #print(f"Rendered image saved to: {output_file}")

    return output_file


def render_notebook(folder, gsd_file):
    """Render for Jupyter notebook display"""
    end1_positions, end2_positions = read_configuration(folder, gsd_file)
    # Create a fresnel scene
    scene = fresnel.Scene()

    # Create cylinder geometry for all spherocylinders
    geometry = fresnel.geometry.Cylinder(scene, N=len(end1_positions))

    # Set the end points for each cylinder
    geometry.points[:] = [[end1_positions[i], end2_positions[i]] for i in range(len(end1_positions))]

    # Set the radius for all cylinders
    geometry.radius[:] = 0.5

    # Set material properties
    geometry.material = fresnel.material.Material(color=fresnel.color.linear([0.25, 0.5, 0.9]), roughness=0.8)

    # Set up the camera
    scene.camera = fresnel.camera.Orthographic.fit(scene, view="front", margin=0.5)

    # Add lighting
    scene.lights = [fresnel.light.Light(direction=(0, 0, 1), color=(0.8, 0.8, 0.8)), fresnel.light.Light(direction=(1, 1, 1), color=(0.3, 0.3, 0.3))]

    # Render and return for notebook display
    return fresnel.pathtrace(scene, samples=64, light_samples=32, w=800, h=600)
