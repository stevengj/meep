from __future__ import division

import math
import meep as mp
from meep import mpb
from scipy.optimize import minimize_scalar


def print_heading(h):
    stars = "*" * 10
    print("{0} {1} {0}".format(stars, h))

# Our First Band Structure

print_heading("Square lattice of rods in air")

num_bands = 8
k_points = [mp.Vector3(),          # Gamma
            mp.Vector3(0.5),       # X
            mp.Vector3(0.5, 0.5),  # M
            mp.Vector3()]          # Gamma

k_points = mp.interpolate(4, k_points)
geometry = [mp.Cylinder(0.2, material=mp.Medium(epsilon=12))]
geometry_lattice = mp.Lattice(size=mp.Vector3(1, 1))
resolution = 32

ms = mpb.ModeSolver(num_bands=num_bands,
                    k_points=k_points,
                    geometry=geometry,
                    geometry_lattice=geometry_lattice,
                    resolution=resolution)

print_heading("Square lattice of rods: TE bands")
ms.run_te()

print_heading("Square lattice of rods: TM bands")
ms.run_tm()

print_heading("Square lattice of rods: TM, w/efield")
ms.run_tm(mpb.output_efield_z)

print_heading("Square lattice of rods: TE, w/hfield & dpwr")
ms.run_te(mpb.output_at_kpoint(mp.Vector3(0.5), mpb.output_hfield_z, mpb.output_dpwr))

# Bands of a Triangular Lattice

print_heading("Triangular lattice of rods in air")

ms.geometry_lattice = mp.Lattice(size=mp.Vector3(1, 1),
                                 basis1=mp.Vector3(math.sqrt(3) / 2, 0.5),
                                 basis2=mp.Vector3(math.sqrt(3) / 2, -0.5))

ms.k_points = [mp.Vector3(),               # Gamma
               mp.Vector3(y=0.5),          # M
               mp.Vector3(-1 / 3, 1 / 3),  # K
               mp.Vector3()]               # Gamma

ms.k_points = mp.interpolate(4, k_points)
ms.run_tm()

# Maximizing the First TM Gap

print_heading('Maximizing the first TM gap')


def first_tm_gap(r):
    ms.geometry = [mp.Cylinder(r, material=mp.Medium(epsilon=12))]
    ms.run_tm()
    return -1 * ms.retrieve_gap(1)

ms.num_bands = 2
ms.mesh_size = 7

result = minimize_scalar(first_tm_gap, method='bounded', bounds=[0.1, 0.5], tol=0.1)
print("radius at maximum: {}".format(result.x))
print("gap size at maximum: {}".format(result.fun * -1))

ms.mesh_size = 3  # Reset to default value of 3

# A Complete 2D Gap with an Anisotropic Dielectric

print_heading('Anisotropic complete 2d gap')

ms.geometry = [mp.Cylinder(0.3, material=mp.Medium(epsilon_diag=mp.Vector3(1, 1, 12)))]

ms.default_material = mp.Medium(epsilon_diag=mp.Vector3(12, 12, 1))
ms.num_bands = 8
ms.run()  # just use run, instead of run_te or run_tm, to find the complete gap
