from __future__ import division

import meep as mp
from meep.geom import Cylinder, Ellipsoid, index, Vector3, Medium
from meep.simulation import Mirror, Pml, Simulation, no_size
from meep.source import GaussianSource, Source


def main():

    c = Cylinder(radius=3, material=Medium(epsilon_diag=index(3.5)))
    e = Ellipsoid(size=Vector3(1, 2, 1e20))

    src_cmpt = mp.Hz
    sources = Source(src=GaussianSource(1, 0.1), component=src_cmpt, center=Vector3())

    if src_cmpt == mp.Ez:
        symmetries = [Mirror(mp.X), Mirror(mp.Y)]

    if src_cmpt == mp.Hz:
        symmetries = [Mirror(mp.X, -1), Mirror(mp.Y, -1)]

    sim = Simulation(cell_size=Vector3(10, 10, no_size),
                     geometry=[c, e],
                     boundary_layers=[Pml(1.0)],
                     sources=[sources],
                     symmetries=symmetries,
                     resolution=100)

    def print_stuff():
        v = Vector3(4.13, 3.75, 0)
        p = sim._get_field_point(src_cmpt, v)
        print("t, Ez: {} {}+{}i".format(sim._round_time(), p.real, p.imag))

    sim.run(sim.at_beginning(sim.output_epsilon),
            sim.at_every(0.25, print_stuff),
            sim.at_end(print_stuff),
            sim.at_end(sim.output_efield_z),
            until=23)

    print("stopped at meep time = {}".format(sim._round_time()))


if __name__ == '__main__':
    main()
