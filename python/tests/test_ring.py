# Python port of meep/examples/ring.ctl
# Calculating 2d ring-resonator modes, from the Meep tutorial.
from __future__ import division

import unittest
import meep as mp


def dummy_eps(vec):
    return 1.0


class TestRing(unittest.TestCase):

    def init(self):
        n = 3.4
        w = 1
        r = 1
        pad = 4
        dpml = 2
        sxy = 2 * (r + w + pad + dpml)

        dielectric = mp.Medium(epsilon=n * n)
        air = mp.Medium()

        c1 = mp.Cylinder(r + w, material=dielectric)
        c2 = mp.Cylinder(r, material=air)

        fcen = 0.15
        df = 0.1

        src = mp.Source(mp.GaussianSource(fcen, fwidth=df), mp.Ez, mp.Vector3(r + 0.1))

        self.sim = mp.Simulation(cell_size=mp.Vector3(sxy, sxy),
                                 geometry=[c1, c2],
                                 sources=[src],
                                 resolution=10,
                                 symmetries=[mp.Mirror(mp.Y)],
                                 boundary_layers=[mp.Pml(dpml)])

        self.h = mp.Harminv(self.sim, mp.Ez, mp.Vector3(r + 0.1), fcen, df)

    def test_harminv(self):
        self.init()

        self.sim.run(
            self.sim.at_beginning(self.sim.output_epsilon),
            self.sim.after_sources(self.h()),
            until_after_sources=300
        )
        band1, band2, band3 = self.h.harminv_results

        self.assertAlmostEqual(band1[0].real, 0.118101315147, places=3)
        self.assertAlmostEqual(band1[0].imag, -0.000731513241623, places=3)
        self.assertAlmostEqual(abs(band1[1].real), 0.00341267634436, places=3)
        self.assertAlmostEqual(band1[1].real, -0.00304951667301, places=3)
        self.assertAlmostEqual(band1[1].imag, -0.00153192946717, places=3)

        fp = self.sim._get_field_point(mp.Ez, mp.Vector3(1, 1))
        self.assertAlmostEqual(fp, -0.08185972142450348)

if __name__ == '__main__':
    unittest.main()
