from __future__ import division

import unittest
# import numpy as np
import meep as mp
from meep import mpb


class TestMPBWrappers(unittest.TestCase):

    def setUp(self):
        self.num_bands = 8
        self.k_points = [mp.Vector3(),
                         mp.Vector3(0.5),
                         mp.Vector3(0.5, 0.5),
                         mp.Vector3()]

        self.k_points = mp.interpolate(4, self.k_points)
        self.geometry = []  # [mp.Cylinder(0.2, material=mp.Medium(epsilon=12))]
        self.geometry_lattice = mp.Lattice(size=mp.Vector3(1, 1))
        self.resolution = 32

    def test_mode_solver_constructor(self):
        mpb.mode_solver(self.num_bands, 0, self.resolution, self.geometry_lattice,
                        1.0e-7, mp.Medium(), self.geometry, True)


class TestModeSolver(unittest.TestCase):

    # def test_list_split(self):
    #     k_points = [
    #         mp.Vector3(),
    #         mp.Vector3(0.5),
    #         mp.Vector3(0.5, 0.5),
    #         mp.Vector3()
    #     ]

    #     k_points = mp.interpolate(4, k_points)

    #     k_split = mp.list_split(k_points, 1, 0)

    #     expected = [
    #         (0, [mp.Vector3(),
    #              mp.Vector3(0.10000000000000003),
    #              mp.Vector3(0.20000000000000004),
    #              mp.Vector3(0.30000000000000004),
    #              mp.Vector3(0.4),
    #              mp.Vector3(0.5),
    #              mp.Vector3(0.5, 0.10000000000000003),
    #              mp.Vector3(0.5, 0.20000000000000004),
    #              mp.Vector3(0.5, 0.30000000000000004),
    #              mp.Vector3(0.5, 0.4),
    #              mp.Vector3(0.5, 0.5),
    #              mp.Vector3(0.4, 0.4),
    #              mp.Vector3(0.30000000000000004, 0.30000000000000004),
    #              mp.Vector3(0.2, 0.2),
    #              mp.Vector3(0.1, 0.1),
    #              mp.Vector3(0.0, 0.0)]),
    #     ]

    #     indx = k_split[0][0]
    #     split_list = k_split[0][1]
    #     self.assertEqual(indx, 0)
    #     for res, exp in zip(split_list, expected[0][1]):
    #         self.assertEqual(res, exp)

    def test_no_geometry(self):
        num_bands = 8
        k_points = [
            mp.Vector3(),
            mp.Vector3(0.5),
            mp.Vector3(0.5, 0.5),
            mp.Vector3()
        ]

        k_points = mp.interpolate(4, k_points)
        geometry_lattice = mp.Lattice(size=mp.Vector3(1, 1))
        resolution = 32

        ms = mpb.ModeSolver(
            num_bands=num_bands,
            k_points=k_points,
            geometry=[],
            geometry_lattice=geometry_lattice,
            resolution=resolution
        )

        ms.run_te()

        expected_freqs = [
            (0.0, 1.0000000001570188, 1.0000000001947995, 1.0000000002026501, 1.000000000750821,
             1.414213562733575, 1.4142135632238149, 1.414213563312462),
            (0.1000000000000001, 0.9000000008929053, 1.0049875625631484, 1.0049875626969098,
             1.1000000004793433, 1.3453624054610431, 1.3453624081388835, 1.4866068768659069),
            (0.2, 0.8000000000719706, 1.0198039027815926, 1.01980390279274, 1.2000000000658297,
             1.280624847523883, 1.280624847606171, 1.5620499354187383),
            (0.30000000000000004, 0.7000000000046291, 1.0440306508946555, 1.0440306508951056,
             1.2206555615753751, 1.2206555615832708, 1.3000000000088705, 1.6401219467331354),
            (0.4, 0.6000000000018006, 1.0770329614278362, 1.0770329614281615, 1.1661903789694668,
             1.1661903789701553, 1.400000000001976, 1.7204650534165242),
            (0.5000000000000002, 0.5000000000025517, 1.1180339887503423, 1.1180339887507464,
             1.118033988750948, 1.1180339887513642, 1.500000000000794, 1.8027756377205417),
            (0.5099019513592784, 0.509901951381978, 1.0295630141059338, 1.0295630141089662,
             1.2083045973641922, 1.2083045973704456, 1.5033296378428687, 1.749285567925871),
            (0.53851648071345, 0.5385164868413189, 0.9433981154548183, 0.9433981163928138,
             1.3000000012439332, 1.3000000028796745, 1.5132745965194043, 1.699999987609613),
            (0.58309518948453, 0.5830951896783285, 0.8602325267882038, 0.8602325268225337,
             1.39283882775581, 1.3928388278036776, 1.5297058541275403, 1.655294522569475),
            (0.640312423743284, 0.640312424422913, 0.7810249679478669, 0.7810249680931175,
             1.486606874864943, 1.4866068750341963, 1.5524174697967996, 1.6155494317516885),
            (0.7071067811865472, 0.7071067811865628, 0.7071067811865641, 0.7071067811865748,
             1.5811388300841978, 1.5811388300842095, 1.581138830084304, 1.5811388300858844),
            (0.5656854249492377, 0.7211102550933052, 0.7211102550936606, 0.8485281374242766,
             1.456021977856692, 1.4560219778567864, 1.5231546211729912, 1.5231546211887619),
            (0.4242640687119287, 0.7615773105865216, 0.761577310586608, 0.9899494936612568,
             1.3341664064127197, 1.3341664064127845, 1.476482306023386, 1.4764823060236447),
            (0.28284271247461906, 0.8246211251235365, 0.8246211251235387, 1.1313708498984796,
             1.2165525060596485, 1.2165525060596536, 1.4422205101855965, 1.4422205101856658),
            (0.1414213562373096, 0.9055385138137416, 0.9055385138137424, 1.1045361017187267,
             1.1045361017187274, 1.272792206135787, 1.421267040355189, 1.4212670403551926),
            (0.0, 0.9999999999999994, 0.9999999999999996, 1.0000000000000002, 1.0000000000000009,
             1.4142135623730947, 1.4142135623730951, 1.4142135623730954)
        ]

        for res, exp in zip(ms.all_freqs, expected_freqs):
            for r, e in zip(res, exp):
                self.assertAlmostEqual(r, e)
            # np.testing.assert_allclose(res, exp)

if __name__ == '__main__':
    unittest.main()
