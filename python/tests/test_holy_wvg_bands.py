import unittest
import meep as mp


class TestHolyWvgBands(unittest.TestCase):

    def setUp(self):
        cell = mp.Vector3(1, 12)
        b = mp.Block(size=mp.Vector3(1e20, 1.2, 1e20), material=mp.Medium(epsilon=13))
        c = mp.Cylinder(0.36)

        self.fcen = 0.25
        self.df = 1.5

        s = mp.Source(
            src=mp.GaussianSource(self.fcen, fwidth=self.df),
            component=mp.Hz,
            center=mp.Vector3(0.1234)
        )

        sym = mp.Mirror(direction=mp.Y, phase=-1)

        self.sim = mp.Simulation(
            cell_size=cell,
            geometry=[b, c],
            sources=[s],
            symmetries=[sym],
            boundary_layers=[mp.Pml(1, direction=mp.Y)],
            resolution=20
        )

    def test_run_k_points(self):
        all_freqs = self.sim.run(mp.interpolate(19, [mp.Vector3(), mp.Vector3(0.5)]), k_points=5)

        self.assertAlmostEqual(all_freqs[17][0].real, 0.1942497850393511)
        self.assertAlmostEqual(all_freqs[17][0].imag, 0.001381460274205755)
        self.assertAlmostEqual(all_freqs[18][0].real, 0.19782709203322993)
        self.assertAlmostEqual(all_freqs[18][0].imag, -0.0013233828667934015)
        self.assertAlmostEqual(all_freqs[19][0].real, 0.1927618763491877)
        self.assertAlmostEqual(all_freqs[19][0].imag, 0.001034260690735336)
        self.assertAlmostEqual(all_freqs[20][0].real, 0.19335527231544278)
        self.assertAlmostEqual(all_freqs[20][0].imag, 4.6649450258959025e-4)

    def test_fields_at_kx(self):
        self.sim.k_point = mp.Vector3(3.5)
        h = mp.Harminv(self.sim, mp.Hz, mp.Vector3(0.1234), self.fcen, self.df)
        self.sim.run(self.sim.after_sources(h()), until_after_sources=300)

        expected = [
            (0.19990240131986522, 3.8522735413802275e-8),
            (0.3050067740183294, 4.720168254531566e-7),
            (0.4396104226078593, 1.6233300291010948e-6),
            (0.4582004346509184, 4.7150006592976396e-7),
            (0.5006556112859917, -0.0014396635723422887),
            (0.7405953267896378, -4.553109069353934e-5),
            (0.7627621012715363, -0.006700351645723407),
            (0.8243404528365005, -5.174379068176951e-4),
            (0.8255990399390389, -0.0016256261502000271),
            (0.9494859645499801, -0.005325208458639275),
            (0.9726561278186849, -0.0031192234222098274),
            (0.9855957702101914, -0.003945157134867143),
        ]
        for (r, i), (res, _, _) in zip(expected, h.results):
            self.assertAlmostEqual(res.real, r)
            self.assertAlmostEqual(res.imag, i)


if __name__ == '__main__':
    unittest.main()
