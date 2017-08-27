import unittest
import numpy as np
import meep as mp


class TestSimulation(unittest.TestCase):

    def test_interpolate_numbers(self):

        expected = [
            1.0, 1.0909090909090908, 1.1818181818181819, 1.2727272727272727, 1.3636363636363635, 1.4545454545454546, 1.5454545454545454, 1.6363636363636365, 1.7272727272727273, 1.8181818181818181, 1.9090909090909092,
            2.0, 2.090909090909091, 2.1818181818181817, 2.272727272727273, 2.3636363636363638, 2.4545454545454546, 2.5454545454545454, 2.6363636363636362, 2.727272727272727, 2.8181818181818183, 2.909090909090909,
            3.0, 3.090909090909091, 3.1818181818181817, 3.272727272727273, 3.3636363636363638, 3.4545454545454546, 3.5454545454545454, 3.6363636363636362, 3.727272727272727, 3.8181818181818183, 3.909090909090909,
            4.0, 4.090909090909091, 4.181818181818182, 4.2727272727272725, 4.363636363636363, 4.454545454545454, 4.545454545454546, 4.636363636363637, 4.7272727272727275, 4.818181818181818, 4.909090909090909,
            5.0, 5.090909090909091, 5.181818181818182, 5.2727272727272725, 5.363636363636363, 5.454545454545454, 5.545454545454546, 5.636363636363637, 5.7272727272727275, 5.818181818181818, 5.909090909090909,
            6.0, 6.090909090909091, 6.181818181818182, 6.2727272727272725, 6.363636363636363, 6.454545454545454, 6.545454545454546, 6.636363636363637, 6.7272727272727275, 6.818181818181818, 6.909090909090909,
            7.0, 7.090909090909091, 7.181818181818182, 7.2727272727272725, 7.363636363636363, 7.454545454545454, 7.545454545454546, 7.636363636363637, 7.7272727272727275, 7.818181818181818, 7.909090909090909,
            8.0, 8.090909090909092, 8.181818181818182, 8.272727272727273, 8.363636363636363, 8.454545454545455, 8.545454545454545, 8.636363636363637, 8.727272727272727, 8.818181818181818, 8.909090909090908,
            9.0, 9.090909090909092, 9.181818181818182, 9.272727272727273, 9.363636363636363, 9.454545454545455, 9.545454545454545, 9.636363636363637, 9.727272727272727, 9.818181818181818, 9.909090909090908,
            10.0
        ]

        nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = mp.interpolate(10, nums)

        np.testing.assert_allclose(expected, result)

    def test_interpolate_vectors(self):

        expected = [
            mp.Vector3(),
            mp.Vector3(0.024999999999999842),
            mp.Vector3(0.04999999999999984),
            mp.Vector3(0.07499999999999984),
            mp.Vector3(0.09999999999999984),
            mp.Vector3(0.12499999999999983),
            mp.Vector3(0.14999999999999983),
            mp.Vector3(0.17499999999999982),
            mp.Vector3(0.19999999999999982),
            mp.Vector3(0.2249999999999998),
            mp.Vector3(0.2499999999999998),
            mp.Vector3(0.2749999999999998),
            mp.Vector3(0.2999999999999998),
            mp.Vector3(0.32499999999999984),
            mp.Vector3(0.34999999999999987),
            mp.Vector3(0.3749999999999999),
            mp.Vector3(0.3999999999999999),
            mp.Vector3(0.42499999999999993),
            mp.Vector3(0.44999999999999996),
            mp.Vector3(0.475),
            mp.Vector3(0.5)
        ]

        res = mp.interpolate(19, [mp.Vector3(), mp.Vector3(0.5)])

        np.testing.assert_allclose([v.x for v in expected], [v.x for v in res])
        np.testing.assert_allclose([v.y for v in expected], [v.y for v in res])
        np.testing.assert_allclose([v.z for v in expected], [v.z for v in res])

    def test_arith_sequence(self):

        expected = [
            0.15,
            0.15040080160320599,
            0.15080160320641198,
            0.15120240480961797,
            0.15160320641282396,
            0.15200400801602995,
            0.15240480961923594,
            0.15280561122244193,
            0.15320641282564793,
            0.15360721442885392
        ]

        res = mp.arith_sequence(0.15, 0.000400801603206, 10)

        self.assertEqual(len(expected), len(res))
        np.testing.assert_allclose(expected, res)


if __name__ == '__main__':
    unittest.main()
