import sys

sys.path.append("../")
import NEMtropy.graph_classes as sample
import numpy as np
import unittest  # test tool
import NEMtropy.network_functions as mf


class MyTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_dyads(self):
        A = np.array(
            [
                [0, 1, 1],
                [1, 0, 0],
                [0, 1, 0],
            ]
        )

        n = mf.count_2motif_2(A)

        """
        g.solve_tool(
            model="dcm",
            max_steps=200,
            verbose=False,
        )
        """

        # debug
        # print(n)

        # test result
        self.assertTrue(n == 2)


    def test_singles(self):
        A = np.array(
            [
                [0, 1, 1],
                [1, 0, 0],
                [0, 1, 0],
            ]
        )

        n = mf.count_2motif_1(A)

        """
        g.solve_tool(
            model="dcm",
            max_steps=200,
            verbose=False,
        )
        """

        # debug
        # print(n)

        # test result
        self.assertTrue(n == 2)

    def test_zeros(self):
        A = np.array(
            [
                [0, 1, 1],
                [1, 0, 0],
                [0, 1, 0],
            ]
        )

        n = mf.count_2motif_0(A)

        """
        g.solve_tool(
            model="dcm",
            max_steps=200,
            verbose=False,
        )
        """

        # debug
        # print(n)

        # test result
        self.assertTrue(n == 0)

    def test_motifs(self):
        A = np.array(
            [
                [0, 1, 1],
                [1, 0, 0],
                [0, 1, 0],
            ]
        )
        g = sample.DirectedGraph(A)

        g.solve_tool(
            model="dcm",
            max_steps=200,
            verbose=False,
        )

        d = g.zscore_2motifs()

        # debug
        # print(n)

        # test result
        #TODO: write a better motif testing
        self.assertTrue(type(d) is dict)




if __name__ == "__main__":
    unittest.main()
