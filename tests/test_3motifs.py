import sys

sys.path.append("../")
import NEMtropy.graph_classes as sample
import numpy as np
import unittest  # test tool
import NEMtropy.network_functions as mf


class MyTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_count_13(self):
        A = np.array(
            [
                [0, 1, 1, 0],
                [1, 0, 1, 0],
                [1, 1, 0, 0],
                [0, 0, 0, 0],
            ]
        )

        n = mf.motif13_count(A)

        """
        g.solve_tool(
            model="dcm",
            max_steps=200,
            verbose=False,
        )
        """

        # debug

        # test result
        self.assertTrue(n == 6)

    def test_zscore_13(self):
        A = np.array(
            [
                [0, 1, 1, 0],
                [1, 0, 1, 0],
                [1, 1, 0, 0],
                [0, 0, 0, 0],
            ]
        )
        g = sample.DirectedGraph(A)

        g.solve_tool(
            model="dcm",
            max_steps=200,
            verbose=False,
        )

        d = g.motifs_3_zscore()

        # debug
        print(d)

        # test result
        #TODO: write a better motif testing
        self.assertTrue(type(d) is dict)




if __name__ == "__main__":
    unittest.main()
