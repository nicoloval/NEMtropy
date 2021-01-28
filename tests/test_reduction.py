import sys

sys.path.append("../")
import NEMtropy.graph_classes as sample
import numpy as np
import unittest  # test tool


class MyTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_dcm(self):
        A = np.array([[0, 1, 1, 0], [1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 0, 0]])

        g = sample.DirectedGraph()
        g._initialize_graph(A)

        g.degree_reduction()

        g.initial_guess = "degrees"
        g._set_initial_guess("dcm")

        # debug
        # print(g.r_dseq_out)
        # print(g.r_dseq_in)
        # print(g.rnz_dseq_out)
        # print(g.rnz_dseq_in)

        sol = np.concatenate((g.r_x, g.r_y))
        g.last_model = "dcm"
        g._set_solved_problem(sol)

        # test result
        self.assertTrue(g.dseq_out.all() == g.x.all())
        self.assertTrue(g.dseq_in.all() == g.y.all())


if __name__ == "__main__":
    unittest.main()
