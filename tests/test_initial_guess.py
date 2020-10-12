import sys

sys.path.append("../")
import Directed_graph_Class as sample
import Matrix_Generator as mg
import numpy as np
import unittest  # test tool


class MyTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_dcm_uniform(self):
        n, seed = (4, 22)
        A = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)

        g = sample.DirectedGraph(A)
        g.initial_guess = 'uniform'
        g._set_initial_guess('dcm')
        self.assertTrue(np.concatenate((g.r_x, g.r_y)).all() == np.array([0.,  0.5, 0.5, 0.5, 0.,  0.5]).all())


    def test_dcm(self):
        n, seed = (4, 22)
        A = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)

        x0 = np.random.rand(2*n)
        g = sample.DirectedGraph(A)
        g.initial_guess = x0
        g._set_initial_guess('dcm')
        g._set_solved_problem_dcm(x0)
        self.assertTrue(np.concatenate((g.x, g.y)).all() == x0.all())


    def test_dcm_new_uniform(self):
        n, seed = (4, 22)
        A = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)

        g = sample.DirectedGraph(A)
        g.initial_guess = 'uniform'
        g._set_initial_guess('dcm_new')
        self.assertTrue(np.concatenate((g.r_x, g.r_y)).all() == np.array([1e3,  0.5, 0.5, 0.5, 1e3,  0.5]).all())


    def test_dcm_new(self):
        n, seed = (4, 22)
        A = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)

        x0 = np.random.rand(2*n)
        g = sample.DirectedGraph(A)
        g.initial_guess = x0
        g._set_initial_guess('dcm_new')
        g._set_solved_problem_dcm(x0)
        self.assertTrue(np.concatenate((g.x, g.y)).all() == x0.all())

    def test_decm_uniform(self):
        n, seed = (4, 22)
        A = mg.random_weighted_matrix_generator_dense(
            n, sym=False, seed=seed, sup_ext=100, intweights=True
        )


        g = sample.DirectedGraph(A)
        g.initial_guess = 'uniform'
        g._set_initial_guess('decm')
        self.assertTrue(np.concatenate((g.x, g.y, g.out_strength, g.in_strength)).all() == np.ones(4*n).all())


    def test_decm(self):
        n, seed = (4, 22)
        A = mg.random_weighted_matrix_generator_dense(
            n, sym=False, seed=seed, sup_ext=100, intweights=True
        )

        x0 = np.random.rand(4*n)
        g = sample.DirectedGraph(A)
        g.initial_guess = x0
        g._set_initial_guess('decm')
        g._set_solved_problem_decm(x0)
        self.assertTrue(np.concatenate((g.x, g.y)).all() == x0.all())



    def test_decm_new_uniform(self):
        n, seed = (4, 22)
        A = mg.random_weighted_matrix_generator_dense(
            n, sym=False, seed=seed, sup_ext=100, intweights=True
        )


        g = sample.DirectedGraph(A)
        g.initial_guess = 'uniform'
        g._set_initial_guess('decm_new')
        self.assertTrue(np.concatenate((g.x, g.y, g.out_strength, g.in_strength)).all() == np.ones(4*n).all())


    def test_decm_new(self):
        n, seed = (4, 22)
        A = mg.random_weighted_matrix_generator_dense(
            n, sym=False, seed=seed, sup_ext=100, intweights=True
        )

        x0 = np.random.rand(4*n)
        g = sample.DirectedGraph(A)
        g.initial_guess = x0
        g._set_initial_guess('decm_new')
        g._set_solved_problem_decm(x0)
        self.assertTrue(np.concatenate((g.x, g.y,g.out_strength, g.in_strength)).all() == x0.all())


    def test_CREAMA_uniform(self):
        n, seed = (4, 22)
        A = mg.random_weighted_matrix_generator_dense(
            n, sym=False, seed=seed, sup_ext=100, intweights=True
        )

        g = sample.DirectedGraph(A)
        g.initial_guess = 'strengths_minor'
        g._set_initial_guess_CReAMa()
        x = np.concatenate((sample.out_strength(A)/(sample.out_strength(A) + 1), sample.in_strength(A)/(sample.in_strength(A) + 1)))
        self.assertTrue(g.x0.all() == x.all())


    def test_CREAMA(self):
        n, seed = (4, 22)
        A = mg.random_weighted_matrix_generator_dense(
            n, sym=False, seed=seed, sup_ext=100, intweights=True
        )

        x0 = np.random.rand(2*n)
        g = sample.DirectedGraph(A)
        g.initial_guess = x0
        g._set_initial_guess_CReAMa()
        g._set_solved_problem_decm(x0)
        self.assertTrue(g.x0.all() == x0.all())


if __name__ == "__main__":
    unittest.main()