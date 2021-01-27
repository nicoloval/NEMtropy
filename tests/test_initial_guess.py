import sys

sys.path.append("../")
import netrecon.graph_classes as sample
import netrecon.matrix_generator as sample_u
import netrecon.Matrix_Generator as mg
import numpy as np
import unittest  # test tool


class MyTest(unittest.TestCase):
    def setUp(self):
        pass


    def test_cm_uniform(self):
        n, seed = (4, 22)
        A = mg.random_binary_matrix_generator_dense(n, sym=True, seed=seed)

        g = sample_u.UndirectedGraph(A)
        g.initial_guess = 'uniform'
        g.last_model = "cm"
        g._set_initial_guess('cm')

        self.assertTrue(g.x0.all() == np.array([0.5,  0.5]).all())


    def test_cm(self):
        n, seed = (4, 22)
        A = mg.random_binary_matrix_generator_dense(n, sym=True, seed=seed)

        x0 = np.random.rand(n)
        g = sample_u.UndirectedGraph(A)
        g.initial_guess = x0
        g._set_initial_guess_cm()
        g.full_return = False
        g.last_model = "cm"
        g._set_solved_problem_cm(g.x0)
        self.assertTrue(g.x.all() == x0.all())


    def test_crema_uniform(self):
        n, seed = (4, 22)
        A = mg.random_weighted_matrix_generator_dense(
            n, sym=False, seed=seed, sup_ext=100, intweights=True
        )

        g = sample_u.UndirectedGraph(A)
        g.initial_guess = 'strengths_minor'
        g._set_initial_guess('crema')

        x = (g.strength_sequence > 0).astype(float) / (
                    g.strength_sequence + 1
                )
        self.assertTrue(g.x0.all() == x.all())


    def test_crema(self):
        n, seed = (4, 22)
        A = mg.random_weighted_matrix_generator_dense(
            n, sym=False, seed=seed, sup_ext=100, intweights=True
        )

        x0 = np.random.rand(n)
        g = sample_u.UndirectedGraph(A)
        g.initial_guess = x0
        g._set_initial_guess_crema()
        g.full_return = False
        g._set_solved_problem_crema(g.x0)
        self.assertTrue(g.beta.all() == x0.all())


    def test_ecm_uniform(self):
        n, seed = (4, 22)
        A = mg.random_weighted_matrix_generator_dense(
            n, sym=False, seed=seed, sup_ext=100, intweights=True
        )

        g = sample_u.UndirectedGraph(A)
        g.initial_guess = 'strengths_minor'
        g.last_model = "ecm"
        g._set_initial_guess('ecm')

        x = (g.strength_sequence > 0).astype(float) / (
                    g.strength_sequence + 1
                )
        self.assertTrue(g.x0.all() == x.all())


    def test_ecm(self):
        n, seed = (4, 22)
        A = mg.random_weighted_matrix_generator_dense(
            n, sym=False, seed=seed, sup_ext=100, intweights=True
        )

        x0 = np.random.rand(n)
        g = sample_u.UndirectedGraph(A)
        g.initial_guess = x0
        g._set_initial_guess_crema()
        self.assertTrue(g.x0.all() == x0.all())



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
        self.assertTrue(np.concatenate((g.r_x, g.r_y)).all() == np.array([1e3,  -np.log(0.5), -np.log(0.5), -np.log(0.5), 1e3,  -np.log(0.5)]).all())


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
        tester = np.exp(np.ones(4*n))
        self.assertTrue(g.x0.all() == tester.all())


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


    def test_crema_uniform(self):
        n, seed = (4, 22)
        A = mg.random_weighted_matrix_generator_dense(
            n, sym=False, seed=seed, sup_ext=100, intweights=True
        )

        g = sample.DirectedGraph(A)
        g.initial_guess = 'strengths_minor'
        g._set_initial_guess_crema()
        x = np.concatenate((sample.out_strength(A)/(sample.out_strength(A) + 1), sample.in_strength(A)/(sample.in_strength(A) + 1)))
        self.assertTrue(g.x0.all() == x.all())


    def test_crema(self):
        n, seed = (4, 22)
        A = mg.random_weighted_matrix_generator_dense(
            n, sym=False, seed=seed, sup_ext=100, intweights=True
        )

        x0 = np.random.rand(2*n)
        g = sample.DirectedGraph(A)
        g.initial_guess = x0
        g._set_initial_guess_crema()
        g._set_solved_problem_decm(x0)
        self.assertTrue(g.x0.all() == x0.all())


if __name__ == "__main__":
    unittest.main()
