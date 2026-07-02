import numpy as np
import pytest

import NEMtropy.graph_classes as gc
import NEMtropy.matrix_generator as mg
import NEMtropy.network_functions as ntw_f


@pytest.mark.math
class TestInitialGuess:
    def test_cm_uniform(self):
        n, seed = (4, 22)
        a = mg.random_binary_matrix_generator_dense(n, sym=True, seed=seed)
        g = gc.UndirectedGraph(a)
        g.initial_guess = "uniform"
        g.last_model = "cm"
        g._set_initial_guess("cm")
        assert (g.x0 == np.array([0.5, 0.5])).all()

    def test_cm_custom(self):
        n, seed = (4, 22)
        a = mg.random_binary_matrix_generator_dense(n, sym=True, seed=seed)
        x0 = np.random.rand(n)
        g = gc.UndirectedGraph(a)
        g.initial_guess = x0
        g._set_initial_guess_cm()
        g.full_return = False
        g.last_model = "cm"
        g._set_solved_problem_cm(g.x0)
        assert g.x.all() == x0.all()

    def test_crema_undirected_uniform(self):
        n, seed = (4, 22)
        a = mg.random_weighted_matrix_generator_dense(
            n, sym=False, seed=seed, sup_ext=100, intweights=True
        )
        g = gc.UndirectedGraph(a)
        g.initial_guess = "strengths_minor"
        g._set_initial_guess("crema")
        x = (g.strength_sequence > 0).astype(float) / (g.strength_sequence + 1)
        assert g.x0.all() == x.all()

    def test_crema_undirected_custom(self):
        n, seed = (4, 22)
        a = mg.random_weighted_matrix_generator_dense(
            n, sym=False, seed=seed, sup_ext=100, intweights=True
        )
        x0 = np.random.rand(n)
        g = gc.UndirectedGraph(a)
        g.initial_guess = x0
        g._set_initial_guess_crema_undirected()
        g.full_return = False
        g._set_solved_problem_crema_undirected(g.x0)
        assert g.beta.all() == x0.all()

    def test_ecm_uniform(self):
        n, seed = (4, 22)
        a = mg.random_weighted_matrix_generator_dense(
            n, sym=False, seed=seed, sup_ext=100, intweights=True
        )
        g = gc.UndirectedGraph(a)
        g.initial_guess = "strengths_minor"
        g.last_model = "ecm"
        g._set_initial_guess("ecm")
        x = (g.strength_sequence > 0).astype(float) / (g.strength_sequence + 1)
        assert g.x0.all() == x.all()

    def test_ecm_custom(self):
        n, seed = (4, 22)
        a = mg.random_weighted_matrix_generator_dense(
            n, sym=False, seed=seed, sup_ext=100, intweights=True
        )
        x0 = np.random.rand(n)
        g = gc.UndirectedGraph(a)
        g.initial_guess = x0
        g._set_initial_guess_crema_undirected()
        assert g.x0.all() == x0.all()

    def test_dcm_uniform(self):
        n, seed = (4, 22)
        a = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)
        g = gc.DirectedGraph(a)
        g.initial_guess = "uniform"
        g._set_initial_guess("dcm")
        expected = np.array([0.0, 0.5, 0.5, 0.5, 0.0, 0.5])
        assert (np.concatenate((g.r_x, g.r_y)) == expected).all()

    def test_dcm_custom(self):
        n, seed = (4, 22)
        a = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)
        x0 = np.random.rand(2 * n)
        g = gc.DirectedGraph(a)
        g.initial_guess = x0
        g._set_initial_guess("dcm")
        g._set_solved_problem_dcm(x0)
        assert np.concatenate((g.x, g.y)).all() == x0.all()

    def test_dcm_exp_uniform(self):
        n, seed = (4, 22)
        a = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)
        g = gc.DirectedGraph(a)
        g.initial_guess = "uniform"
        g._set_initial_guess("dcm_exp")
        expected = np.array(
            [1e3, -np.log(0.5), -np.log(0.5), -np.log(0.5), 1e3, -np.log(0.5)]
        )
        assert (np.concatenate((g.r_x, g.r_y)) == expected).all()

    def test_dcm_exp_custom(self):
        n, seed = (4, 22)
        a = mg.random_binary_matrix_generator_dense(n, sym=False, seed=seed)
        x0 = np.random.rand(2 * n)
        g = gc.DirectedGraph(a)
        g.initial_guess = x0
        g._set_initial_guess("dcm_exp")
        g._set_solved_problem_dcm(x0)
        assert np.concatenate((g.x, g.y)).all() == x0.all()

    def test_decm_uniform(self):
        n, seed = (4, 22)
        a = mg.random_weighted_matrix_generator_dense(
            n, sym=False, seed=seed, sup_ext=100, intweights=True
        )
        g = gc.DirectedGraph(a)
        g.initial_guess = "uniform"
        g._set_initial_guess("decm")
        assert (
            np.concatenate((g.x, g.y, g.out_strength, g.in_strength)).all()
            == np.ones(4 * n).all()
        )

    def test_decm_custom(self):
        n, seed = (4, 22)
        a = mg.random_weighted_matrix_generator_dense(
            n, sym=False, seed=seed, sup_ext=100, intweights=True
        )
        x0 = np.random.rand(4 * n)
        g = gc.DirectedGraph(a)
        g.initial_guess = x0
        g._set_initial_guess("decm")
        g._set_solved_problem_decm(x0)
        assert np.concatenate((g.x, g.y)).all() == x0.all()

    def test_decm_exp_uniform(self):
        n, seed = (4, 22)
        a = mg.random_weighted_matrix_generator_dense(
            n, sym=False, seed=seed, sup_ext=100, intweights=True
        )
        g = gc.DirectedGraph(a)
        g.initial_guess = "uniform"
        g._set_initial_guess("decm_exp")
        tester = np.exp(np.ones(4 * n))
        assert g.x0.all() == tester.all()

    def test_decm_exp_custom(self):
        n, seed = (4, 22)
        a = mg.random_weighted_matrix_generator_dense(
            n, sym=False, seed=seed, sup_ext=100, intweights=True
        )
        x0 = np.random.rand(4 * n)
        g = gc.DirectedGraph(a)
        g.initial_guess = x0
        g._set_initial_guess("decm_exp")
        g._set_solved_problem_decm(x0)
        assert (
            np.concatenate((g.x, g.y, g.out_strength, g.in_strength)).all() == x0.all()
        )

    def test_crema_directed_uniform(self):
        n, seed = (4, 22)
        a = mg.random_weighted_matrix_generator_dense(
            n, sym=False, seed=seed, sup_ext=100, intweights=True
        )
        g = gc.DirectedGraph(a)
        g.initial_guess = "strengths_minor"
        g._set_initial_guess_crema_directed()
        x = np.concatenate(
            (
                ntw_f.out_strength(a) / (ntw_f.out_strength(a) + 1),
                ntw_f.in_strength(a) / (ntw_f.in_strength(a) + 1),
            )
        )
        assert g.x0.all() == x.all()

    def test_crema_directed_custom(self):
        n, seed = (4, 22)
        a = mg.random_weighted_matrix_generator_dense(
            n, sym=False, seed=seed, sup_ext=100, intweights=True
        )
        x0 = np.random.rand(2 * n)
        g = gc.DirectedGraph(a)
        g.initial_guess = x0
        g._set_initial_guess_crema_directed()
        g._set_solved_problem_decm(x0)
        assert g.x0.all() == x0.all()
