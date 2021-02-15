import numpy as np
import scipy.sparse
from scipy.stats import norm, poisson
import scipy
import os
from tqdm import tqdm
from functools import partial
from platform import system
if system() != 'Windows':
    from multiprocessing import Pool
from . import models_functions as mof
from . import solver_functions as sof
from . import network_functions as nef
from . import ensemble_generator as eg
from . import poibin as pb


class UndirectedGraph:
    """Undirected Graph instance can be initialised with adjacency
    matrix, edgelist, degrees sequence or strengths sequence.

    :param adjacency: Adjacency matrix, defaults to None.
    :type adjacency: numpy.ndarray, list or scipy.sparse_matrix, optional
    :param edgelist: edgelist, defaults to None.
    :type edgelist: numpy.ndarray, list, optional
    :param degree_sequence: degrees sequence, defaults to None.
    :type degree_sequence: numpy.ndarray, optional
    :param strength_sequence: strengths sequence, defaults to None.
    :type strength_sequence: numpy.ndarray, optional
    """
    def __init__(
        self,
        adjacency=None,
        edgelist=None,
        degree_sequence=None,
        strength_sequence=None,
    ):
        self.n_nodes = None
        self.n_edges = None
        self.adjacency = None
        self.is_sparse = False
        self.edgelist = None
        self.dseq = None
        self.strength_sequence = None
        self.nodes_dict = None
        self.is_initialized = False
        self.is_randomized = False
        self.is_weighted = False
        self._initialize_graph(
            adjacency=adjacency,
            edgelist=edgelist,
            degree_sequence=degree_sequence,
            strength_sequence=strength_sequence,
        )

        self.avg_mat = None

        self.initial_guess = None
        # Reduced problem parameters
        self.is_reduced = False
        self.r_dseq = None
        self.r_n = None
        self.r_invert_dseq = None
        self.r_dim = None
        self.r_multiplicity = None

        # Problem solutions
        self.x = None
        self.beta = None

        # reduced solutions
        self.r_x = None

        # Problem (reduced) residuals
        self.residuals = None
        self.final_result = None
        self.r_beta = None

        self.nz_index = None
        self.rnz_n = None

        # model
        self.x0 = None
        self.error = None
        self.error_degree = None
        self.relative_error_degree = None
        self.error_strength = None
        self.relative_error_strength = None
        self.full_return = False
        self.last_model = None

        # function
        self.args = None

    def _initialize_graph(
        self,
        adjacency=None,
        edgelist=None,
        degree_sequence=None,
        strength_sequence=None,
    ):
        # Here we can put controls over the type of input. For instance, if the graph is directed,
        # i.e. adjacency matrix is asymmetric, the class to use must be the DiGraph,
        # or if the graph is weighted (edgelist contains triplets or matrix is not binary) or bipartite

        if adjacency is not None:
            if not isinstance(
                adjacency, (list, np.ndarray)
            ) and not scipy.sparse.isspmatrix(adjacency):
                raise TypeError(
                    "The adjacency matrix must be passed as a list or"
                    "numpy array or scipy sparse matrix."
                )
            elif adjacency.size > 0:
                if np.sum(adjacency < 0):
                    raise TypeError(
                        "The adjacency matrix entries must be positive."
                    )
                if isinstance(
                    adjacency, list
                ):
                    self.adjacency = np.array(adjacency)
                elif isinstance(adjacency, np.ndarray):
                    self.adjacency = adjacency
                else:
                    self.adjacency = adjacency
                    self.is_sparse = True
                if np.sum(adjacency) == np.sum(adjacency > 0):
                    self.dseq = nef.degree(adjacency).astype(np.float64)
                else:
                    self.dseq = nef.degree(adjacency).astype(np.float64)
                    self.strength_sequence = nef.strength(adjacency).astype(
                        np.float64
                    )
                    self.nz_index = np.nonzero(self.strength_sequence)[0]
                    self.is_weighted = True

                self.n_nodes = len(self.dseq)
                self.n_edges = np.sum(self.dseq)/2
                self.is_initialized = True

        elif edgelist is not None:
            if not isinstance(edgelist, (list, np.ndarray)):
                raise TypeError(
                    "The edgelist must be passed as a list or numpy array."
                )
            elif len(edgelist) > 0:
                if len(edgelist[0]) > 3:
                    raise ValueError(
                        "This is not an edgelist. An edgelist must be a list"
                        "or array of couples of nodes with optional weights."
                        "Is this an adjacency matrix?"
                    )
                elif len(edgelist[0]) == 2:
                    (
                        self.edgelist,
                        self.dseq,
                        self.nodes_dict,
                    ) = nef.edgelist_from_edgelist_undirected(edgelist)
                else:
                    (
                        self.edgelist,
                        self.dseq,
                        self.strength_sequence,
                        self.nodes_dict,
                    ) = nef.edgelist_from_edgelist_undirected(edgelist)
                self.n_nodes = len(self.dseq)
                self.n_edges = np.sum(self.dseq)/2
                self.is_initialized = True

        elif degree_sequence is not None:
            if not isinstance(degree_sequence, (list, np.ndarray)):
                raise TypeError(
                    "The degree sequence must be passed as a list or"
                    "numpy array."
                )
            elif len(degree_sequence) > 0:
                try:
                    int(degree_sequence[0])
                except:
                    raise TypeError(
                        "The degree sequence must contain numeric values."
                    )
                if (np.array(degree_sequence) < 0).sum() > 0:
                    raise ValueError("A degree cannot be negative.")
                else:
                    self.n_nodes = int(len(degree_sequence))
                    self.dseq = degree_sequence.astype(np.float64)
                    self.n_edges = np.sum(self.dseq)/2
                    self.is_initialized = True

                if strength_sequence is not None:
                    if not isinstance(strength_sequence, (list, np.ndarray)):
                        raise TypeError(
                            "The strength sequence must be passed as a"
                            "list or numpy array."
                        )
                    elif len(strength_sequence):
                        try:
                            int(strength_sequence[0])
                        except:
                            raise TypeError(
                                "The strength sequence must contain numeric"
                                "values."
                            )
                        if (np.array(strength_sequence) < 0).sum() > 0:
                            raise ValueError("A strength cannot be negative.")
                        else:
                            if len(strength_sequence) != len(degree_sequence):
                                raise ValueError(
                                    "Degrees and strengths arrays must have"
                                    "same length."
                                )
                            self.n_nodes = int(len(strength_sequence))
                            self.strength_sequence = strength_sequence.astype(
                                np.float64
                            )
                            self.nz_index = np.nonzero(self.strength_sequence)[
                                0
                            ]
                            self.is_weighted = True
                            self.is_initialized = True

        elif strength_sequence is not None:
            if not isinstance(strength_sequence, (list, np.ndarray)):
                raise TypeError(
                    "The strength sequence must be passed as a list"
                    "or numpy array."
                )
            elif len(strength_sequence):
                try:
                    int(strength_sequence[0])
                except:
                    raise TypeError(
                        "The strength sequence must contain numeric values."
                    )
                if (np.array(strength_sequence) < 0).sum() > 0:
                    raise ValueError("A strength cannot be negative.")
                else:
                    self.n_nodes = int(len(strength_sequence))
                    self.strength_sequence = strength_sequence
                    self.nz_index = np.nonzero(self.strength_sequence)[0]
                    self.is_weighted = True
                    self.is_initialized = True

    def set_adjacency_matrix(self, adjacency):
        """Initialises graph given the adjacency matrix.

        :param adjacency: ajdacency matrix.
        :type adjacency: numpy.ndarray, list, scipy.sparse_matrix
        """
        if self.is_initialized:
            print(
                "Graph already contains edges or has a degree sequence."
                "Use clean_edges() first."
            )
        else:
            self._initialize_graph(adjacency=adjacency)

    def set_edgelist(self, edgelist):
        """Initialises graph given the edgelist.

        :param edgelist: edgelist.
        :type edgelist: numpy.ndarray, list
        """
        if self.is_initialized:
            print(
                "Graph already contains edges or has a degree sequence."
                "Use clean_edges() first."
            )
        else:
            self._initialize_graph(edgelist=edgelist)

    def set_degree_sequences(self, degree_sequence):
        """Initialises graph given the degrees sequence.

        :param degree_sequence: degrees sequence.
        :type degree_sequence: numpy.ndarray
        """
        if self.is_initialized:
            print(
                "Graph already contains edges or has a degree sequence."
                "Use clean_edges() first."
            )
        else:
            self._initialize_graph(degree_sequence=degree_sequence)

    def clean_edges(self):
        """Deletes all the initialiased attributes.
        """
        self.adjacency = None
        self.edgelist = None
        self.deg_seq = None
        self.is_initialized = False

    def _solve_problem(
        self,
        initial_guess=None,
        model="cm",
        method="quasinewton",
        max_steps=100,
        tol=1e-8,
        eps=1e-8,
        full_return=False,
        verbose=False,
        linsearch=True,
        regularise=True,
    ):
        self.last_model = model
        self.full_return = full_return
        self.initial_guess = initial_guess
        self.regularise = regularise
        self._initialize_problem(self.last_model, method)
        x0 = self.x0

        sol = sof.solver(
            x0,
            fun=self.fun,
            fun_jac=self.fun_jac,
            step_fun=self.step_fun,
            linsearch_fun=self.fun_linsearch,
            hessian_regulariser=self.hessian_regulariser,
            tol=tol,
            eps=eps,
            max_steps=max_steps,
            method=method,
            verbose=verbose,
            regularise=regularise,
            full_return=full_return,
            linsearch=linsearch,
        )

        self._set_solved_problem(sol)

    def _set_solved_problem_cm(self, solution):
        if self.full_return:
            self.r_xy = solution[0]
            self.comput_time = solution[1]
            self.n_steps = solution[2]
            self.norm_seq = solution[3]
            self.diff_seq = solution[4]
            self.alfa_seq = solution[5]
        else:
            self.r_xy = solution

        self.r_x = self.r_xy
        if self.last_model == "cm":
            self.x = self.r_x[self.r_invert_dseq]
        elif self.last_model == "cm_exp":
            self.x = np.exp(-self.r_x[self.r_invert_dseq])

    def _set_solved_problem(self, solution):
        model = self.last_model
        if model in ["cm", "cm_exp"]:
            self._set_solved_problem_cm(solution)
        elif model in ["ecm", "ecm_exp"]:
            self._set_solved_problem_ecm(solution)
        elif model in ["crema", "crema-sparse"]:
            self._set_solved_problem_crema_undirected(solution)

    def degree_reduction(self):
        """
        Carries out degree reduction for UBCM.
        The graph should be initialized.
        """
        self.r_dseq, self.r_index_dseq, self.r_invert_dseq, self.r_multiplicity = np.unique(
            self.dseq,
            return_index=True,
            return_inverse=True,
            return_counts=True,
            axis=0,
        )

        self.rnz_n = self.r_dseq.size

        self.is_reduced = True

    def _set_initial_guess(self, model):

        if model in ["cm", "cm_exp"]:
            self._set_initial_guess_cm()
        elif model in ["ecm", "ecm_exp"]:
            self._set_initial_guess_ecm()
        elif model in ["crema", "crema-sparse"]:
            self._set_initial_guess_crema_undirected()

    def _set_initial_guess_cm(self):

        if ~self.is_reduced:
            self.degree_reduction()

        if isinstance(self.initial_guess, np.ndarray):
            self.r_x = self.initial_guess[self.r_index_dseq]
        elif isinstance(self.initial_guess, str):
            if self.initial_guess == "degrees_minor":
                self.r_x = self.r_dseq / (
                    np.sqrt(self.n_edges) + 1
                )  # This +1 increases the stability of the solutions.
            elif self.initial_guess == "random":
                self.r_x = np.random.rand(self.rnz_n).astype(np.float64)
            elif self.initial_guess == "uniform":
                self.r_x = 0.5 * np.ones(
                    self.rnz_n, dtype=np.float64
                )  # All probabilities will be 1/2 initially
            elif self.initial_guess == "degrees":
                self.r_x = self.r_dseq.astype(np.float64)
            elif self.initial_guess == "chung_lu":
                self.r_x = self.r_dseq.astype(np.float64)/(2*self.n_edges)
            else:
                raise ValueError(
                    '{} is not an available initial guess'.format(
                        self.initial_guess
                        )
                    )
        else:
            raise TypeError('initial_guess must be str or numpy.ndarray')

        self.r_x[self.r_dseq == 0] = 0

        if isinstance(self.initial_guess, str):
            if self.last_model == "cm":
                self.x0 = self.r_x
            elif self.last_model == "cm_exp":
                self.r_x[self.r_x != 0] = -np.log(self.r_x[self.r_x != 0])
                self.x0 = self.r_x
        elif isinstance(self.initial_guess, np.ndarray):
            self.x0 = self.r_x

    def _set_initial_guess_crema_undirected(self):

        if isinstance(self.initial_guess, np.ndarray):
            self.beta = self.initial_guess
        elif isinstance(self.initial_guess, str):
            if self.initial_guess == "strengths":
                self.beta = (self.strength_sequence > 0).astype(
                    float
                ) / self.strength_sequence.sum()
            elif self.initial_guess == "strengths_minor":
                self.beta = (self.strength_sequence > 0).astype(float) / (
                    self.strength_sequence + 1
                )
            elif self.initial_guess == "random":
                self.beta = np.random.rand(self.n_nodes).astype(np.float64)
            else:
                raise ValueError(
                    '{} is not an available initial guess'.format(
                        self.initial_guess
                        )
                    )
        else:
            raise TypeError('initial_guess must be str or numpy.ndarray')

        self.beta[self.strength_sequence == 0] = 0

        self.x0 = self.beta

    def _set_initial_guess_ecm(self):

        if isinstance(self.initial_guess, np.ndarray):
            self.x = self.initial_guess[:self.n_nodes]
            self.y = self.initial_guess[self.n_nodes:]
        elif isinstance(self.initial_guess, str):
            if self.initial_guess == "strengths":
                self.x = self.dseq.astype(float) / (
                    self.n_edges + 1
                )  # This +1 increases the stability of the solutions.
                self.y = (
                    self.strength_sequence.astype(float)
                    / self.strength_sequence.sum()
                )
            elif self.initial_guess == "strengths_minor":
                self.x = np.ones_like(self.dseq, dtype=np.float64) / (
                    self.dseq + 1
                )
                self.y = np.ones_like(self.strength_sequence, dtype=np.float64) / (
                    self.strength_sequence + 1
                )
            elif self.initial_guess == "random":
                self.x = np.random.rand(self.n_nodes).astype(np.float64)
                self.y = np.random.rand(self.n_nodes).astype(np.float64)
            elif self.initial_guess == "uniform":
                self.x = 0.001 * np.ones(self.n_nodes, dtype=np.float64)
                self.y = 0.001 * np.ones(self.n_nodes, dtype=np.float64)
            else:
                raise ValueError(
                    '{} is not an available initial guess'.format(
                        self.initial_guess
                        )
                    )
        else:
            raise TypeError('initial_guess must be str or numpy.ndarray')

        self.x[self.dseq == 0] = 0
        self.y[self.strength_sequence == 0] = 0

        if isinstance(self.initial_guess, str):
            if self.last_model == "ecm":
                self.x0 = np.concatenate((self.x, self.y))
            elif self.last_model == "ecm_exp":
                self.x[self.x != 0] = -np.log(self.x[self.x != 0])
                self.y[self.y != 0] = -np.log(self.y[self.y != 0])
                self.x0 = np.concatenate((self.x, self.y))
        elif isinstance(self.initial_guess, np.ndarray):
            self.x0 = np.concatenate((self.x, self.y))

    # DA SISTEMARE
    def _solution_error(self):
        
        if self.last_model in ["cm", "cm_exp", "crema", "crema-sparse"]:
            if self.x is not None:
                ex_k = mof.expected_degree_cm(self.x)
                # print(k, ex_k)
                self.expected_dseq = ex_k
                # error output
                self.error_degree = np.linalg.norm(ex_k - self.dseq, ord=np.inf)
                self.relative_error_degree = np.linalg.norm((ex_k - self.dseq)/(self.dseq + np.exp(-100)), ord=np.inf)
                self.error = self.error_degree

            if self.beta is not None:
                if self.is_sparse:
                    ex_s = mof.expected_strength_crema_undirected_sparse(
                        self.beta, self.adjacency_crema
                    )
                else:
                    ex_s = mof.expected_strength_crema_undirected(
                        self.beta, self.adjacency_crema
                    )
                self.expected_stregth_seq = ex_s
                # error output
                self.error_strength = np.linalg.norm(
                    ex_s - self.strength_sequence, ord=np.inf
                )
                self.relative_error_strength = np.max(
                    abs(
                     (ex_s - self.strength_sequence) / (self.strength_sequence + np.exp(-100))
                    )
                )

                if self.adjacency_given:
                    self.error = self.error_strength
                else:
                    self.error = max(self.error_strength, self.error_degree)

        # potremmo strutturarlo così per evitare ridondanze
        elif self.last_model in ["ecm", "ecm_exp"]:
            sol = np.concatenate((self.x, self.y))
            ex = mof.expected_ecm(sol)
            k = np.concatenate((self.dseq, self.strength_sequence))
            self.expected_dseq = ex[: self.n_nodes]
            self.expected_strength_seq = ex[self.n_nodes:]

            # error output
            self.error_degree = np.linalg.norm(self.expected_dseq -
                                               self.dseq, ord=np.inf)
            self.error_strength = np.linalg.norm(self.expected_strength_seq -
                                                 self.strength_sequence,
                                                 ord=np.inf)
            self.relative_error_strength = max(
                abs(
                    (self.strength_sequence -
                     self.expected_strength_seq)/self.strength_sequence
                    )
                )

            self.relative_error_degree = max(
                abs(
                    (self.dseq - self.expected_dseq)/self.dseq
                    )
                )
            self.error = max(self.error_strength, self.error_degree)

    def _set_args(self, model):

        if model in ["crema", "crema-sparse"]:
            self.args = (
                self.strength_sequence,
                self.adjacency_crema,
                self.nz_index,
            )
        elif model in ["cm", "cm_exp"]:
            self.args = (self.r_dseq, self.r_multiplicity)
        elif model in ["ecm", "ecm_exp"]:
            self.args = (self.dseq, self.strength_sequence)

    def _initialize_problem(self, model, method):

        self._set_initial_guess(model)

        self._set_args(model)

        mod_met = "-"
        mod_met = mod_met.join([model, method])

        d_fun = {
            "cm-newton": lambda x: -mof.loglikelihood_prime_cm(x, self.args),
            "cm-quasinewton": lambda x: -mof.loglikelihood_prime_cm(x, self.args),
            "cm-fixed-point": lambda x: mof.iterative_cm(x, self.args),
            "crema-newton": lambda x: -mof.loglikelihood_prime_crema_undirected(
                x, self.args
            ),
            "crema-quasinewton": lambda x: -mof.loglikelihood_prime_crema_undirected(
                x, self.args
            ),
            "crema-fixed-point": lambda x: -mof.iterative_crema_undirected(x, self.args),
            "ecm-newton": lambda x: -mof.loglikelihood_prime_ecm(x, self.args),
            "ecm-quasinewton": lambda x: -mof.loglikelihood_prime_ecm(
                x, self.args
            ),
            "ecm-fixed-point": lambda x: mof.iterative_ecm(x, self.args),
            "crema-sparse-newton": lambda x: -mof.loglikelihood_prime_crema_undirected_sparse(
                x, self.args
            ),
            "crema-sparse-quasinewton": lambda x: -mof.loglikelihood_prime_crema_undirected_sparse(
                x, self.args
            ),
            "crema-sparse-fixed-point": lambda x: -mof.iterative_crema_undirected_sparse(
                x, self.args
            ),
            "cm_exp-newton": lambda x: -mof.loglikelihood_prime_cm_exp(
                x, self.args
            ),
            "cm_exp-quasinewton": lambda x: -mof.loglikelihood_prime_cm_exp(
                x, self.args
            ),
            "cm_exp-fixed-point": lambda x: mof.iterative_cm_exp(x, self.args),
            "ecm_exp-newton": lambda x: -mof.loglikelihood_prime_ecm_exp(
                x, self.args
            ),
            "ecm_exp-quasinewton": lambda x: -mof.loglikelihood_prime_ecm_exp(
                x, self.args
            ),
            "ecm_exp-fixed-point": lambda x: mof.iterative_ecm_exp(x, self.args),
        }

        d_fun_jac = {
            "cm-newton": lambda x: -mof.loglikelihood_hessian_cm(x, self.args),
            "cm-quasinewton": lambda x: -mof.loglikelihood_hessian_diag_cm(
                x, self.args
            ),
            "cm-fixed-point": None,
            "crema-newton": lambda x: -mof.loglikelihood_hessian_crema_undirected(
                x, self.args
            ),
            "crema-quasinewton": lambda x: -mof.loglikelihood_hessian_diag_crema_undirected(
                x, self.args
            ),
            "crema-fixed-point": None,
            "ecm-newton": lambda x: -mof.loglikelihood_hessian_ecm(x, self.args),
            "ecm-quasinewton": lambda x: -mof.loglikelihood_hessian_diag_ecm(
                x, self.args
            ),
            "ecm-fixed-point": None,
            "crema-sparse-newton": lambda x: -mof.loglikelihood_hessian_crema_undirected(
                x, self.args
            ),
            "crema-sparse-quasinewton": lambda x: -mof.loglikelihood_hessian_diag_crema_undirected_sparse(
                x, self.args
            ),
            "crema-sparse-fixed-point": None,
            "cm_exp-newton": lambda x: -mof.loglikelihood_hessian_cm_exp(
                x, self.args
            ),
            "cm_exp-quasinewton": lambda x: -mof.loglikelihood_hessian_diag_cm_exp(
                x, self.args
            ),
            "cm_exp-fixed-point": None,
            "ecm_exp-newton": lambda x: -mof.loglikelihood_hessian_ecm_exp(
                x, self.args
            ),
            "ecm_exp-quasinewton": lambda x: -mof.loglikelihood_hessian_diag_ecm_exp(
                x, self.args
            ),
            "ecm_exp-fixed-point": None,
        }

        d_fun_stop = {
            "cm-newton": lambda x: -mof.loglikelihood_cm(x, self.args),
            "cm-quasinewton": lambda x: -mof.loglikelihood_cm(x, self.args),
            "cm-fixed-point": lambda x: -mof.loglikelihood_cm(x, self.args),
            "crema-newton": lambda x: -mof.loglikelihood_crema_undirected(x, self.args),
            "crema-quasinewton": lambda x: -mof.loglikelihood_crema_undirected(
                x, self.args
            ),
            "crema-fixed-point": lambda x: -mof.loglikelihood_crema_undirected(
                x, self.args
            ),
            "ecm-newton": lambda x: -mof.loglikelihood_ecm(x, self.args),
            "ecm-quasinewton": lambda x: -mof.loglikelihood_ecm(x, self.args),
            "ecm-fixed-point": lambda x: -mof.loglikelihood_ecm(x, self.args),
            "crema-sparse-newton": lambda x: -mof.loglikelihood_crema_undirected_sparse(
                x, self.args
            ),
            "crema-sparse-quasinewton": lambda x: -mof.loglikelihood_crema_undirected_sparse(
                x, self.args
            ),
            "crema-sparse-fixed-point": lambda x: -mof.loglikelihood_crema_undirected_sparse(
                x, self.args
            ),
            "cm_exp-newton": lambda x: -mof.loglikelihood_cm_exp(x, self.args),
            "cm_exp-quasinewton": lambda x: -mof.loglikelihood_cm_exp(
                x, self.args
            ),
            "cm_exp-fixed-point": lambda x: -mof.loglikelihood_cm_exp(
                x, self.args
            ),
            "ecm_exp-newton": lambda x: -mof.loglikelihood_ecm_exp(x, self.args),
            "ecm_exp-quasinewton": lambda x: -mof.loglikelihood_ecm_exp(
                x, self.args
            ),
            "ecm_exp-fixed-point": lambda x: -mof.loglikelihood_ecm_exp(
                x, self.args
            ),
        }
        try:
            self.fun = d_fun[mod_met]
            self.fun_jac = d_fun_jac[mod_met]
            self.step_fun = d_fun_stop[mod_met]
        except:
            raise ValueError(
                'Method must be "newton","quasi-newton", or "fixed-point".'
            )

        d_pmatrix = {
            "cm": mof.pmatrix_cm,
            "cm_exp": mof.pmatrix_cm,
        }

        if model in ["cm", "cm_exp"]:
            self.args_p = (self.n_nodes, np.nonzero(self.dseq)[0])
            self.fun_pmatrix = lambda x: d_pmatrix[model](x, self.args_p)

        args_lin = {
            "cm": (mof.loglikelihood_cm, self.args),
            "crema": (mof.loglikelihood_crema_undirected, self.args),
            "crema-sparse": (mof.loglikelihood_crema_undirected_sparse, self.args),
            "ecm": (mof.loglikelihood_ecm, self.args),
            "cm_exp": (mof.loglikelihood_cm_exp, self.args),
            "ecm_exp": (mof.loglikelihood_ecm_exp, self.args),
        }

        self.args_lins = args_lin[model]

        lins_fun = {
            "cm-newton": lambda x: mof.linsearch_fun_CM(x, self.args_lins),
            "cm-quasinewton": lambda x: mof.linsearch_fun_CM(x, self.args_lins),
            "cm-fixed-point": lambda x: mof.linsearch_fun_CM_fixed(x),
            "crema-newton": lambda x: mof.linsearch_fun_crema_undirected(x, self.args_lins),
            "crema-quasinewton": lambda x: mof.linsearch_fun_crema_undirected(x, self.args_lins),
            "crema-fixed-point": lambda x: mof.linsearch_fun_crema_undirected_fixed(x),
            "crema-sparse-newton": lambda x: mof.linsearch_fun_crema_undirected(x, self.args_lins),
            "crema-sparse-quasinewton": lambda x: mof.linsearch_fun_crema_undirected(x, self.args_lins),
            "crema-sparse-fixed-point": lambda x: mof.linsearch_fun_crema_undirected_fixed(x),
            "ecm-newton": lambda x: mof.linsearch_fun_ECM(x, self.args_lins),
            "ecm-quasinewton": lambda x: mof.linsearch_fun_ECM(x, self.args_lins),
            "ecm-fixed-point": lambda x: mof.linsearch_fun_ECM_fixed(x),
            "cm_exp-newton": lambda x: mof.linsearch_fun_CM_exp(x, self.args_lins),
            "cm_exp-quasinewton": lambda x: mof.linsearch_fun_CM_exp(x, self.args_lins),
            "cm_exp-fixed-point": lambda x: mof.linsearch_fun_CM_exp_fixed(x),
            "ecm_exp-newton": lambda x: mof.linsearch_fun_ECM_exp(x, self.args_lins),
            "ecm_exp-quasinewton": lambda x: mof.linsearch_fun_ECM_exp(x, self.args_lins),
            "ecm_exp-fixed-point": lambda x: mof.linsearch_fun_ECM_exp_fixed(x)
        }

        self.fun_linsearch = lins_fun[mod_met]

        hess_reg = {
            "cm": sof.matrix_regulariser_function_eigen_based,
            "cm_exp": sof.matrix_regulariser_function,
            "ecm": sof.matrix_regulariser_function_eigen_based,
            "ecm_exp": sof.matrix_regulariser_function,
            "crema": sof.matrix_regulariser_function,
            "crema-sparse": sof.matrix_regulariser_function,
        }

        self.hessian_regulariser = hess_reg[model]

        if isinstance(self.regularise, str):
            if self.regularise == "eigenvalues":
                self.hessian_regulariser = sof.matrix_regulariser_function_eigen_based
            elif self.regularise == "identity":
                self.hessian_regulariser = hessian_regulariser_function

    def _solve_problem_crema_undirected(
        self,
        initial_guess=None,
        model="crema",
        adjacency="cm",
        method="quasinewton",
        method_adjacency="newton",
        initial_guess_adjacency="random",
        max_steps=100,
        tol=1e-8,
        eps=1e-8,
        full_return=False,
        verbose=False,
        linsearch=True,
        regularise=True,
    ):
        if model == "crema-sparse":
            self.is_sparse = True
        else:
            self.is_sparse = False
        if not isinstance(adjacency, (list, np.ndarray, str)) and (
            not scipy.sparse.isspmatrix(adjacency)
        ):
            raise ValueError("adjacency must be a matrix or a method")
        elif isinstance(adjacency, str):

            # aggiungere check sul modello passato per l'adjacency matrix

            self._solve_problem(
                initial_guess=initial_guess_adjacency,
                model=adjacency,
                method=method_adjacency,
                max_steps=max_steps,
                tol=tol,
                eps=eps,
                full_return=full_return,
                verbose=verbose,
                linsearch=linsearch,
                regularise=regularise
            )

            if self.is_sparse:
                self.adjacency_crema = (self.x,)
                self.adjacency_given = False
            else:
                pmatrix = self.fun_pmatrix(self.x)
                raw_ind, col_ind = np.nonzero(np.triu(pmatrix))
                raw_ind = raw_ind.astype(np.int64)
                col_ind = col_ind.astype(np.int64)
                weigths_value = pmatrix[raw_ind, col_ind]
                self.adjacency_crema = (raw_ind, col_ind, weigths_value)
                self.is_sparse = False
                self.adjacency_given = False
        elif isinstance(adjacency, list):
            adjacency = np.array(adjacency).astype(np.float64)
            raw_ind, col_ind = np.nonzero(np.triu(adjacency))
            raw_ind = raw_ind.astype(np.int64)
            col_ind = col_ind.astype(np.int64)
            weigths_value = adjacency[raw_ind, col_ind]
            self.adjacency_crema = (raw_ind, col_ind, weigths_value)
            self.is_sparse = False
            self.adjacency_given = True
        elif isinstance(adjacency, np.ndarray):
            adjacency = adjacency.astype(np.float64)
            raw_ind, col_ind = np.nonzero(np.triu(adjacency))
            raw_ind = raw_ind.astype(np.int64)
            col_ind = col_ind.astype(np.int64)
            weigths_value = adjacency[raw_ind, col_ind]
            self.adjacency_crema = (raw_ind, col_ind, weigths_value)
            self.is_sparse = False
            self.adjacency_given = True
        elif scipy.sparse.isspmatrix(adjacency):
            raw_ind, col_ind = scipy.sparse.triu(adjacency).nonzero()
            raw_ind = raw_ind.astype(np.int64)
            col_ind = col_ind.astype(np.int64)
            weigths_value = (adjacency[raw_ind, col_ind].A1).astype(np.float64)
            self.adjacency_crema = (raw_ind, col_ind, weigths_value)
            self.is_sparse = False
            self.adjacency_given = True

        if self.is_sparse:
            self.last_model = "crema-sparse"
        else:
            self.last_model = model
            linsearch = linsearch
            regularise = regularise

        self.regularise = regularise
        self.full_return = full_return
        self.initial_guess = initial_guess
        self._initialize_problem(self.last_model, method)
        x0 = self.x0

        sol = sof.solver(
            x0,
            fun=self.fun,
            fun_jac=self.fun_jac,
            step_fun=self.step_fun,
            linsearch_fun=self.fun_linsearch,
            hessian_regulariser=self.hessian_regulariser,
            tol=tol,
            eps=eps,
            max_steps=max_steps,
            method=method,
            verbose=verbose,
            regularise=regularise,
            linsearch=linsearch,
            full_return=full_return,
        )

        self._set_solved_problem(sol)

    def _set_solved_problem_crema_undirected(self, solution):
        if self.full_return:
            self.beta = solution[0]
            self.comput_time_crema = solution[1]
            self.n_steps_crema = solution[2]
            self.norm_seq_crema = solution[3]
            self.diff_seq_crema = solution[4]
            self.alfa_seq_crema = solution[5]
        else:
            self.beta = solution

    def _set_solved_problem_ecm(self, solution):
        if self.full_return:
            self.r_xy = solution[0]
            self.comput_time = solution[1]
            self.n_steps = solution[2]
            self.norm_seq = solution[3]
            self.diff_seq = solution[4]
            self.alfa_seq = solution[5]
        else:
            self.r_xy = solution

        if self.last_model == "ecm":
            self.x = self.r_xy[: self.n_nodes]
            self.y = self.r_xy[self.n_nodes:]
        elif self.last_model == "ecm_exp":
            self.x = np.exp(-self.r_xy[:self.n_nodes])
            self.y = np.exp(-self.r_xy[self.n_nodes:])

    def solve_tool(
        self,
        model,
        method,
        initial_guess=None,
        adjacency="cm_exp",
        method_adjacency="newton",
        initial_guess_adjacency="random",
        max_steps=100,
        full_return=False,
        verbose=False,
        linsearch=True,
        tol=1e-8,
        eps=1e-8,
    ):
        """The function solves the ERGM optimization problem from
        a range of available models. The user can choose among three
        optimization methods.
        The graph should be initialized.

        :param model: Available models are:

            - *cm*: solves UBCM respect to the parameters *x* of the mof.loglikelihood function, it works for uweighted undirected graphs [insert ref].
            - *cm_exp*: differently from the *cm* option, *cm_exp* considers the exponents of *x* as parameters [insert ref].
            - *ecm*: solves UECM respect to the parameters *x* and *y* of the mof.loglikelihood function, it is conceived for weighted undirected graphs [insert ref].
            - *ecm_exp*: differently from the *ecm* option, *ecm_exp* considers the exponents of *x* and *y* as parameters [insert ref].
            - *crema*: solves CReMa for a weighted undirectd graphs. In order to compute beta parameters, it requires information about the binary structure of the network. These can be provided by the user by using *adjacency* paramenter.
            - *crema-sparse*: alternative implementetio of *crema* for large graphs. The *creama-sparse* model doesn't compute the binary probability matrix avoing memory problems for large graphs.

        :type model: str
        :param method: Available methods to solve the given *model* are:

            - *newton*: uses Newton-Rhapson method to solve the selected model, it can be memory demanding for *crema* because it requires the computation of the entire Hessian matrix. This method is not available for *creama-sparse*.
            - *quasinewton*: uses Newton-Rhapson method with Hessian matrix approximated by its principal diagonal to find parameters maximising mof.loglikelihood function.
            - *fixed-point*: uses a fixed-point method to find parameters maximising mof.loglikelihood function.

        :type method: str
        :param initial_guess: Starting point solution may affect the results of the optization process. The user can provid an initial guess or choose between the following options:

            - **Binary Models**:
                - *random*: random numbers in (0, 1);
                - *uniform*: uniform initial guess in (0, 1);
                - *degrees*: initial guess of each node is proportianal to its degree;
                - *degrees_minor*: initial guess of each node is inversely proportional to its degree;
                - *chung_lu*: initial guess given by Chung-Lu formula;
            - **Weighted Models**:
                - *random*: random numbers in (0, 1);
                - *uniform*: uniform initial guess in (0, 1);
                - *strengths*: initial guess of each node is proportianal to its stength;
                - *strengths_minor*: initial guess of each node is inversely proportional to its strength;
        :type initial_guess: str, optional
        :param adjacency: Adjacency can be a binary method (defaults is *cm_exp*) or an adjacency matrix.
        :type adjacency: str or numpy.ndarray, optional
        :param method_adjacency: If adjacency is a *model*, it is the *methdod* used to solve it. Defaults to "newton".
        :type method_adjacency: str, optional
        :param initial_guess_adjacency: If adjacency is a *model*, it is the chosen initial guess. Defaults to "random".
        :type initial_guess_adjacency: str, optional
        :param max_steps: maximum number of iteration, defaults to 100.
        :type max_steps: int, optional
        :param full_return: If True the algorithm returns more statistics than the obtained solution, defaults to False.
        :type full_return: bool, optional
        :param verbose: If True the algorithm prints a bunch of statistics at each step, defaults to False.
        :type verbose: bool, optional
        :param linsearch: If True the linsearch function is active, defaults to True.
        :type linsearch: bool, optional
        :param tol: parameter controlling the tolerance of the norm the gradient function, defaults to 1e-8.
        :type tol: float, optional
        :param eps: parameter controlling the tolerance of the difference between two iterations, defaults to 1e-8.
        :type eps: float, optional
        """
        # TODO: aggiungere tutti i metodi
        if model in ["cm", "cm_exp", "ecm", "ecm_exp"]:
            self._solve_problem(
                initial_guess=initial_guess,
                model=model,
                method=method,
                max_steps=max_steps,
                full_return=full_return,
                verbose=verbose,
                linsearch=linsearch,
                tol=tol,
                eps=eps,
            )
        elif model in ["crema", "crema-sparse"]:
            self._solve_problem_crema_undirected(
                initial_guess=initial_guess,
                model=model,
                adjacency=adjacency,
                method=method,
                method_adjacency=method_adjacency,
                initial_guess_adjacency=initial_guess_adjacency,
                max_steps=max_steps,
                full_return=full_return,
                verbose=verbose,
                linsearch=linsearch,
                tol=tol,
                eps=eps,
            )
        self._solution_error()
        if verbose:
            print("\nmin eig = {}".format(self.error))

    def ensemble_sampler(self, n, cpu_n=1, output_dir="sample/", seed=42):
        """The function sample a given number of graphs in the ensemble
        generated from the last model solved. Each grpah is an edgelist
        written in the output directory as `.txt` file.
        The function is parallelised and can run on multiple cpus.

        :param n: Number of graphs to sample.
        :type n: int
        :param cpu_n: Number of cpus to use, defaults to 1.
        :type cpu_n: int, optional
        :param output_dir: Name of the output directory, defaults to "sample/".
        :type output_dir: str, optional
        :param seed: Random seed, defaults to 42.
        :type seed: int, optional
        :raises ValueError: [description]
        """
        # al momento funziona solo sull'ultimo problema risolto

        # create the output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # compute the sample

        # seed specification
        np.random.seed(seed)
        s = [np.random.randint(0, 1000000) for i in range(n)]

        if self.last_model in ["cm", "cm_exp"]:
            iter_files = iter(
                output_dir + "{}.txt".format(i) for i in range(n))
            i = 0
            for item in iter_files:
                eg.ensemble_sampler_cm_graph(
                    outfile_name=item,
                    x=self.x,
                    cpu_n=cpu_n,
                    seed=s[i])
                i += 1

        elif self.last_model in ["ecm", "ecm_exp"]:
            iter_files = iter(
                output_dir + "{}.txt".format(i) for i in range(n))
            i = 0
            for item in iter_files:
                eg.ensemble_sampler_ecm_graph(
                    outfile_name=item,
                    x=self.x,
                    y=self.y,
                    cpu_n=cpu_n,
                    seed=s[i])
                i += 1

        elif self.last_model in ["crema"]:
            if self.adjacency_given:
                # deterministic adj matrix
                iter_files = iter(
                    output_dir + "{}.txt".format(i) for i in range(n))
                i = 0
                for item in iter_files:
                    eg.ensemble_sampler_crema_ecm_det_graph(
                        outfile_name=item,
                        beta=self.beta,
                        adj=self.adjacency_crema,
                        cpu_n=cpu_n,
                        seed=s[i])
                    i += 1
            else:
                # probabilistic adj matrix
                iter_files = iter(
                    output_dir + "{}.txt".format(i) for i in range(n))
                i = 0
                for item in iter_files:
                    eg.ensemble_sampler_crema_ecm_prob_graph(
                        outfile_name=item,
                        beta=self.beta,
                        adj=self.adjacency_crema,
                        cpu_n=cpu_n,
                        seed=s[i])
                    i += 1
        elif self.last_model in ["crema-sparse"]:
            if not self.adjacency_given:
                # probabilistic adj matrix
                iter_files = iter(
                    output_dir + "{}.txt".format(i) for i in range(n))
                i = 0
                for item in iter_files:
                    eg.ensemble_sampler_crema_sparse_ecm_prob_graph(
                        outfile_name=item,
                        beta=self.beta,
                        adj=self.adjacency_crema,
                        cpu_n=cpu_n,
                        seed=s[i])
                    i += 1

        else:
            raise ValueError("insert a model")


class DirectedGraph:
    """Directed graph instance can be initialised with
    adjacency matrix, edgelist, degree sequence or strengths sequence.

    :param adjacency: Adjacency matrix, defaults to None.
    :type adjacency: numpy.ndarray, list, scipy.sparse_matrix, optional
    :param edgelist: edgelist, defaults to None.
    :type edgelist: numpy.ndarray, list, optional
    :param degree_sequence: degrees sequence, defaults to None.
    :type degree_sequence: numpy.ndarray, optional
    :param strength_sequence: strengths sequence, defaults to None.
    :type strength_sequence: numpy.ndarray, optional
    """
    def __init__(
        self,
        adjacency=None,
        edgelist=None,
        degree_sequence=None,
        strength_sequence=None,
    ):

        self.n_nodes = None
        self.n_edges = None
        self.adjacency = None
        self.is_sparse = False
        self.edgelist = None
        self.dseq = None
        self.dseq_out = None
        self.dseq_in = None
        self.out_strength = None
        self.in_strength = None
        self.nz_index_sout = None
        self.nz_index_sin = None
        self.nodes_dict = None
        self.is_initialized = False
        self.is_randomized = False
        self.is_weighted = False
        self._initialize_graph(
            adjacency=adjacency,
            edgelist=edgelist,
            degree_sequence=degree_sequence,
            strength_sequence=strength_sequence,
        )
        self.avg_mat = None

        self.initial_guess = None
        # Reduced problem parameters
        self.is_reduced = False
        self.r_dseq = None
        self.r_dseq_out = None
        self.r_dseq_in = None
        self.r_n_out = None
        self.r_n_in = None
        self.r_invert_dseq = None
        self.r_invert_dseq_out = None
        self.r_invert_dseq_in = None
        self.r_dim = None
        self.r_multiplicity = None
        # Problem solutions
        self.x = None
        self.y = None
        self.xy = None
        self.b_out = None
        self.b_in = None
        # reduced solutions
        self.r_x = None
        self.r_y = None
        self.r_xy = None
        # Problem (reduced) residuals
        self.residuals = None
        self.final_result = None
        self.r_beta = None
        # non-zero indices
        self.nz_index_out = None
        self.rnz_dseq_out = None
        self.nz_index_in = None
        self.rnz_dseq_in = None
        # model
        self.x0 = None
        self.error = None
        self.error_degree = None
        self.relative_error_degree = None
        self.error_strength = None
        self.relative_error_strength = None
        self.full_return = False
        self.last_model = None
        # functen
        self.args = None

    def _initialize_graph(
        self,
        adjacency=None,
        edgelist=None,
        degree_sequence=None,
        strength_sequence=None,
    ):

        if adjacency is not None:
            if not isinstance(
                adjacency, (list, np.ndarray)
            ) and not scipy.sparse.isspmatrix(adjacency):
                raise TypeError(
                    ("The adjacency matrix must be passed as a list or numpy"
                     " array or scipy sparse matrix.")
                )
            elif adjacency.size > 0:
                if np.sum(adjacency < 0):
                    raise TypeError(
                        "The adjacency matrix entries must be positive."
                    )
                if isinstance(
                    adjacency, list
                ):
                    # Cast it to a numpy array: if it is given as a list
                    # it should not be too large
                    self.adjacency = np.array(adjacency)
                elif isinstance(adjacency, np.ndarray):
                    self.adjacency = adjacency
                else:
                    self.adjacency = adjacency
                    self.is_sparse = True
                if np.sum(adjacency) == np.sum(adjacency > 0):
                    self.dseq_in = nef.in_degree(adjacency)
                    self.dseq_out = nef.out_degree(adjacency)
                else:
                    self.dseq_in = nef.in_degree(adjacency)
                    self.dseq_out = nef.out_degree(adjacency)
                    self.in_strength = nef.in_strength(adjacency).astype(
                        np.float64
                    )
                    self.out_strength = nef.out_strength(adjacency).astype(
                        np.float64
                    )
                    self.nz_index_sout = np.nonzero(self.out_strength)[0]
                    self.nz_index_sin = np.nonzero(self.in_strength)[0]
                    self.is_weighted = True

                self.n_nodes = len(self.dseq_out)
                self.n_edges = np.sum(self.dseq_out)
                self.is_initialized = True

        elif edgelist is not None:
            if not isinstance(edgelist, (list, np.ndarray)):
                raise TypeError(
                    "The edgelist must be passed as a list or numpy array."
                )
            elif len(edgelist) > 0:
                if len(edgelist[0]) > 3:
                    raise ValueError(
                        ("This is not an edgelist. An edgelist must be a list"
                         " or array of couples of nodes with optional weights."
                         " Is this an adjacency matrix?")
                    )
                elif len(edgelist[0]) == 2:
                    (
                        self.edgelist,
                        self.dseq_out,
                        self.dseq_in,
                        self.nodes_dict,
                    ) = nef.edgelist_from_edgelist_directed(edgelist)
                else:
                    (
                        self.edgelist,
                        self.dseq_out,
                        self.dseq_in,
                        self.out_strength,
                        self.in_strength,
                        self.nodes_dict,
                    ) = nef.edgelist_from_edgelist_directed(edgelist)
                self.n_nodes = len(self.dseq_out)
                self.n_edges = np.sum(self.dseq_out)
                self.is_initialized = True

        elif degree_sequence is not None:
            if not isinstance(degree_sequence, (list, np.ndarray)):
                raise TypeError(
                    ("The degree sequence must be passed as a list"
                     " or numpy array.")
                )
            elif len(degree_sequence) > 0:
                try:
                    int(degree_sequence[0])
                except:  # TODO: bare exception
                    raise TypeError(
                        "The degree sequence must contain numeric values."
                    )
                if (np.array(degree_sequence) < 0).sum() > 0:
                    raise ValueError("A degree cannot be negative.")
                else:
                    if len(degree_sequence) % 2 != 0:
                        raise ValueError(
                            "Strength-in/out arrays must have same length."
                        )
                    self.n_nodes = int(len(degree_sequence) / 2)
                    self.dseq_out = degree_sequence[: self.n_nodes]
                    self.dseq_in = degree_sequence[self.n_nodes:]
                    self.n_edges = np.sum(self.dseq_out)
                    self.is_initialized = True

                if strength_sequence is not None:
                    if not isinstance(strength_sequence, (list, np.ndarray)):
                        raise TypeError(
                            ("The strength sequence must be passed as a"
                             " list or numpy array.")
                        )
                    elif len(strength_sequence):
                        try:
                            int(strength_sequence[0])
                        except:  # TODO: bare exception to check
                            raise TypeError(
                                ("The strength sequence must contain"
                                 " numeric values.")
                            )
                        if (np.array(strength_sequence) < 0).sum() > 0:
                            raise ValueError("A strength cannot be negative.")
                        else:
                            if len(strength_sequence) % 2 != 0:
                                raise ValueError(
                                    ("Strength-in/out arrays must have"
                                     " same length.")
                                )
                            self.n_nodes = int(len(strength_sequence) / 2)
                            self.out_strength = strength_sequence[
                                : self.n_nodes
                            ]
                            self.in_strength = strength_sequence[
                                self.n_nodes:
                            ]
                            self.nz_index_sout = np.nonzero(self.out_strength)[
                                0
                            ]
                            self.nz_index_sin = np.nonzero(self.in_strength)[0]
                            self.is_weighted = True
                            self.is_initialized = True

        elif strength_sequence is not None:
            if not isinstance(strength_sequence, (list, np.ndarray)):
                raise TypeError(
                    ("The strength sequence must be passed as a list or"
                     " numpy array.")
                )
            elif len(strength_sequence):
                try:
                    int(strength_sequence[0])
                except:  # TODO: bare exception
                    raise TypeError(
                        "The strength sequence must contain numeric values."
                    )
                if (np.array(strength_sequence) < 0).sum() > 0:
                    raise ValueError("A strength cannot be negative.")
                else:
                    if len(strength_sequence) % 2 != 0:
                        raise ValueError(
                            "Strength-in/out arrays must have same length."
                        )
                    self.n_nodes = int(len(strength_sequence) / 2)
                    self.out_strength = strength_sequence[: self.n_nodes]
                    self.in_strength = strength_sequence[self.n_nodes:]
                    self.nz_index_sout = np.nonzero(self.out_strength)[0]
                    self.nz_index_sin = np.nonzero(self.in_strength)[0]
                    self.is_weighted = True
                    self.is_initialized = True

    def set_adjacency_matrix(self, adjacency):
        """Initializes a graph from the adjacency matrix.

        :param adjacency: Adjacency matrix.
        :type adjacency: numpy.ndarray, list, scipy.sparse_matrix
        """
        if self.is_initialized:
            print(
                ("Graph already contains edges or has a degree sequence."
                 " Use clean_edges() first.")
            )
        else:
            self._initialize_graph(adjacency=adjacency)

    def set_edgelist(self, edgelist):
        """Initializes a graph from the edgelist.

        :param adjacency: Edgelist
        :type adjacency: numpy.ndarray, list
        """
        if self.is_initialized:
            print(
                ("Graph already contains edges or has a degree sequence."
                 " Use clean_edges() first.")
            )
        else:
            self._initialize_graph(edgelist=edgelist)

    def set_degree_sequences(self, degree_sequence):
        """Initializes graph from the degrees sequence.

        :param adjacency: Degrees sequence
        :type adjacency: numpy.ndarray
        """
        if self.is_initialized:
            print(
                ("Graph already contains edges or has a degree sequence."
                 " Use clean_edges() first.")
            )
        else:
            self._initialize_graph(degree_sequence=degree_sequence)

    def clean_edges(self):
        """Deletes all initialized graph attributes
        """
        self.adjacency = None
        self.edgelist = None
        self.deg_seq = None
        self.is_initialized = False

    def _solve_problem(
        self,
        initial_guess=None,  # TODO:aggiungere un default a initial guess
        model="dcm",
        method="quasinewton",
        max_steps=100,
        tol=1e-8,
        eps=1e-8,
        full_return=False,
        verbose=False,
        linsearch=True,
        regularise=True,
        regularise_eps=1e-3,
    ):

        self.last_model = model
        self.full_return = full_return
        self.initial_guess = initial_guess
        self.regularise = regularise
        self._initialize_problem(model, method)
        x0 = self.x0

        sol = sof.solver(
            x0,
            fun=self.fun,
            fun_jac=self.fun_jac,
            step_fun=self.step_fun,
            linsearch_fun=self.fun_linsearch,
            hessian_regulariser=self.hessian_regulariser,
            tol=tol,
            eps=eps,
            max_steps=max_steps,
            method=method,
            verbose=verbose,
            regularise=self.regularise,
            regularise_eps=regularise_eps,
            full_return=full_return,
            linsearch=linsearch,
        )

        self._set_solved_problem(sol)

    def _set_solved_problem_dcm(self, solution):
        if self.full_return:
            self.r_xy = solution[0]
            self.comput_time = solution[1]
            self.n_steps = solution[2]
            self.norm_seq = solution[3]
            self.diff_seq = solution[4]
            self.alfa_seq = solution[5]
        else:
            self.r_xy = solution

        self.r_x = self.r_xy[: self.rnz_n_out]
        self.r_y = self.r_xy[self.rnz_n_out:]

        self.x = self.r_x[self.r_invert_dseq]
        self.y = self.r_y[self.r_invert_dseq]

    def _set_solved_problem_dcm_exp(self, solution):
        if self.full_return:
            # conversion from theta to x
            self.r_xy = np.exp(-solution[0])
            self.comput_time = solution[1]
            self.n_steps = solution[2]
            self.norm_seq = solution[3]
            self.diff_seq = solution[4]
            self.alfa_seq = solution[5]
        else:
            # conversion from theta to x
            self.r_xy = np.exp(-solution)

        self.r_x = self.r_xy[: self.rnz_n_out]
        self.r_y = self.r_xy[self.rnz_n_out:]

        self.x = self.r_x[self.r_invert_dseq]
        self.y = self.r_y[self.r_invert_dseq]

    def _set_solved_problem_decm(self, solution):
        if self.full_return:
            self.r_xy = solution[0]
            self.comput_time = solution[1]
            self.n_steps = solution[2]
            self.norm_seq = solution[3]
            self.diff_seq = solution[4]
            self.alfa_seq = solution[5]
        else:
            self.r_xy = solution

        self.x = self.r_xy[: self.n_nodes]
        self.y = self.r_xy[self.n_nodes: 2 * self.n_nodes]
        self.b_out = self.r_xy[2 * self.n_nodes: 3 * self.n_nodes]
        self.b_in = self.r_xy[3 * self.n_nodes:]

    def _set_solved_problem_decm_exp(self, solution):
        if self.full_return:
            # conversion from theta to x
            self.r_xy = np.exp(-solution[0])
            self.comput_time = solution[1]
            self.n_steps = solution[2]
            self.norm_seq = solution[3]
            self.diff_seq = solution[4]
            self.alfa_seq = solution[5]
        else:
            # conversion from theta to x
            self.r_xy = np.exp(-solution)

        self.x = self.r_xy[: self.n_nodes]
        self.y = self.r_xy[self.n_nodes: 2 * self.n_nodes]
        self.b_out = self.r_xy[2 * self.n_nodes: 3 * self.n_nodes]
        self.b_in = self.r_xy[3 * self.n_nodes:]

    def _set_solved_problem(self, solution):
        model = self.last_model
        if model in ["dcm"]:
            self._set_solved_problem_dcm(solution)
        if model in ["dcm_exp"]:
            self._set_solved_problem_dcm_exp(solution)
        elif model in ["decm"]:
            self._set_solved_problem_decm(solution)
        elif model in ["decm_exp"]:
            self._set_solved_problem_decm_exp(solution)
        elif model in ["crema", "crema-sparse"]:
            self._set_solved_problem_crema_directed(solution)

    def degree_reduction(self):
        """Carries out degree reduction for DBCM.
        The graph should be initialized.
        """
        self.dseq = np.array(list(zip(self.dseq_out, self.dseq_in)))
        (
            self.r_dseq,
            self.r_index_dseq,
            self.r_invert_dseq,
            self.r_multiplicity
        ) = np.unique(
            self.dseq,
            return_index=True,
            return_inverse=True,
            return_counts=True,
            axis=0,
        )

        self.rnz_dseq_out = self.r_dseq[:, 0]
        self.rnz_dseq_in = self.r_dseq[:, 1]

        self.nz_index_out = np.nonzero(self.rnz_dseq_out)[0]
        self.nz_index_in = np.nonzero(self.rnz_dseq_in)[0]

        self.rnz_n_out = self.rnz_dseq_out.size
        self.rnz_n_in = self.rnz_dseq_in.size
        self.rnz_dim = self.rnz_n_out + self.rnz_n_in

        self.is_reduced = True

    def _set_initial_guess(self, model):
        if model in ["dcm"]:
            self._set_initial_guess_dcm()
        if model in ["dcm_exp"]:
            self._set_initial_guess_dcm_exp()
        elif model in ["decm"]:
            self._set_initial_guess_decm()
        elif model in ["decm_exp"]:
            self._set_initial_guess_decm_exp()
        elif model in ["crema", "crema-sparse"]:
            self._set_initial_guess_crema_directed()

    def _set_initial_guess_dcm(self):
        # The preselected initial guess works best usually.
        # The suggestion is, if this does not work,
        # trying with random initial conditions several times.
        # If you want to customize the initial guess,
        # remember that the code starts with a reduced number
        # of rows and columns.
        # remember if you insert your choice as initial choice,
        # it should be numpy.ndarray
        if ~self.is_reduced:
            self.degree_reduction()

        if isinstance(self.initial_guess, np.ndarray):
            # we reduce the full x0, it's not very honest
            # but it's better to ask to provide an already reduced x0
            self.r_x = self.initial_guess[:self.n_nodes][self.r_index_dseq]
            self.r_y = self.initial_guess[self.n_nodes:][self.r_index_dseq]
        elif isinstance(self.initial_guess, str):
            if self.initial_guess == 'degrees_minor':
                # This +1 increases the stability of the solutions.
                self.r_x = self.rnz_dseq_out / (np.sqrt(self.n_edges) + 1)
                self.r_y = self.rnz_dseq_in / (np.sqrt(self.n_edges) + 1)
            elif self.initial_guess == "random":
                self.r_x = np.random.rand(self.rnz_n_out).astype(np.float64)
                self.r_y = np.random.rand(self.rnz_n_in).astype(np.float64)
            elif self.initial_guess == "uniform":
                # All probabilities will be 1/2 initially
                self.r_x = 0.5 * np.ones(self.rnz_n_out, dtype=np.float64)
                self.r_y = 0.5 * np.ones(self.rnz_n_in, dtype=np.float64)
            elif self.initial_guess == "degrees":
                self.r_x = self.rnz_dseq_out.astype(np.float64)
                self.r_y = self.rnz_dseq_in.astype(np.float64)
            elif self.initial_guess == "chung_lu":
                self.r_x = self.rnz_dseq_out.astype(np.float64) / \
                    (2*self.n_edges)
                self.r_y = self.rnz_dseq_in.astype(np.float64)/(2*self.n_edges)
            else:
                raise ValueError(
                    '{} is not an available initial guess'.format(
                        self.initial_guess
                        )
                    )
        else:
            raise TypeError('initial_guess must be str or numpy.ndarray')

        self.r_x[self.rnz_dseq_out == 0] = 0
        self.r_y[self.rnz_dseq_in == 0] = 0

        self.x0 = np.concatenate((self.r_x, self.r_y))

    def _set_initial_guess_dcm_exp(self):
        # The preselected initial guess works best usually.
        # The suggestion is, if this does not work,
        # trying with random initial conditions several times.
        # If you want to customize the initial guess, remember that the code
        # starts with a reduced number of rows and columns.

        if ~self.is_reduced:
            self.degree_reduction()

        if isinstance(self.initial_guess, np.ndarray):
            # we reduce the full x0, it's not very honest
            # but it's better to ask to provide an already reduced x0
            self.r_x = self.initial_guess[:self.n_nodes][self.r_index_dseq]
            self.r_y = self.initial_guess[self.n_nodes:][self.r_index_dseq]
        elif isinstance(self.initial_guess, str):
            if self.initial_guess == 'degrees_minor':
                self.r_x = self.rnz_dseq_out / (
                    np.sqrt(self.n_edges) + 1
                )  # This +1 increases the stability of the solutions.
                self.r_y = self.rnz_dseq_in / (np.sqrt(self.n_edges) + 1)
            elif self.initial_guess == "random":
                self.r_x = np.random.rand(self.rnz_n_out).astype(np.float64)
                self.r_y = np.random.rand(self.rnz_n_in).astype(np.float64)
            elif self.initial_guess == "uniform":
                self.r_x = 0.5 * np.ones(
                    self.rnz_n_out, dtype=np.float64
                )  # All probabilities will be 1/2 initially
                self.r_y = 0.5 * np.ones(self.rnz_n_in, dtype=np.float64)
            elif self.initial_guess == "degrees":
                self.r_x = self.rnz_dseq_out.astype(np.float64)
                self.r_y = self.rnz_dseq_in.astype(np.float64)
            elif self.initial_guess == "chung_lu":
                self.r_x = self.rnz_dseq_out.astype(np.float64) / \
                    (2*self.n_edges)
                self.r_y = self.rnz_dseq_in.astype(np.float64) / \
                    (2*self.n_edges)
            else:
                raise ValueError(
                    '{} is not an available initial guess'.format(
                        self.initial_guess
                        )
                    )
        else:
            raise TypeError('initial_guess must be str or numpy.ndarray')

        not_zero_ind_x = self.r_x != 0
        self.r_x[not_zero_ind_x] = -np.log(self.r_x[not_zero_ind_x])
        self.r_x[self.rnz_dseq_out == 0] = 1e3
        not_zero_ind_y = self.r_y != 0
        self.r_y[not_zero_ind_y] = -np.log(self.r_y[not_zero_ind_y])
        self.r_y[self.rnz_dseq_in == 0] = 1e3

        self.x0 = np.concatenate((self.r_x, self.r_y))

    def _set_initial_guess_crema_directed(self):
        # The preselected initial guess works best usually.
        # The suggestion is, if this does not work,
        # trying with random initial conditions several times.
        # If you want to customize the initial guess, remember that
        # the code starts with a reduced number of rows and columns.
        # TODO: mettere un self.is_weighted bool
        if isinstance(self.initial_guess, np.ndarray):
            self.b_out = self.initial_guess[:self.n_nodes]
            self.b_in = self.initial_guess[self.n_nodes:]
        elif isinstance(self.initial_guess, str):
            if self.initial_guess == "strengths":
                self.b_out = (self.out_strength > 0).astype(
                    float
                ) / self.out_strength.sum()
                self.b_in = (self.in_strength > 0).astype(
                    float
                ) / self.in_strength.sum()
            elif self.initial_guess == "strengths_minor":
                # This +1 increases the stability of the solutions.
                self.b_out = (self.out_strength > 0).astype(float) / (
                    self.out_strength + 1
                )
                self.b_in = (self.in_strength > 0).astype(float) / (
                    self.in_strength + 1
                )
            elif self.initial_guess == "random":
                self.b_out = np.random.rand(self.n_nodes).astype(np.float64)
                self.b_in = np.random.rand(self.n_nodes).astype(np.float64)
            else:
                raise ValueError(
                    '{} is not an available initial guess'.format(
                        self.initial_guess
                        )
                    )
        else:
            raise TypeError('initial_guess must be str or numpy.ndarray')

        self.b_out[self.out_strength == 0] = 0
        self.b_in[self.in_strength == 0] = 0

        self.x0 = np.concatenate((self.b_out, self.b_in))

    def _set_initial_guess_decm(self):
        # The preselected initial guess works best usually.
        # The suggestion is, if this does not work,
        # trying with random initial conditions several times.
        # If you want to customize the initial guess,
        # remember that the code starts with a reduced number
        # of rows and columns.
        if isinstance(self.initial_guess, np.ndarray):
            self.x = self.initial_guess[:self.n_nodes]
            self.y = self.initial_guess[self.n_nodes:2*self.n_nodes]
            self.b_out = self.initial_guess[2*self.n_nodes:3*self.n_nodes]
            self.b_in = self.initial_guess[3*self.n_nodes:]
        elif isinstance(self.initial_guess, str):
            if self.initial_guess == 'strengths':
                self.x = self.dseq_out.astype(float) / (self.n_edges + 1)
                self.y = self.dseq_in.astype(float) / (self.n_edges + 1)
                self.b_out = (
                    self.out_strength.astype(float) / self.out_strength.sum()
                )  # This +1 increases the stability of the solutions.
                self.b_in = self.in_strength.astype(float) /\
                    self.in_strength.sum()
            elif self.initial_guess == "strengths_minor":
                self.x = np.ones_like(self.dseq_out) / (self.dseq_out + 1)
                self.y = np.ones_like(self.dseq_in) / (self.dseq_in + 1)
                self.b_out = np.ones_like(self.out_strength) / (
                    self.out_strength + 1
                )
                self.b_in = np.ones_like(self.in_strength) /\
                    (self.in_strength + 1)
            elif self.initial_guess == "random":
                self.x = np.random.rand(self.n_nodes).astype(np.float64)
                self.y = np.random.rand(self.n_nodes).astype(np.float64)
                self.b_out = np.random.rand(self.n_nodes).astype(np.float64)
                self.b_in = np.random.rand(self.n_nodes).astype(np.float64)
            elif self.initial_guess == "uniform":
                # All probabilities will be 0.9 initially
                self.x = 0.9 * np.ones(self.n_nodes, dtype=np.float64)
                self.y = 0.9 * np.ones(self.n_nodes, dtype=np.float64)
                self.b_out = 0.9 * np.ones(self.n_nodes, dtype=np.float64)
                self.b_in = 0.9 * np.ones(self.n_nodes, dtype=np.float64)
            else:
                raise ValueError(
                    '{} is not an available initial guess'.format(
                        self.initial_guess
                        )
                    )
        else:
            raise TypeError('initial_guess must be str or numpy.ndarray')

        self.x[self.dseq_out == 0] = 0
        self.y[self.dseq_in == 0] = 0
        self.b_out[self.out_strength == 0] = 0
        self.b_in[self.in_strength == 0] = 0

        self.x0 = np.concatenate((self.x, self.y, self.b_out, self.b_in))

    def _set_initial_guess_decm_exp(self):
        # The preselected initial guess works best usually.
        # The suggestion is, if this does not work,
        #  trying with random initial conditions several times.
        # If you want to customize the initial guess, remember that
        # the code starts with a reduced number of rows and columns.
        if isinstance(self.initial_guess, np.ndarray):
            self.x = self.initial_guess[:self.n_nodes]
            self.y = self.initial_guess[self.n_nodes:2*self.n_nodes]
            self.b_out = self.initial_guess[2*self.n_nodes:3*self.n_nodes]
            self.b_in = self.initial_guess[3*self.n_nodes:]
        elif isinstance(self.initial_guess, str):
            if self.initial_guess == "strengths":
                self.x = self.dseq_out.astype(float) / (self.n_edges + 1)
                self.y = self.dseq_in.astype(float) / (self.n_edges + 1)
                self.b_out = (
                    self.out_strength.astype(float) / self.out_strength.sum()
                )  # This +1 increases the stability of the solutions.
                self.b_in = self.in_strength.astype(float) /\
                    self.in_strength.sum()
            elif self.initial_guess == "strengths_minor":
                self.x = np.ones_like(self.dseq_out) / (self.dseq_out + 1)
                self.y = np.ones_like(self.dseq_in) / (self.dseq_in + 1)
                self.b_out = np.ones_like(self.out_strength) / (
                    self.out_strength + 1
                )
                self.b_in = np.ones_like(self.in_strength) /\
                    (self.in_strength + 1)
            elif self.initial_guess == "random":
                self.x = np.random.rand(self.n_nodes).astype(np.float64)
                self.y = np.random.rand(self.n_nodes).astype(np.float64)
                self.b_out = np.random.rand(self.n_nodes).astype(np.float64)
                self.b_in = np.random.rand(self.n_nodes).astype(np.float64)
            elif self.initial_guess == "uniform":
                self.x = 0.1 * np.ones(
                    self.n_nodes, dtype=np.float64
                )  # All probabilities will be 1/2 initially
                self.y = 0.1 * np.ones(self.n_nodes, dtype=np.float64)
                self.b_out = 0.1 * np.ones(self.n_nodes, dtype=np.float64)
                self.b_in = 0.1 * np.ones(self.n_nodes, dtype=np.float64)
            else:
                raise ValueError(
                    '{} is not an available initial guess'.format(
                        self.initial_guess
                        )
                    )
        else:
            raise TypeError('initial_guess must be str or numpy.ndarray')

        not_zero_ind_x = self.x != 0
        self.x[not_zero_ind_x] = -np.log(self.x[not_zero_ind_x])

        not_zero_ind_y = self.y != 0
        self.y[not_zero_ind_y] = -np.log(self.y[not_zero_ind_y])

        not_zero_ind_b_out = self.b_out != 0
        self.b_out[not_zero_ind_b_out] = -np.log(
            self.b_out[not_zero_ind_b_out])

        not_zero_ind_b_in = self.b_in != 0
        self.b_in[not_zero_ind_b_in] = -np.log(self.b_in[not_zero_ind_b_in])

        self.x[self.dseq_out == 0] = 1e3
        self.y[self.dseq_in == 0] = 1e3
        self.b_out[self.out_strength == 0] = 1e3
        self.b_in[self.in_strength == 0] = 1e3

        self.x0 = np.concatenate((self.x, self.y, self.b_out, self.b_in))

    def _solution_error(self):

        if self.last_model in ["dcm_exp", "dcm", "crema", "crema-sparse"]:
            if (self.x is not None) and (self.y is not None):
                sol = np.concatenate((self.x, self.y))
                ex_k_out = mof.expected_out_degree_dcm(sol)
                ex_k_in = mof.expected_in_degree_dcm(sol)
                ex_k = np.concatenate((ex_k_out, ex_k_in))
                k = np.concatenate((self.dseq_out, self.dseq_in))
                # print(k, ex_k)
                self.expected_dseq = ex_k
                # error output
                self.error_degree = np.linalg.norm(ex_k - k, ord=np.inf)
                self.relative_error_degree = np.linalg.norm(
                    (ex_k - k) / (k + + np.exp(-100)),
                    ord=np.inf
                    )
                self.error = self.error_degree

            if (self.b_out is not None) and (self.b_in is not None):
                sol = np.concatenate([self.b_out, self.b_in])
                if self.is_sparse:
                    ex_s_out = mof.expected_out_strength_crema_directed_sparse(
                        sol, self.adjacency_crema
                    )
                    ex_s_in = mof.expected_in_stregth_crema_directed_sparse(
                        sol, self.adjacency_crema
                    )
                else:
                    ex_s_out = mof.expected_out_strength_crema_directed(
                        sol, self.adjacency_crema
                    )
                    ex_s_in = mof.expected_in_stregth_crema_directed(
                        sol, self.adjacency_crema
                    )
                ex_s = np.concatenate([ex_s_out, ex_s_in])
                s = np.concatenate([self.out_strength, self.in_strength])
                self.expected_stregth_seq = ex_s
                # error output
                self.error_strength = np.linalg.norm(ex_s - s, ord=np.inf)
                self.relative_error_strength = np.max(
                    abs(
                        (ex_s - s) / (s + np.exp(-100))
                    )
                )
                if self.adjacency_given:
                    self.error = self.error_strength
                else:
                    self.error = max(self.error_strength, self.error_degree)

        # potremmo strutturarlo così per evitare ridondanze
        elif self.last_model in ["decm", "decm_exp"]:
            sol = np.concatenate((self.x, self.y, self.b_out, self.b_in))
            ex = mof.expected_decm(sol)
            k = np.concatenate(
                (
                    self.dseq_out,
                    self.dseq_in,
                    self.out_strength,
                    self.in_strength,
                )
            )
            self.expected_dseq = ex[: 2 * self.n_nodes]

            self.expected_strength_seq = ex[2 * self.n_nodes:]

            # error putput
            self.error_degree = max(
                abs(
                    (
                        np.concatenate((self.dseq_out, self.dseq_in))
                        - self.expected_dseq
                    )
                )
            )
            self.error_strength = max(
                abs(
                    np.concatenate((self.out_strength, self.in_strength))
                    - self.expected_strength_seq
                )
            )
            self.relative_error_strength = max(
                abs(
                    (np.concatenate((self.out_strength, self.in_strength))
                     - self.expected_strength_seq)
                    / np.concatenate((self.out_strength, self.in_strength)
                                     + np.exp(-100))
                )
            )
            self.relative_error_degree = max(
                abs(
                    (np.concatenate((self.dseq_out, self.dseq_in))
                     - self.expected_dseq)
                    / np.concatenate((self.dseq_out, self.dseq_in)
                                     + np.exp(-100))
                 )
            )
            self.error = np.linalg.norm(ex - k, ord=np.inf)

    def _set_args(self, model):
        if model in ["crema", "crema-sparse"]:
            self.args = (
                self.out_strength,
                self.in_strength,
                self.adjacency_crema,
                self.nz_index_sout,
                self.nz_index_sin,
            )
        elif model in ["dcm", "dcm_exp"]:
            self.args = (
                self.rnz_dseq_out,
                self.rnz_dseq_in,
                self.nz_index_out,
                self.nz_index_in,
                self.r_multiplicity,
            )
        elif model in ["decm", "decm_exp"]:
            self.args = (
                self.dseq_out,
                self.dseq_in,
                self.out_strength,
                self.in_strength,
            )

    def _initialize_problem(self, model, method):

        self._set_initial_guess(model)

        self._set_args(model)

        mod_met = "-"
        mod_met = mod_met.join([model, method])

        d_fun = {
            "dcm-newton": lambda x: -mof.loglikelihood_prime_dcm(x, self.args),
            "dcm-quasinewton": lambda x: -mof.loglikelihood_prime_dcm(
                x,
                self.args
            ),
            "dcm-fixed-point": lambda x: mof.iterative_dcm(x, self.args),
            "dcm_exp-newton": lambda x: -mof.loglikelihood_prime_dcm_exp(
                x,
                self.args
            ),
            "dcm_exp-quasinewton": lambda x: -mof.loglikelihood_prime_dcm_exp(
                x,
                self.args
            ),
            "dcm_exp-fixed-point": lambda x: mof.iterative_dcm_exp(x, self.args),
            "crema-newton": lambda x: -mof.loglikelihood_prime_crema_directed(
                x,
                self.args
            ),
            "crema-quasinewton": lambda x: -mof.loglikelihood_prime_crema_directed(
                x,
                self.args
            ),
            "crema-fixed-point": lambda x: -mof.iterative_crema_directed(x, self.args),
            "decm-newton": lambda x: -mof.loglikelihood_prime_decm(x, self.args),
            "decm-quasinewton": lambda x: -mof.loglikelihood_prime_decm(
                x,
                self.args
            ),
            "decm-fixed-point": lambda x: mof.iterative_decm(x, self.args),
            "decm_exp-newton": lambda x: -mof.loglikelihood_prime_decm_exp(
                x,
                self.args
            ),
            "decm_exp-quasinewton": lambda x: -mof.loglikelihood_prime_decm_exp(
                x,
                self.args
            ),
            "decm_exp-fixed-point": lambda x: mof.iterative_decm_exp(x, self.args),
            "crema-sparse-newton": lambda x: -mof.loglikelihood_prime_crema_directed_sparse(
                x,
                self.args
            ),
            "crema-sparse-quasinewton": lambda x:
                -mof.loglikelihood_prime_crema_directed_sparse(
                    x,
                    self.args
                ),
            "crema-sparse-fixed-point": lambda x: -mof.iterative_crema_directed_sparse(
                x,
                self.args
            ),
        }

        d_fun_jac = {
            "dcm-newton": lambda x: -mof.loglikelihood_hessian_dcm(x, self.args),
            "dcm-quasinewton": lambda x: -mof.loglikelihood_hessian_diag_dcm(
                x,
                self.args
            ),
            "dcm-fixed-point": None,
            "dcm_exp-newton": lambda x: -mof.loglikelihood_hessian_dcm_exp(
                x,
                self.args
            ),
            "dcm_exp-quasinewton": lambda x:
                -mof.loglikelihood_hessian_diag_dcm_exp(
                    x,
                    self.args
                ),
            "dcm_exp-fixed-point": None,
            "crema-newton": lambda x: -mof.loglikelihood_hessian_crema_directed(
                x,
                self.args
            ),
            "crema-quasinewton": lambda x: -mof.loglikelihood_hessian_diag_crema_directed(
                x,
                self.args
            ),
            "crema-fixed-point": None,
            "decm-newton": lambda x: -mof.loglikelihood_hessian_decm(x, self.args),
            "decm-quasinewton": lambda x: -mof.loglikelihood_hessian_diag_decm(
                x,
                self.args
            ),
            "decm-fixed-point": None,
            "decm_exp-newton": lambda x: -mof.loglikelihood_hessian_decm_exp(
                x,
                self.args
            ),
            "decm_exp-quasinewton": lambda x:
                -mof.loglikelihood_hessian_diag_decm_exp(
                    x,
                    self.args
                ),
            "decm_exp-fixed-point": None,
            "crema-sparse-newton": lambda x: -mof.loglikelihood_hessian_crema_directed(
                x,
                self.args
            ),
            "crema-sparse-quasinewton": lambda x:
                -mof.loglikelihood_hessian_diag_crema_directed_sparse(
                    x,
                    self.args
                ),
            "crema-sparse-fixed-point": None,
        }

        d_fun_step = {
            "dcm-newton": lambda x: -mof.loglikelihood_dcm(x, self.args),
            "dcm-quasinewton": lambda x: -mof.loglikelihood_dcm(x, self.args),
            "dcm-fixed-point": lambda x: -mof.loglikelihood_dcm(x, self.args),
            "dcm_exp-newton": lambda x: -mof.loglikelihood_dcm_exp(x, self.args),
            "dcm_exp-quasinewton": lambda x: -mof.loglikelihood_dcm_exp(
                x,
                self.args
            ),
            "dcm_exp-fixed-point": lambda x: -mof.loglikelihood_dcm_exp(
                x,
                self.args
            ),
            "crema-newton": lambda x: -mof.loglikelihood_crema_directed(x, self.args),
            "crema-quasinewton": lambda x: -mof.loglikelihood_crema_directed(
                x,
                self.args
            ),
            "crema-fixed-point": lambda x: -mof.loglikelihood_crema_directed(
                x,
                self.args
            ),
            "decm-newton": lambda x: -mof.loglikelihood_decm(x, self.args),
            "decm-quasinewton": lambda x: -mof.loglikelihood_decm(x, self.args),
            "decm-fixed-point": lambda x: -mof.loglikelihood_decm(x, self.args),
            "decm_exp-newton": lambda x: -mof.loglikelihood_decm_exp(x, self.args),
            "decm_exp-quasinewton": lambda x: -mof.loglikelihood_decm_exp(
                x,
                self.args
            ),
            "decm_exp-fixed-point": lambda x: -mof.loglikelihood_decm_exp(
                x,
                self.args
            ),
            "crema-sparse-newton": lambda x: -mof.loglikelihood_crema_directed_sparse(
                x,
                self.args
            ),
            "crema-sparse-quasinewton": lambda x: -mof.loglikelihood_crema_directed_sparse(
                x,
                self.args
            ),
            "crema-sparse-fixed-point": lambda x: -mof.loglikelihood_crema_directed_sparse(
                x,
                self.args
            ),
        }

        try:
            self.fun = d_fun[mod_met]
            self.fun_jac = d_fun_jac[mod_met]
            self.step_fun = d_fun_step[mod_met]
        except:  # TODO: remove bare excpets
            raise ValueError(
                'Method must be "newton","quasinewton", or "fixed-point".'
            )

        # TODO: mancano metodi
        d_pmatrix = {"dcm": mof.pmatrix_dcm, "dcm_exp": mof.pmatrix_dcm}

        # Così basta aggiungere il decm e funziona tutto
        if model in ["dcm", "dcm_exp"]:
            self.args_p = (
                self.n_nodes,
                np.nonzero(self.dseq_out)[0],
                np.nonzero(self.dseq_in)[0],
            )
            self.fun_pmatrix = lambda x: d_pmatrix[model](x, self.args_p)

        args_lin = {
            "dcm": (mof.loglikelihood_dcm, self.args),
            "crema": (mof.loglikelihood_crema_directed, self.args),
            "crema-sparse": (mof.loglikelihood_crema_directed_sparse, self.args),
            "decm": (mof.loglikelihood_decm, self.args),
            "dcm_exp": (mof.loglikelihood_dcm_exp, self.args),
            "decm_exp": (mof.loglikelihood_decm_exp, self.args),
        }

        self.args_lins = args_lin[model]

        lins_fun = {
            "dcm-newton": lambda x: mof.linsearch_fun_DCM(x, self.args_lins),
            "dcm-quasinewton": lambda x: mof.linsearch_fun_DCM(x, self.args_lins),
            "dcm-fixed-point": lambda x: mof.linsearch_fun_DCM_fixed(x),
            "dcm_exp-newton": lambda x: mof.linsearch_fun_DCM_exp(
                x,
                self.args_lins),
            "dcm_exp-quasinewton": lambda x: mof.linsearch_fun_DCM_exp(
                x,
                self.args_lins),
            "dcm_exp-fixed-point": lambda x: mof.linsearch_fun_DCM_exp_fixed(x),
            "crema-newton": lambda x: mof.linsearch_fun_crema_directed(x, self.args_lins),
            "crema-quasinewton": lambda x: mof.linsearch_fun_crema_directed(
                x,
                self.args_lins),
            "crema-fixed-point": lambda x: mof.linsearch_fun_crema_directed_fixed(x),
            "crema-sparse-newton": lambda x: mof.linsearch_fun_crema_directed(
                x,
                self.args_lins),
            "crema-sparse-quasinewton": lambda x: mof.linsearch_fun_crema_directed(
                x,
                self.args_lins),
            "crema-sparse-fixed-point": lambda x: mof.linsearch_fun_crema_directed_fixed(
                x),
            "decm-newton": lambda x: mof.linsearch_fun_DECM(
                x,
                self.args_lins),
            "decm-quasinewton": lambda x: mof.linsearch_fun_DECM(
                x,
                self.args_lins),
            "decm-fixed-point": lambda x: mof.linsearch_fun_DECM_fixed(x),
            "decm_exp-newton": lambda x: mof.linsearch_fun_DECM_exp(
                x,
                self.args_lins),
            "decm_exp-quasinewton": lambda x: mof.linsearch_fun_DECM_exp(
                x,
                self.args_lins),
            "decm_exp-fixed-point": lambda x: mof.linsearch_fun_DECM_exp_fixed(x),
        }

        self.fun_linsearch = lins_fun[mod_met]

        hess_reg = {
            "dcm": sof.matrix_regulariser_function_eigen_based,
            "dcm_exp": sof.matrix_regulariser_function,
            "decm": sof.matrix_regulariser_function_eigen_based,
            "decm_exp": sof.matrix_regulariser_function,
            "crema": sof.matrix_regulariser_function,
            "crema-sparse": sof.matrix_regulariser_function,
        }

        self.hessian_regulariser = hess_reg[model]

        if isinstance(self.regularise, str):
            if self.regularise == "eigenvalues":
                self.hessian_regulariser = \
                    sof.matrix_regulariser_function_eigen_based
            elif self.regularise == "identity":
                self.hessian_regulariser = sof.matrix_regulariser_function

    def _solve_problem_crema_directed(
        self,
        initial_guess=None,
        model="crema",
        adjacency="dcm",
        method="quasinewton",
        method_adjacency="newton",
        initial_guess_adjacency="random",
        max_steps=100,
        tol=1e-8,
        eps=1e-8,
        full_return=False,
        verbose=False,
        linsearch=True,
        regularise=True,
        regularise_eps=1e-3,
    ):
        if model == "crema-sparse":
            self.is_sparse = True
        else:
            self.is_sparse = False

        if not isinstance(adjacency, (list, np.ndarray, str)) and (
            not scipy.sparse.isspmatrix(adjacency)
        ):
            raise ValueError("adjacency must be a matrix or a method")
        elif isinstance(adjacency, str):
            self._solve_problem(
                initial_guess=initial_guess_adjacency,
                model=adjacency,
                method=method_adjacency,
                max_steps=max_steps,
                tol=tol,
                eps=eps,
                full_return=full_return,
                verbose=verbose,
            )
            if self.is_sparse:
                self.adjacency_crema = (self.x, self.y)
                self.adjacency_given = False
            else:
                pmatrix = self.fun_pmatrix(np.concatenate([self.x, self.y]))
                raw_ind, col_ind = np.nonzero(pmatrix)
                raw_ind = raw_ind.astype(np.int64)
                col_ind = col_ind.astype(np.int64)
                weigths_value = pmatrix[raw_ind, col_ind]
                self.adjacency_crema = (raw_ind, col_ind, weigths_value)
                self.is_sparse = False
                self.adjacency_given = False
        elif isinstance(adjacency, list):
            adjacency = np.array(adjacency).astype(np.float64)
            raw_ind, col_ind = np.nonzero(adjacency)
            raw_ind = raw_ind.astype(np.int64)
            col_ind = col_ind.astype(np.int64)
            weigths_value = adjacency[raw_ind, col_ind]
            self.adjacency_crema = (raw_ind, col_ind, weigths_value)
            self.is_sparse = False
            self.adjacency_given = True
        elif isinstance(adjacency, np.ndarray):
            adjacency = adjacency.astype(np.float64)
            raw_ind, col_ind = np.nonzero(adjacency)
            raw_ind = raw_ind.astype(np.int64)
            col_ind = col_ind.astype(np.int64)
            weigths_value = adjacency[raw_ind, col_ind]
            self.adjacency_crema = (raw_ind, col_ind, weigths_value)
            self.is_sparse = False
            self.adjacency_given = True
        elif scipy.sparse.isspmatrix(adjacency):
            raw_ind, col_ind = adjacency.nonzero()
            raw_ind = raw_ind.astype(np.int64)
            col_ind = col_ind.astype(np.int64)
            weigths_value = (adjacency[raw_ind, col_ind].A1).astype(np.float64)
            self.adjacency_crema = (raw_ind, col_ind, weigths_value)
            self.is_sparse = False
            self.adjacency_given = True

        if self.is_sparse:
            self.last_model = "crema-sparse"
        else:
            self.last_model = model

        self.regularise = regularise
        self.full_return = full_return
        self.initial_guess = initial_guess
        self._initialize_problem(self.last_model, method)
        x0 = self.x0

        sol = sof.solver(
            x0,
            fun=self.fun,
            fun_jac=self.fun_jac,
            step_fun=self.step_fun,
            linsearch_fun=self.fun_linsearch,
            hessian_regulariser=self.hessian_regulariser,
            tol=tol,
            eps=eps,
            max_steps=max_steps,
            method=method,
            verbose=verbose,
            regularise=regularise,
            regularise_eps=regularise_eps,
            linsearch=linsearch,
            full_return=full_return,
        )

        self._set_solved_problem_crema_directed(sol)

    def _set_solved_problem_crema_directed(self, solution):
        if self.full_return:
            self.b_out = solution[0][: self.n_nodes]
            self.b_in = solution[0][self.n_nodes:]
            self.comput_time_crema = solution[1]
            self.n_steps_crema = solution[2]
            self.norm_seq_crema = solution[3]
            self.diff_seq_crema = solution[4]
            self.alfa_seq_crema = solution[5]
        else:
            self.b_out = solution[: self.n_nodes]
            self.b_in = solution[self.n_nodes:]

    def solve_tool(
        self,
        model,
        method,
        initial_guess=None,
        adjacency=None,
        method_adjacency='newton',
        initial_guess_adjacency="random",
        max_steps=100,
        full_return=False,
        verbose=False,
        linsearch=True,
        tol=1e-8,
        eps=1e-8,
    ):
        """The function solves the ERGM optimization problem from
        a range of available models. The user can choose among three
        optimization methods.
        The graph should be initialized.

        :param model: Available models are:
            - *dcm*: solves DBCM respect to the parameters *x* and "y" of the loglikelihood function, it works for uweighted directed graphs [insert ref].
            - *dcm_exp*: differently from the *dcm* option, *dcm_exp* considers the exponents of *x* and *y* as parameters [insert ref].
            - *decm*: solves DECM respect to the parameters *a_out*, *a_in*, *b_out* and *b_in* of the loglikelihood function, it is conceived for weighted directed graphs [insert ref].
            - *decm_exp*: differently from the *decm* option, *decm_exp* considers the exponents of *a_out*, *a_in*, *b_out* and *b_in** as parameters [insert ref].
            - *crema*: solves CReMa for a weighted directd graphs. In order to compute beta parameters, it requires information about the binary structure of the network. These can be provided by the user by using *adjacency* paramenter.
            - *crema-sparse*: alternative implementetio of *crema* for large graphs. The *creama-sparse* model doesn't compute the binary probability matrix avoing memory problems for large graphs.
        :type model: str
        :param method: Available methods to solve the given *model* are:
            - *newton*: uses Newton-Rhapson method to solve the selected model, it can be memory demanding for *crema* because it requires the computation of the entire Hessian matrix. This method is not available for *creama-sparse*.
            - *quasinewton*: uses Newton-Rhapson method with Hessian matrix approximated by its principal diagonal to find parameters maximising loglikelihood function.
            - *fixed-point*: uses a fixed-point method to find parameters maximising loglikelihood function.
        :type method: str
        :param initial_guess: Starting point solution may affect the results of the optization process. The user can provid an initial guess or choose between the following options:

            - **Binary Models**:

                - *random*: random numbers in (0, 1);
                - *uniform*: uniform initial guess in (0, 1);
                - *degrees*: initial guess of each node is proportianal to its degree;
                - *degrees_minor*: initial guess of each node is inversely proportional to its degree;
                - *chung_lu*: initial guess given by Chung-Lu formula;

            - **Weighted Models**:

                - *random*: random numbers in (0, 1);
                - *uniform*: uniform initial guess in (0, 1);
                - *strengths*: initial guess of each node is proportianal to its stength;
                - *strengths_minor*: initial guess of each node is inversely proportional to its strength;

        :type initial_guess: str, optional
        :param adjacency: Adjacency can be a binary method (defaults is *dcm_exp*) or an adjacency matrix.
        :type adjacency: str or numpy.ndarray, optional
        :param method_adjacency: If adjacency is a *model*, it is the *methdod* used to solve it. Defaults to "newton".
        :type method_adjacency: str, optional
        :param initial_guess_adjacency: If adjacency is a *model*, it is the chosen initial guess. Defaults to "random".
        :type initial_guess_adjacency: str, optional
        :param max_steps: maximum number of iteration, defaults to 100.
        :type max_steps: int, optional
        :param full_return: If True the algorithm returns more statistics than the obtained solution, defaults to False.
        :type full_return: bool, optional
        :param verbose: If True the algorithm prints a bunch of statistics at each step, defaults to False.
        :type verbose: bool, optional
        :param linsearch: If True the linsearch function is active, defaults to True.
        :type linsearch: bool, optional
        :param tol: parameter controlling the tollerance of the norm the gradient function, defaults to 1e-8.
        :type tol: float, optional
        :param eps: parameter controlling the tollerance of the difference between two iterations, defaults to 1e-8.
        :type eps: float, optional
        """
        # TODO: aggiungere tutti i metodi
        if model in ["dcm", "dcm_exp", "decm", "decm_exp"]:
            self._solve_problem(
                initial_guess=initial_guess,
                model=model,
                method=method,
                max_steps=max_steps,
                full_return=full_return,
                verbose=verbose,
                linsearch=linsearch,
                tol=tol,
                eps=eps,
            )
        elif model in ["crema", 'crema-sparse']:
            self._solve_problem_crema_directed(
                initial_guess=initial_guess,
                model=model,
                adjacency=adjacency,
                method=method,
                method_adjacency = method_adjacency,
                initial_guess_adjacency = initial_guess_adjacency,
                max_steps=max_steps,
                full_return=full_return,
                verbose=verbose,
                linsearch=linsearch,
                tol=tol,
                eps=eps,
            )
        self._solution_error()
        if verbose:
            print("\nmin eig = {}".format(self.error))

    def ensemble_sampler(self, n, cpu_n=1, output_dir="sample/", seed=42):
        """The function sample a given number of graphs in the ensemble
        generated from the last model solved. Each grpah is an edgelist
        written in the output directory as `.txt` file.
        The function is parallelised and can run on multiple cpus.

        :param n: Number of graphs to sample.
        :type n: int
        :param cpu_n: Number of cpus to use, defaults to 1.
        :type cpu_n: int, optional
        :param output_dir: Name of the output directory, defaults to "sample/".
        :type output_dir: str, optional
        :param seed: Random seed, defaults to 42.
        :type seed: int, optional
        :raises ValueError: [description]
        """
        # al momento funziona solo sull'ultimo problema risolto
        # unico input possibile e' la cartella dove salvare i samples
        # ed il numero di samples

        # create the output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # compute the sample

        # seed specification
        np.random.seed(seed)
        s = [np.random.randint(0, 1000000) for i in range(n)]

        if self.last_model in ["dcm", "dcm_exp"]:
            iter_files = iter(
                output_dir + "{}.txt".format(i) for i in range(n))
            i = 0
            for item in iter_files:
                eg.ensemble_sampler_dcm_graph(
                    outfile_name=item,
                    x=self.x,
                    y=self.y,
                    cpu_n=cpu_n,
                    seed=s[i])
                i += 1

        elif self.last_model in ["decm", "decm_exp"]:
            iter_files = iter(
                output_dir + "{}.txt".format(i) for i in range(n))
            i = 0
            for item in iter_files:
                eg.ensemble_sampler_decm_graph(
                    outfile_name=item,
                    a_out=self.x,
                    a_in=self.y,
                    b_out=self.b_out,
                    b_in=self.b_in,
                    cpu_n=cpu_n,
                    seed=s[i])
                i += 1

        elif self.last_model in ["crema"]:
            if self.adjacency_given:
                # deterministic adj matrix
                iter_files = iter(
                    output_dir + "{}.txt".format(i) for i in range(n))
                i = 0
                for item in iter_files:
                    eg.ensemble_sampler_crema_decm_det_graph(
                        outfile_name=item,
                        beta=(self.b_out, self.b_in),
                        adj=self.adjacency_crema,
                        cpu_n=cpu_n,
                        seed=s[i])
                    i += 1
            else:
                # probabilistic adj matrix
                iter_files = iter(
                    output_dir + "{}.txt".format(i) for i in range(n))
                i = 0
                for item in iter_files:
                    eg.ensemble_sampler_crema_decm_prob_graph(
                        outfile_name=item,
                        beta=(self.b_out, self.b_in),
                        adj=self.adjacency_crema,
                        cpu_n=cpu_n,
                        seed=s[i])
                    i += 1
        elif self.last_model in ["crema-sparse"]:
            if not self.adjacency_given:
                # probabilistic adj matrix
                iter_files = iter(
                    output_dir + "{}.txt".format(i) for i in range(n))
                i = 0
                for item in iter_files:
                    eg.ensemble_sampler_crema_sparse_decm_prob_graph(
                        outfile_name=item,
                        beta=(self.b_out, self.b_in),
                        adj=self.adjacency_crema,
                        cpu_n=cpu_n,
                        seed=s[i])
                    i += 1
        else:
            raise ValueError("insert a model")


class BipartiteGraph:
    """Bipartite Graph class for undirected binary bipartite networks.

    This class handles the bipartite graph object to compute the
    Bipartite Configuration Model (BiCM), which can be used as a null model
    for the analysis of undirected and binary bipartite networks.
    The class provides methods for calculating the probabilities and matrices
    of the null model and for projecting the bipartite network on its layers.
    The object can be initialized passing one of the parameters, or the nodes and
    edges can be passed later.

    :param biadjacency: binary input matrix describing the biadjacency matrix
            of a bipartite graph with the nodes of one layer along the rows
            and the nodes of the other layer along the columns.
    :type biadjacency: numpy.array, scipy.sparse, list, optional
    :param adjacency_list: dictionary that contains the adjacency list of nodes.
        The keys are considered as the nodes on the rows layer and the values,
        to be given as lists or numpy arrays, are considered as the nodes on the columns layer.
    :type adjacency_list: dict, optional
    :param edgelist: list of edges containing couples (row_node, col_node) of
        nodes forming an edge. each element in the couples must belong to
        the respective layer.
    :type edgelist: list, numpy.array, optional
    :param degree_sequences: couple of lists describing the degree sequences
        of both layers.
    :type degree_sequences: list, numpy.array, tuple, optional
    """

    def __init__(self, biadjacency=None, adjacency_list=None, edgelist=None, degree_sequences=None):
        self.n_rows = None
        self.n_cols = None
        self.r_n_rows = None
        self.r_n_cols = None
        self.shape = None
        self.n_edges = None
        self.r_n_edges = None
        self.n_nodes = None
        self.r_n_nodes = None
        self.biadjacency = None
        self.edgelist = None
        self.adj_list = None
        self.inv_adj_list = None
        self.rows_deg = None
        self.cols_deg = None
        self.r_rows_deg = None
        self.r_cols_deg = None
        self.rows_dict = None
        self.cols_dict = None
        self.is_initialized = False
        self.is_randomized = False
        self.is_reduced = False
        self.rows_projection = None
        self._initialize_graph(biadjacency=biadjacency, adjacency_list=adjacency_list, edgelist=edgelist,
                               degree_sequences=degree_sequences)
        self.avg_mat = None
        self.x = None
        self.y = None
        self.r_x = None
        self.r_y = None
        self.r_xy = None
        self.dict_x = None
        self.dict_y = None
        self.theta_x = None
        self.theta_y = None
        self.r_theta_x = None
        self.r_theta_y = None
        self.r_theta_xy = None
        self.projected_rows_adj_list = None
        self.projected_cols_adj_list = None
        self.v_adj_list = None
        self.projection_method = 'poisson'
        self.threads_num = 1
        self.progress_bar = True
        self.rows_pvals = None
        self.cols_pvals = None
        self.is_rows_projected = False
        self.is_cols_projected = False
        self.initial_guess = None
        self.method = None
        self.rows_multiplicity = None
        self.cols_multiplicity = None
        self.r_invert_rows_deg = None
        self.r_invert_cols_deg = None
        self.r_dim = None
        self.verbose = False
        self.full_return = False
        self.linsearch = True
        self.regularise = True
        self.tol = None
        self.eps = None
        self.nonfixed_rows = None
        self.fixed_rows = None
        self.full_rows_num = None
        self.nonfixed_cols = None
        self.fixed_cols = None
        self.full_cols_num = None
        self.J_T = None
        self.residuals = None
        self.full_rows_num = None
        self.full_rows_num = None
        self.solution_converged = None

    def _initialize_graph(self, biadjacency=None, adjacency_list=None, edgelist=None, degree_sequences=None):
        """
        Internal method for the initialization of the graph.
        Use the setter methods instead.

        :param biadjacency: binary input matrix describing the biadjacency matrix
                of a bipartite graph with the nodes of one layer along the rows
                and the nodes of the other layer along the columns.
        :type biadjacency: numpy.array, scipy.sparse, list, optional
        :param adjacency_list: dictionary that contains the adjacency list of nodes.
            The keys are considered as the nodes on the rows layer and the values,
            to be given as lists or numpy arrays, are considered as the nodes on the columns layer.
        :type adjacency_list: dict, optional
        :param edgelist: list of edges containing couples (row_node, col_node) of
            nodes forming an edge. each element in the couples must belong to
            the respective layer.
        :type edgelist: list, numpy.array, optional
        :param degree_sequences: couple of lists describing the degree sequences
            of both layers.
        :type degree_sequences: list, numpy.array, tuple, optional
        """
        if biadjacency is not None:
            if not isinstance(biadjacency, (list, np.ndarray)) and not scipy.sparse.isspmatrix(biadjacency):
                raise TypeError(
                    'The biadjacency matrix must be passed as a list or numpy array or scipy sparse matrix')
            else:
                if isinstance(biadjacency, list):
                    self.biadjacency = np.array(biadjacency)
                else:
                    self.biadjacency = biadjacency
                if self.biadjacency.shape[0] == self.biadjacency.shape[1]:
                    print(
                        'Your matrix is square. Please remember that it is treated as a biadjacency matrix, not an adjacency matrix.')
                self.adj_list, self.inv_adj_list, self.rows_deg, self.cols_deg = \
                    nef.adjacency_list_from_biadjacency(self.biadjacency)
                self.n_rows = len(self.rows_deg)
                self.n_cols = len(self.cols_deg)
                self._initialize_node_dictionaries()
                self.is_initialized = True

        elif edgelist is not None:
            if not isinstance(edgelist, (list, np.ndarray)):
                raise TypeError('The edgelist must be passed as a list or numpy array')
            elif len(edgelist[0]) != 2:
                raise ValueError(
                    'This is not an edgelist. An edgelist must be a vector of couples of nodes. Try passing a biadjacency matrix')
            else:
                self.adj_list, self.inv_adj_list, self.rows_deg, self.cols_deg, self.rows_dict, self.cols_dict = \
                    nef.adjacency_list_from_edgelist_bipartite(edgelist)
                self.inv_rows_dict = {v: k for k, v in self.rows_dict.items()}
                self.inv_cols_dict = {v: k for k, v in self.cols_dict.items()}
                self.is_initialized = True

        elif adjacency_list is not None:
            if not isinstance(adjacency_list, dict):
                raise TypeError('The adjacency list must be passed as a dictionary')
            else:
                self.adj_list, self.inv_adj_list, self.rows_deg, self.cols_deg, self.rows_dict, self.cols_dict = \
                    nef.adjacency_list_from_adjacency_list_bipartite(adjacency_list)
                self.inv_rows_dict = {v: k for k, v in self.rows_dict.items()}
                self.inv_cols_dict = {v: k for k, v in self.cols_dict.items()}
                self.is_initialized = True

        elif degree_sequences is not None:
            if not isinstance(degree_sequences, (list, np.ndarray, tuple)):
                raise TypeError('The degree sequences must be passed as a list, tuple or numpy array')
            elif len(degree_sequences) != 2:
                raise TypeError('degree_sequences must contain two vectors, the two layers degree sequences')
            elif not isinstance(degree_sequences[0], (list, np.ndarray)) or not isinstance(degree_sequences[1],
                                                                                           (list, np.ndarray)):
                raise TypeError('The two degree sequences must be lists or numpy arrays')
            elif np.sum(degree_sequences[0]) != np.sum(degree_sequences[1]):
                raise ValueError('The two degree sequences must have the same total sum.')
            elif (np.array(degree_sequences[0]) < 0).sum() + (np.array(degree_sequences[1]) < 0).sum() > 0:
                raise ValueError('A degree cannot be negative.')
            else:
                self.rows_deg = degree_sequences[0]
                self.cols_deg = degree_sequences[1]
                self.n_rows = len(self.rows_deg)
                self.n_cols = len(self.cols_deg)
                self._initialize_node_dictionaries()
                self.is_initialized = True

        if self.is_initialized:
            self.n_rows = len(self.rows_deg)
            self.n_cols = len(self.cols_deg)
            self.shape = [self.n_rows, self.n_cols]
            self.n_edges = np.sum(self.rows_deg)
            self.n_nodes = self.n_rows + self.n_cols

    def _initialize_node_dictionaries(self):
        self.rows_dict = dict(zip(np.arange(self.n_rows), np.arange(self.n_rows)))
        self.cols_dict = dict(zip(np.arange(self.n_cols), np.arange(self.n_cols)))
        self.inv_rows_dict = dict(zip(np.arange(self.n_rows), np.arange(self.n_rows)))
        self.inv_cols_dict = dict(zip(np.arange(self.n_cols), np.arange(self.n_cols)))

    def degree_reduction(self, rows_deg=None, cols_deg=None):
        """Reduce the degree sequences to contain no repetition of the degrees.
        The two parameters rows_deg and cols_deg are passed if there were some full or empty rows or columns.
        """
        if rows_deg is None:
            rows_deg = self.rows_deg
        else:
            cols_deg -= self.full_rows_num
        if cols_deg is None:
            cols_deg = self.cols_deg
        else:
            rows_deg -= self.full_cols_num
        self.r_rows_deg, self.r_invert_rows_deg, self.rows_multiplicity \
            = np.unique(rows_deg, return_index=False, return_inverse=True, return_counts=True)
        self.r_cols_deg, self.r_invert_cols_deg, self.cols_multiplicity \
            = np.unique(cols_deg, return_index=False, return_inverse=True, return_counts=True)
        self.r_n_rows = self.r_rows_deg.size
        self.r_n_cols = self.r_cols_deg.size
        self.r_dim = self.r_n_rows + self.r_n_cols
        self.r_n_edges = np.sum(rows_deg)
        self.is_reduced = True

    def _set_initial_guess(self):
        """
        Internal method to set the initial point of the solver.
        """
        if self.initial_guess is None:
            self.r_x = self.r_rows_deg / (
                np.sqrt(self.r_n_edges))  # This +1 increases the stability of the solutions.
            self.r_y = self.r_cols_deg / (np.sqrt(self.r_n_edges))
        elif self.initial_guess == 'random':
            self.r_x = np.random.rand(self.r_n_rows).astype(np.float64)
            self.r_y = np.random.rand(self.r_n_cols).astype(np.float64)
        elif self.initial_guess == 'uniform':
            self.r_x = np.ones(self.r_n_rows, dtype=np.float64)  # All probabilities will be 1/2 initially
            self.r_y = np.ones(self.r_n_cols, dtype=np.float64)
        elif self.initial_guess == 'degrees':
            self.r_x = self.r_rows_deg.astype(np.float64)
            self.r_y = self.r_cols_deg.astype(np.float64)
        if not self.exp:
            self.r_theta_x = - np.log(self.r_x)
            self.r_theta_y = - np.log(self.r_y)
            self.x0 = np.concatenate((self.r_theta_x, self.r_theta_y))
        else:
            self.x0 = np.concatenate((self.r_x, self.r_y))

    def initialize_avg_mat(self):
        """
        Reduces the matrix eliminating empty or full rows or columns.
        It repeats the process on the so reduced matrix until no more reductions are possible.
        For instance, a perfectly nested matrix will be reduced until all entries are set to 0 or 1.
        """
        self.avg_mat = np.zeros_like(self.biadjacency, dtype=float)
        r_biad_mat = np.copy(self.biadjacency)
        rows_num, cols_num = self.biadjacency.shape
        rows_degs = self.biadjacency.sum(1)
        cols_degs = self.biadjacency.sum(0)
        good_rows = np.arange(rows_num)
        good_cols = np.arange(cols_num)
        zero_rows = np.where(rows_degs == 0)[0]
        zero_cols = np.where(cols_degs == 0)[0]
        full_rows = np.where(rows_degs == cols_num)[0]
        full_cols = np.where(cols_degs == rows_num)[0]
        self.full_rows_num = 0
        self.full_cols_num = 0
        while zero_rows.size + zero_cols.size + full_rows.size + full_cols.size > 0:
            r_biad_mat = r_biad_mat[np.delete(np.arange(r_biad_mat.shape[0]), zero_rows), :]
            r_biad_mat = r_biad_mat[:, np.delete(np.arange(r_biad_mat.shape[1]), zero_cols)]
            good_rows = np.delete(good_rows, zero_rows)
            good_cols = np.delete(good_cols, zero_cols)
            full_rows = np.where(r_biad_mat.sum(1) == r_biad_mat.shape[1])[0]
            full_cols = np.where(r_biad_mat.sum(0) == r_biad_mat.shape[0])[0]
            self.full_rows_num += len(full_rows)
            self.full_cols_num += len(full_cols)
            self.avg_mat[good_rows[full_rows][:, None], good_cols] = 1
            self.avg_mat[good_rows[:, None], good_cols[full_cols]] = 1
            good_rows = np.delete(good_rows, full_rows)
            good_cols = np.delete(good_cols, full_cols)
            r_biad_mat = r_biad_mat[np.delete(np.arange(r_biad_mat.shape[0]), full_rows), :]
            r_biad_mat = r_biad_mat[:, np.delete(np.arange(r_biad_mat.shape[1]), full_cols)]
            zero_rows = np.where(r_biad_mat.sum(1) == 0)[0]
            zero_cols = np.where(r_biad_mat.sum(0) == 0)[0]

        self.nonfixed_rows = good_rows
        self.fixed_rows = np.delete(np.arange(rows_num), good_rows)
        self.nonfixed_cols = good_cols
        self.fixed_cols = np.delete(np.arange(cols_num), good_cols)
        return r_biad_mat

    def _initialize_fitnesses(self):
        """
        Internal method to initialize the fitnesses of the BiCM.
        If there are empty rows, the corresponding fitnesses are set to 0,
        while for full rows the corresponding columns are set to numpy.inf.
        """
        if not self.exp:
            self.theta_x = np.zeros(self.n_rows, dtype=float)
            self.theta_y = np.zeros(self.n_cols, dtype=float)
        self.x = np.zeros(self.n_rows, dtype=float)
        self.y = np.zeros(self.n_cols, dtype=float)
        good_rows = np.arange(self.n_rows)
        good_cols = np.arange(self.n_cols)
        bad_rows = np.array([])
        bad_cols = np.array([])
        self.full_rows_num = 0
        self.full_cols_num = 0
        if np.any(np.isin(self.rows_deg, (0, self.n_cols))) or np.any(np.isin(self.cols_deg, (0, self.n_rows))):
            print('''
                      WARNING: this system has at least a node that is disconnected or connected to all nodes of the opposite layer. 
                      This may cause some convergence issues.
                      Please use the full mode providing a biadjacency matrix or an edgelist, or clean your data from these nodes. 
                      ''')
            zero_rows = np.where(self.rows_deg == 0)[0]
            zero_cols = np.where(self.cols_deg == 0)[0]
            full_rows = np.where(self.rows_deg == self.n_cols)[0]
            full_cols = np.where(self.cols_deg == self.n_rows)[0]
            self.x[full_rows] = np.inf
            self.y[full_cols] = np.inf
            if not self.exp:
                self.theta_x[full_rows] = - np.inf
                self.theta_y[full_rows] = - np.inf
                self.theta_x[zero_rows] = np.inf
                self.theta_y[zero_cols] = np.inf
            bad_rows = np.concatenate((zero_rows, full_rows))
            bad_cols = np.concatenate((zero_cols, full_cols))
            good_rows = np.delete(np.arange(self.n_rows), bad_rows)
            good_cols = np.delete(np.arange(self.n_cols), bad_cols)
            self.full_rows_num += len(full_rows)
            self.full_cols_num += len(full_cols)

        self.nonfixed_rows = good_rows
        self.fixed_rows = bad_rows
        self.nonfixed_cols = good_cols
        self.fixed_cols = bad_cols

    def _initialize_problem(self, rows_deg=None, cols_deg=None):
        """
        Initializes the solver reducing the degree sequences,
        setting the initial guess and setting the functions for the solver.
        The two parameters rows_deg and cols_deg are passed if there were some full or empty rows or columns.
        """
        if ~self.is_reduced:
            self.degree_reduction(rows_deg=rows_deg, cols_deg=cols_deg)
        self._set_initial_guess()
        if self.method == 'root':
            self.J_T = np.zeros((self.r_dim, self.r_dim), dtype=np.float64)
            self.residuals = np.zeros(self.r_dim, dtype=np.float64)
        else:
            self.args = (self.r_rows_deg, self.r_cols_deg, self.rows_multiplicity, self.cols_multiplicity)
            d_fun = {
                'newton': lambda x: - mof.loglikelihood_prime_bicm(x, self.args),
                'quasinewton': lambda x: - mof.loglikelihood_prime_bicm(x, self.args),
                'fixed-point': lambda x: mof.iterative_bicm(x, self.args),
                'newton_exp': lambda x: - mof.loglikelihood_prime_bicm_exp(x, self.args),
                'quasinewton_exp': lambda x: - mof.loglikelihood_prime_bicm_exp(x, self.args),
                'fixed-point_exp': lambda x: mof.iterative_bicm_exp(x, self.args),
            }
            d_fun_jac = {
                'newton': lambda x: - mof.loglikelihood_hessian_bicm(x, self.args),
                'quasinewton': lambda x: - mof.loglikelihood_hessian_diag_bicm(x, self.args),
                'fixed-point': None,
                'newton_exp': lambda x: - mof.loglikelihood_hessian_bicm_exp(x, self.args),
                'quasinewton_exp': lambda x: - mof.loglikelihood_hessian_diag_bicm_exp(x, self.args),
                'fixed-point_exp': None,
            }
            d_fun_step = {
                'newton': lambda x: - mof.loglikelihood_bicm(x, self.args),
                'quasinewton': lambda x: - mof.loglikelihood_bicm(x, self.args),
                'fixed-point': lambda x: - mof.loglikelihood_bicm(x, self.args),
                'newton_exp': lambda x: - mof.loglikelihood_bicm_exp(x, self.args),
                'quasinewton_exp': lambda x: - mof.loglikelihood_bicm_exp(x, self.args),
                'fixed-point_exp': lambda x: - mof.loglikelihood_bicm_exp(x, self.args),
            }

            if self.exp:
                self.hessian_regulariser = sof.matrix_regulariser_function_eigen_based
            else:
                self.hessian_regulariser = sof.matrix_regulariser_function

            if self.exp:
                lins_args = (mof.loglikelihood_bicm_exp, self.args)
            else:
                lins_args = (mof.loglikelihood_bicm, self.args)
            lins_fun = {
                'newton': lambda x: mof.linsearch_fun_BiCM(x, lins_args),
                'quasinewton': lambda x: mof.linsearch_fun_BiCM(x, lins_args),
                'fixed-point': lambda x: mof.linsearch_fun_BiCM_fixed(x),
                'newton_exp': lambda x: mof.linsearch_fun_BiCM_exp(x, lins_args),
                'quasinewton_exp': lambda x: mof.linsearch_fun_BiCM_exp(x, lins_args),
                'fixed-point_exp': lambda x: mof.linsearch_fun_BiCM_exp_fixed(x),
            }
            if self.exp:
                method = self.method + '_exp'
            else:
                method = self.method
            try:
                self.fun = d_fun[method]
                self.fun_jac = d_fun_jac[method]
                self.step_fun = d_fun_step[method]
                self.fun_linsearch = lins_fun[method]
            except (TypeError, KeyError):
                raise ValueError('Method must be "newton","quasinewton", "fixed-point" or "root".')

    def _equations_root(self, x):
        """
        Equations for the *root* solver
        """
        mof.eqs_root(x, self.r_rows_deg, self.r_cols_deg,
                     self.rows_multiplicity, self.cols_multiplicity,
                     self.r_n_rows, self.r_n_cols, self.residuals)

    def _jacobian_root(self, x):
        """
        Jacobian for the *root* solver
        """
        mof.jac_root(x, self.rows_multiplicity, self.cols_multiplicity,
                     self.r_n_rows, self.r_n_cols, self.J_T)

    def _residuals_jacobian(self, x):
        """
        Residuals and jacobian for the *root* solver
        """
        self._equations_root(x)
        self._jacobian_root(x)
        return self.residuals, self.J_T

    def _clean_root(self):
        """
        Clean variables used for the *root* solver
        """
        self.J_T = None
        self.residuals = None

    def _solve_root(self):
        """
        Internal *root* solver
        """
        x0 = self.x0
        opz = {'col_deriv': True, 'diag': None}
        res = scipy.optimize.root(self._residuals_jacobian, x0,
                                  method='hybr', jac=True, options=opz)
        self._clean_root()
        return res.x

    @staticmethod
    def check_sol(biad_mat, avg_bicm, return_error=False, in_place=False):
        """
        This function prints the rows sums differences between two matrices, that originally are the biadjacency matrix and its bicm2 average matrix.
        The intended use of this is to check if an average matrix is actually a solution for a bipartite configuration model.

        If return_error is set to True, it returns 1 if the sum of the differences is bigger than 1.

        If in_place is set to True, it checks and sums also the total error entry by entry.
        The intended use of this is to check if two solutions are the same solution.
        """
        error = 0
        if np.any(avg_bicm < 0):
            print('Negative probabilities in the average matrix! This means negative node fitnesses.')
            error = 1
        if np.any(avg_bicm > 1):
            print('Probabilities greater than 1 in the average matrix! This means negative node fitnesses.')
            error = 1
        rows_error_vec = np.abs(np.sum(biad_mat, axis=1) - np.sum(avg_bicm, axis=1))
        err_rows = np.max(rows_error_vec)
        print('max rows error =', err_rows)
        cols_error_vec = np.abs(np.sum(biad_mat, axis=0) - np.sum(avg_bicm, axis=0))
        err_cols = np.max(cols_error_vec)
        print('max columns error =', err_cols)
        tot_err = np.sum(rows_error_vec) + np.sum(cols_error_vec)
        print('total error =', tot_err)
        if tot_err > 1:
            error = 1
            print('WARNING total error > 1')
            if tot_err > 10:
                print('total error > 10')
        if err_rows + err_cols > 1:
            print('max error > 1')
            error = 1
            if err_rows + err_cols > 10:
                print('max error > 10')
        if in_place:
            diff_mat = np.abs(biad_mat - avg_bicm)
            print('In-place total error:', np.sum(diff_mat))
            print('In-place max error:', np.max(diff_mat))
        if return_error:
            return error
        else:
            return

    @staticmethod
    def check_sol_light(x, y, rows_deg, cols_deg, return_error=False):
        """
        Light version of the check_sol function, working only on the fitnesses and the degree sequences.
        """
        error = 0
        rows_error_vec = []
        for i in range(len(x)):
            row_avgs = x[i] * y / (1 + x[i] * y)
            if error == 0:
                if np.any(row_avgs < 0):
                    print('Warning: negative link probabilities')
                    error = 1
                if np.any(row_avgs > 1):
                    print('Warning: link probabilities > 1')
                    error = 1
            rows_error_vec.append(np.sum(row_avgs) - rows_deg[i])
        rows_error_vec = np.abs(rows_error_vec)
        err_rows = np.max(rows_error_vec)
        print('max rows error =', err_rows)
        cols_error_vec = np.abs([(x * y[j] / (1 + x * y[j])).sum() - cols_deg[j] for j in range(len(y))])
        err_cols = np.max(cols_error_vec)
        print('max columns error =', err_cols)
        tot_err = np.sum(rows_error_vec) + np.sum(cols_error_vec)
        print('total error =', tot_err)
        if tot_err > 1:
            error = 1
            print('WARNING total error > 1')
            if tot_err > 10:
                print('total error > 10')
        if err_rows + err_cols > 1:
            print('max error > 1')
            error = 1
            if err_rows + err_cols > 10:
                print('max error > 10')
        if return_error:
            return error
        else:
            return

    def _check_solution(self, return_error=False, in_place=False):
        """
        Check if the solution of the BiCM is compatible with the degree sequences of the graph.

        :param bool return_error: If this is set to true, return 1 if the solution is not correct, 0 otherwise.
        :param bool in_place: check also the error in the single entries of the matrices.
            Always False unless comparing two different solutions.
        """
        if self.biadjacency is not None and self.avg_mat is not None:
            return self.check_sol(self.biadjacency, self.avg_mat, return_error=return_error, in_place=in_place)
        else:
            return self.check_sol_light(self.x[self.nonfixed_rows], self.y[self.nonfixed_cols],
                                        self.rows_deg[self.nonfixed_rows] - self.full_cols_num,
                                        self.cols_deg[self.nonfixed_cols] - self.full_rows_num,
                                        return_error=return_error)

    def _set_solved_problem(self, solution):
        """
        Sets the solution of the problem.

        :param numpy.ndarray solution: A numpy array containing that reduced fitnesses of the two layers, consecutively.
        """
        if not self.exp:
            if self.theta_x is None:
                self.theta_x = np.zeros(self.n_rows)
            if self.theta_y is None:
                self.theta_y = np.zeros(self.n_cols)
            self.r_theta_xy = solution
            self.r_theta_x = self.r_theta_xy[:self.r_n_rows]
            self.r_theta_y = self.r_theta_xy[self.r_n_rows:]
            self.r_xy = np.exp(- self.r_theta_xy)
            self.r_x = np.exp(- self.r_theta_x)
            self.r_y = np.exp(- self.r_theta_y)
            self.theta_x[self.nonfixed_rows] = self.r_theta_x[self.r_invert_rows_deg]
            self.theta_y[self.nonfixed_cols] = self.r_theta_y[self.r_invert_cols_deg]
        else:
            self.r_xy = solution
            self.r_x = self.r_xy[:self.r_n_rows]
            self.r_y = self.r_xy[self.r_n_rows:]
        if self.x is None:
            self.x = np.zeros(self.n_rows)
        if self.y is None:
            self.y = np.zeros(self.n_cols)
        self.x[self.nonfixed_rows] = self.r_x[self.r_invert_rows_deg]
        self.y[self.nonfixed_cols] = self.r_y[self.r_invert_cols_deg]
        self.dict_x = dict([(self.rows_dict[i], self.x[i]) for i in range(len(self.x))])
        self.dict_y = dict([(self.cols_dict[j], self.y[j]) for j in range(len(self.y))])
        if self.full_return:
            self.comput_time = solution[1]
            self.n_steps = solution[2]
            self.norm_seq = solution[3]
            self.diff_seq = solution[4]
            self.alfa_seq = solution[5]
        # Reset solver lambda functions for multiprocessing compatibility
        self.hessian_regulariser = None
        self.fun = None
        self.fun_jac = None
        self.step_fun = None
        self.fun_linsearch = None

    def _solve_bicm_full(self):
        """
        Internal method for computing the solution of the BiCM via matrices.
        """
        r_biadjacency = self.initialize_avg_mat()
        if len(r_biadjacency) > 0:  # Every time the matrix is not perfectly nested
            rows_deg = self.rows_deg[self.nonfixed_rows]
            cols_deg = self.cols_deg[self.nonfixed_cols]
            self._initialize_problem(rows_deg=rows_deg, cols_deg=cols_deg)
            if self.method == 'root':
                sol = self._solve_root()
            else:
                x0 = self.x0
                sol = sof.solver(
                    x0,
                    fun=self.fun,
                    fun_jac=self.fun_jac,
                    step_fun=self.step_fun,
                    linsearch_fun=self.fun_linsearch,
                    hessian_regulariser=self.hessian_regulariser,
                    tol=self.tol,
                    eps=self.eps,
                    max_steps=self.max_steps,
                    method=self.method,
                    verbose=self.verbose,
                    regularise=self.regularise,
                    full_return=self.full_return,
                    linsearch=self.linsearch,
                )
            self._set_solved_problem(sol)
            r_avg_mat = nef.bicm_from_fitnesses(self.x[self.nonfixed_rows], self.y[self.nonfixed_cols])
            self.avg_mat[self.nonfixed_rows[:, None], self.nonfixed_cols] = np.copy(r_avg_mat)

    def _solve_bicm_light(self):
        """
        Internal method for computing the solution of the BiCM via degree sequences.
        """
        self._initialize_fitnesses()
        rows_deg = self.rows_deg[self.nonfixed_rows]
        cols_deg = self.cols_deg[self.nonfixed_cols]
        self._initialize_problem(rows_deg=rows_deg, cols_deg=cols_deg)
        if self.method == 'root':
            sol = self._solve_root()
        else:
            x0 = self.x0
            sol = sof.solver(
                x0,
                fun=self.fun,
                fun_jac=self.fun_jac,
                step_fun=self.step_fun,
                linsearch_fun=self.fun_linsearch,
                hessian_regulariser=self.hessian_regulariser,
                tol=self.tol,
                eps=self.eps,
                max_steps=self.max_steps,
                method=self.method,
                verbose=self.verbose,
                regularise=self.regularise,
                full_return=self.full_return,
                linsearch=self.linsearch,
            )
        self._set_solved_problem(sol)

    def _set_parameters(self, method, initial_guess, tol, eps, regularise, max_steps, verbose, linsearch, exp,
                        full_return):
        """
        Internal method for setting the parameters of the solver.
        """
        self.method = method
        self.initial_guess = initial_guess
        self.tol = tol
        self.eps = eps
        self.verbose = verbose
        self.linsearch = linsearch
        self.regularise = regularise
        self.exp = exp
        self.full_return = full_return
        if max_steps is None:
            if method == 'fixed-point':
                self.max_steps = 200
            else:
                self.max_steps = 100
        else:
            self.max_steps = max_steps

    def solve_tool(
            self,
            method='newton',
            initial_guess=None,
            light_mode=None,
            tol=1e-8,
            eps=1e-8,
            max_steps=None,
            verbose=False,
            linsearch=True,
            regularise=None,
            print_error=True,
            full_return=False,
            exp=False):
        """Solve the BiCM of the graph.
        It does not return the solution, use the getter methods instead.

        :param bool light_mode: Doesn't use matrices in the computation if this is set to True.
            If the graph has been initialized without the matrix, the light mode is used regardless.
        :param str method: Method of choice among *newton*, *quasinewton* or *iterative*, default is newton
        :param str initial_guess: Initial guess of choice among *None*, *random*, *uniform* or *degrees*, default is None
        :param float tol: Tolerance of the solution, optional
        :param int max_steps: Maximum number of steps, optional
        :param bool, optional verbose: Print elapsed time, errors and iteration steps, optional
        :param bool linsearch: Implement the linesearch when searching for roots, default is True
        :param bool regularise: Regularise the matrices in the computations, optional
        :param bool print_error: Print the final error of the solution
        :param bool exp: if this is set to true the solver works with the reparameterization $x_i = e^{-\theta_i}$,
            $y_\alpha = e^{-\theta_\alpha}$. It might be slightly faster but also might not converge.
        :param full_return: If True the algorithm returns more statistics than the obtained solution, defaults to False.
        :type full_return: bool, optional
        :param eps: parameter controlling the tolerance of the difference between two iterations, defaults to 1e-8.
        :type eps: float, optional
        """
        if not self.is_initialized:
            print('Graph is not initialized. I can\'t compute the BiCM.')
            return
        if regularise is None:
            if exp:
                regularise = False
            else:
                regularise = True
        if regularise and exp:
            print('Warning: regularise is only recommended in non-exp mode.')
        if method == 'root':
            exp = True
        self._set_parameters(method=method, initial_guess=initial_guess, tol=tol, eps=eps, regularise=regularise,
                             max_steps=max_steps, verbose=verbose, linsearch=linsearch, exp=exp,
                             full_return=full_return)
        if self.biadjacency is not None and (light_mode is None or not light_mode):
            self._solve_bicm_full()
        else:
            if light_mode is False:
                print('''
                I cannot work with the full mode without the biadjacency matrix.
                This will not account for disconnected or fully connected nodes.
                Solving in light mode...
                ''')
            self._solve_bicm_light()
        if print_error:
            self.solution_converged = not bool(self._check_solution(return_error=True))
            if self.solution_converged:
                print('Solver converged.')
            else:
                print('Solver did not converge.')
        self.is_randomized = True

    def get_bicm_matrix(self):
        """Get the matrix of probabilities of the BiCM.
        If the BiCM has not been computed, it also computes it with standard settings.

        :returns: The average matrix of the BiCM
        :rtype: numpy.ndarray
        """
        if not self.is_initialized:
            raise ValueError('Graph is not initialized. I can\'t compute the BiCM')
        elif not self.is_randomized:
            self.solve_tool()
        if self.avg_mat is not None:
            return self.avg_mat
        else:
            self.avg_mat = nef.bicm_from_fitnesses(self.x, self.y)
            return self.avg_mat

    def get_bicm_fitnesses(self):
        """Get the fitnesses of the BiCM.
        If the BiCM has not been computed, it also computes it with standard settings.

        :returns: The fitnesses of the BiCM in the format **rows fitnesses dictionary, columns fitnesses dictionary**
        """
        if not self.is_initialized:
            raise ValueError('Graph is not initialized. I can\'t compute the BiCM')
        elif not self.is_randomized:
            print('Computing the BiCM...')
            self.solve_tool()
        return self.dict_x, self.dict_y

    def _pval_calculator(self, v_list_key, x, y):
        """
        Calculate the p-values of the v-motifs numbers of one vertices and all its neighbours.

        :param int v_list_key: the key of the node to consider for the adjacency list of the v-motifs.
        :param numpy.ndarray x: the fitnesses of the layer of the desired projection.
        :param numpy.ndarray y: the fitnesses of the opposite layer.
        :returns: a dictionary containing as keys the nodes that form v-motifs with the considered node, and as values the corresponding p-values.
        """
        node_xy = x[v_list_key] * y
        temp_pvals_dict = {}
        for neighbor in self.v_adj_list[v_list_key]:
            neighbor_xy = x[neighbor] * y
            probs = node_xy * neighbor_xy / ((1 + node_xy) * (1 + neighbor_xy))
            avg_v = np.sum(probs)
            if self.projection_method == 'poisson':
                temp_pvals_dict[neighbor] = \
                    poisson.sf(k=self.v_adj_list[v_list_key][neighbor] - 1, mu=avg_v)
            elif self.projection_method == 'normal':
                sigma_v = np.sqrt(np.sum(probs * (1 - probs)))
                temp_pvals_dict[neighbor] = \
                    norm.cdf((self.v_adj_list[v_list_key][neighbor] + 0.5 - avg_v) / sigma_v)
            elif self.projection_method == 'rna':
                var_v_arr = probs * (1 - probs)
                sigma_v = np.sqrt(np.sum(var_v_arr))
                gamma_v = (sigma_v ** (-3)) * np.sum(var_v_arr * (1 - 2 * probs))
                eval_x = (self.v_adj_list[v_list_key][neighbor] + 0.5 - avg_v) / sigma_v
                pval = norm.cdf(eval_x) + gamma_v * (1 - eval_x ** 2) * norm.pdf(eval_x) / 6
                temp_pvals_dict[neighbor] = max(min(pval, 1), 0)
        return temp_pvals_dict

    def _pval_calculator_poibin(self, deg_couple, deg_dict, x, y):
        """
        Calculate the p-values of the v-motifs numbers of all nodes with a given couple degrees.

        :param tuple deg_couple: the couple of degrees considered.
        :param dict deg_dict: the dictionary containing for each degree the list of nodes of that degree.
        :param numpy.ndarray x: the fitnesses of the layer of the desired projection.
        :param numpy.ndarray y: the fitnesses of the opposite layer.
        :returns: a list containing 3-tuples with the two nodes considered and their p-value.
        """
        node_xy = x[deg_dict[deg_couple[0]][0]] * y
        neighbor_xy = x[deg_dict[deg_couple[1]][0]] * y
        probs = node_xy * neighbor_xy / ((1 + node_xy) * (1 + neighbor_xy))
        pb_obj = pb.PoiBin(probs)
        pval_dict = dict()
        for node1 in deg_dict[deg_couple[0]]:
            for node2 in deg_dict[deg_couple[1]]:
                if node2 in self.v_adj_list[node1]:
                    if node1 not in pval_dict:
                        pval_dict[node1] = dict()
                    pval_dict[node1][node2] = pb_obj.pval(int(self.v_adj_list[node1][node2]))
                elif node1 in self.v_adj_list[node2]:
                    if node2 not in pval_dict:
                        pval_dict[node2] = dict()
                    pval_dict[node2][node1] = pb_obj.pval(int(self.v_adj_list[node2][node1]))
        return pval_dict

    def _projection_calculator(self):
        if self.rows_projection:
            adj_list = self.adj_list
            inv_adj_list = self.inv_adj_list
            x = self.x
            y = self.y
        else:
            adj_list = self.inv_adj_list
            inv_adj_list = self.adj_list
            x = self.y
            y = self.x

        self.v_adj_list = {k: dict() for k in list(adj_list.keys())}
        for node_i in np.sort(list(adj_list.keys())):
            for neighbor in adj_list[node_i]:
                for node_j in inv_adj_list[neighbor]:
                    if node_j > node_i:
                        self.v_adj_list[node_i][node_j] = self.v_adj_list[node_i].get(node_j, 0) + 1
        if self.projection_method != 'poibin':
            v_list_keys = list(self.v_adj_list.keys())
            pval_adj_list = dict()
            if self.threads_num > 1:
                with Pool(processes=self.threads_num) as pool:
                    partial_function = partial(self._pval_calculator, x=x, y=y)
                    if self.progress_bar:
                        pvals_dicts = pool.map(partial_function, tqdm(v_list_keys))
                    else:
                        pvals_dicts = pool.map(partial_function, v_list_keys)
                for k_i in range(len(v_list_keys)):
                    k = v_list_keys[k_i]
                    pval_adj_list[k] = pvals_dicts[k_i]
            else:
                if self.progress_bar:
                    for k in tqdm(v_list_keys):
                        pval_adj_list[k] = self._pval_calculator(k, x=x, y=y)
                else:
                    for k in v_list_keys:
                        pval_adj_list[k] = self._pval_calculator(k, x=x, y=y)
        else:
            if self.rows_projection:
                degs = self.rows_deg
            else:
                degs = self.cols_deg
            unique_degs = np.unique(degs)
            deg_dict = {k: [] for k in unique_degs}
            for k_i in range(len(degs)):
                deg_dict[degs[k_i]].append(k_i)

            v_list_coupled = []
            deg_couples = [(unique_degs[deg_i], unique_degs[deg_j])
                           for deg_i in range(len(unique_degs))
                           for deg_j in range(deg_i, len(unique_degs))]
            if self.progress_bar:
                print('Calculating p-values...')
            if self.threads_num > 1:
                with Pool(processes=self.threads_num) as pool:
                    partial_function = partial(self._pval_calculator_poibin, deg_dict=deg_dict, x=x, y=y)
                    if self.progress_bar:
                        pvals_dicts = pool.map(partial_function, tqdm(deg_couples))
                    else:
                        pvals_dicts = pool.map(partial_function, deg_couples)
            else:
                pvals_dicts = []
                if self.progress_bar:
                    for deg_couple in tqdm(deg_couples):
                        pvals_dicts.append(
                            self._pval_calculator_poibin(deg_couple, deg_dict=deg_dict, x=x, y=y))
                else:
                    for deg_couple in v_list_coupled:
                        pvals_dicts.append(
                            self._pval_calculator_poibin(deg_couple, deg_dict=deg_dict, x=x, y=y))
            pval_adj_list = {k: dict() for k in self.v_adj_list}
            for pvals_dict in pvals_dicts:
                for node in pvals_dict:
                    pval_adj_list[node].update(pvals_dict[node])
        return pval_adj_list

    def compute_projection(self, rows=True, alpha=0.05, method='poisson', threads_num=None, progress_bar=True):
        """Compute the projection of the network on the rows or columns layer.
        If the BiCM has not been computed, it also computes it with standard settings.
        This is the most customizable method for the pvalues computation.

        :param bool rows: True if requesting the rows projection.
        :param float alpha: Threshold for the FDR validation.
        :param str method: Method for the approximation of the pvalues computation.
            Implemented methods are *poisson*, *poibin*, *normal*, *rna*.
        :param threads_num: Number of threads to use for the parallelization. If it is set to 1,
            the computation is not parallelized.
        :param bool progress_bar: Show progress bar of the pvalues computation.
        """
        self.rows_projection = rows
        self.projection_method = method
        self.progress_bar = progress_bar
        if threads_num is None:
            if system() == 'Windows':
                threads_num = 1
            else:
                threads_num = 4
        else:
            if system() == 'Windows' and threads_num != 1:
                threads_num = 1
                print('Parallel processes not yet implemented on Windows, computing anyway...')
        self.threads_num = threads_num
        if self.adj_list is None:
            print('''
            Without the edges I can't compute the projection. 
            Use set_biadjacency_matrix, set_adjacency_list or set_edgelist to add edges.
            ''')
            return
        else:
            if not self.is_randomized:
                print('First I have to compute the BiCM. Computing...')
                self.solve_tool()
            if rows:
                self.rows_pvals = self._projection_calculator()
                self.projected_rows_adj_list = self._projection_from_pvals(alpha=alpha)
                self.is_rows_projected = True
            else:
                self.cols_pvals = self._projection_calculator()
                self.projected_cols_adj_list = self._projection_from_pvals(alpha=alpha)
                self.is_cols_projected = True

    def _pvals_validator(self, pval_list, alpha=0.05):
        sorted_pvals = np.sort(pval_list)
        if self.rows_projection:
            multiplier = 2 * alpha / (self.n_rows * (self.n_rows - 1))
        else:
            multiplier = 2 * alpha / (self.n_cols * (self.n_cols - 1))
        try:
            eff_fdr_pos = np.where(sorted_pvals <= (np.arange(1, len(sorted_pvals) + 1) * alpha * multiplier))[0][-1]
        except IndexError:
            print('No V-motifs will be validated. Try increasing alpha')
            eff_fdr_pos = 0
        eff_fdr_th = eff_fdr_pos * multiplier
        return eff_fdr_th

    def _projection_from_pvals(self, alpha=0.05):
        """Internal method to build the projected network from pvalues.

        :param float alpha:  Threshold for the FDR validation.
        """
        pval_list = []
        if self.rows_projection:
            pvals_adj_list = self.rows_pvals
        else:
            pvals_adj_list = self.cols_pvals
        for node in pvals_adj_list:
            for neighbor in pvals_adj_list[node]:
                pval_list.append(pvals_adj_list[node][neighbor])
        eff_fdr_th = self._pvals_validator(pval_list, alpha=alpha)
        projected_adj_list = dict([])
        for node in self.v_adj_list:
            for neighbor in self.v_adj_list[node]:
                if pvals_adj_list[node][neighbor] <= eff_fdr_th:
                    if node not in projected_adj_list.keys():
                        projected_adj_list[node] = []
                    projected_adj_list[node].append(neighbor)
        return projected_adj_list
        #     return np.array([(v[0], v[1]) for v in self.rows_pvals if v[2] <= eff_fdr_th])
        # else:
        #     eff_fdr_th = nef.pvals_validator([v[2] for v in self.cols_pvals], self.n_cols, alpha=alpha)
        #     return np.array([(v[0], v[1]) for v in self.cols_pvals if v[2] <= eff_fdr_th])

    def get_rows_projection(self, alpha=0.05, method='poisson', threads_num=None, progress_bar=True):
        """Get the projected network on the rows layer of the graph.

        :param alpha: threshold for the validation of the projected edges.
        :type alpha: float, optional
        :param method: approximation method for the calculation of the p-values.a
            Implemented choices are: poisson, poibin, normal, rna
        :type method: str, optional
        :param threads_num: number of threads to use for the parallelization. If it is set to 1,
            the computation is not parallelized.
        :type threads_num: int, optional
        :param bool progress_bar: Show the progress bar
        :returns: edgelist of the projected network on the rows layer
        :rtype: numpy.array
        """
        if not self.is_rows_projected:
            self.compute_projection(rows=True, alpha=alpha, method=method, threads_num=threads_num,
                                    progress_bar=progress_bar)
        if self.rows_dict is None:
            return self.projected_rows_adj_list
        else:
            adj_list_to_return = {}
            for node in self.projected_rows_adj_list:
                adj_list_to_return[self.rows_dict[node]] = []
                for neighbor in self.projected_rows_adj_list[node]:
                    adj_list_to_return[self.rows_dict[node]].append(self.rows_dict[neighbor])
            return adj_list_to_return

    def get_cols_projection(self, alpha=0.05, method='poisson', threads_num=4, progress_bar=True):
        """Get the projected network on the columns layer of the graph.

        :param alpha: threshold for the validation of the projected edges.
        :type alpha: float, optional
        :param method: approximation method for the calculation of the p-values.
            Implemented choices are: poisson, poibin, normal, rna
        :type method: str, optional
        :param threads_num: number of threads to use for the parallelization. If it is set to 1,
            the computation is not parallelized.
        :type threads_num: int, optional
        :param bool progress_bar: Show the progress bar
        :returns: edgelist of the projected network on the columns layer
        :rtype: numpy.array
        """
        if not self.is_cols_projected:
            self.compute_projection(rows=False,
                                    alpha=alpha, method=method, threads_num=threads_num, progress_bar=progress_bar)
        if self.cols_dict is None:
            return self.projected_cols_adj_list
        else:
            adj_list_to_return = {}
            for node in self.projected_cols_adj_list:
                adj_list_to_return[self.cols_dict[node]] = []
                for neighbor in self.projected_cols_adj_list[node]:
                    adj_list_to_return[self.cols_dict[node]].append(self.cols_dict[neighbor])
            return adj_list_to_return

    def set_biadjacency_matrix(self, biadjacency):
        """Set the biadjacency matrix of the graph.

        :param biadjacency: binary input matrix describing the biadjacency matrix
                of a bipartite graph with the nodes of one layer along the rows
                and the nodes of the other layer along the columns.
        :type biadjacency: numpy.array, scipy.sparse, list
        """
        if self.is_initialized:
            print('Graph already contains edges or has a degree sequence. Use clean_edges() first.')
        else:
            self._initialize_graph(biadjacency=biadjacency)

    def set_adjacency_list(self, adj_list):
        """Set the adjacency list of the graph.

        :param adj_list: a dictionary containing the adjacency list
                of a bipartite graph with the nodes of one layer as keys
                and lists of neighbor nodes of the other layer as values.
        :type adj_list: dict
        """
        if self.is_initialized:
            print('Graph already contains edges or has a degree sequence. Use clean_edges() first.')
        else:
            self._initialize_graph(adjacency_list=adj_list)

    def set_edgelist(self, edgelist):
        """Set the edgelist of the graph.

        :param edgelist: list of edges containing couples (row_node, col_node) of
            nodes forming an edge. each element in the couples must belong to
            the respective layer.
        :type edgelist: list, numpy.array
        """
        if self.is_initialized:
            print('Graph already contains edges or has a degree sequence. Use clean_edges() first.')
        else:
            self._initialize_graph(edgelist=edgelist)

    def set_degree_sequences(self, degree_sequences):
        """Set the degree sequence of the graph.

        :param degree_sequences: couple of lists describing the degree sequences
            of both layers.
        :type degree_sequences: list, numpy.array, tuple
        """
        if self.is_initialized:
            print('Graph already contains edges or has a degree sequence. Use clean_edges() first.')
        else:
            self._initialize_graph(degree_sequences=degree_sequences)

    def clean_edges(self):
        """Clean the edges of the graph.
        """
        self.biadjacency = None
        self.edgelist = None
        self.adj_list = None
        self.rows_deg = None
        self.cols_deg = None
        self.is_initialized = False
