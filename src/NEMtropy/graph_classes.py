import numpy as np
import scipy.sparse
import scipy
import os
from . import models_functions as mof
from . import solver_functions as sof
from . import network_functions as nef
from . import ensemble_generator as eg


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
        :param tol: parameter controlling the tollerance of the norm the gradient function, defaults to 1e-8.
        :type tol: float, optional
        :param eps: parameter controlling the tollerance of the difference between two iterations, defaults to 1e-8.
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
