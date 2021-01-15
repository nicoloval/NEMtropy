import sys
import os
sys.path.append("../")
import Undirected_graph_Class as sample
import Matrix_Generator as mg
import numpy as np
import unittest  # test tool
import random
import networkx as nx

N, seed = (50, 42)
A  = mg.random_weighted_matrix_generator_dense(
    n=N, sup_ext=10, sym=True, seed=seed, intweights=True
    )
