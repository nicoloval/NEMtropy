import sys
import os
sys.path.append('../')
import Directed_graph_Class as sample
import numpy as np
import unittest


class MyTest(unittest.TestCase):


    def setUp(self):
        pass


    def test_CReAMa_dcm_Newton_Emid(self):
        """test with 3 classes of cardinality 1
        and no zero degrees
        """
        __file__ = "network.txt"
        __location__ = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__)))
        path = os.path.join(__location__,__file__)
        network = np.loadtxt(path,delimiter=';')
    
        g = sample.DirectedGraph(adjacency=network)

        g.solve_tool(model='CReAMa', method='fixed-point', adjacency='dcm',  max_steps=1000, verbose=False)

        g.solution_error()
        g.solution_error_CReAMa()

        # test result
        
        self.assertTrue(g.relative_error_strength < 1e-1)
        self.assertTrue(g.relative_error_strength < 1e-2)
        self.assertTrue((g.weighted_realisation()>=0).all())


    def test_CReAMa_Orignial_Newton_Emid(self):
        """test with 3 classes of cardinality 1
        and no zero degrees
        """
        __file__ = "network.txt"
        __location__ = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__)))
        path = os.path.join(__location__,__file__)
        network = np.loadtxt(path,delimiter=';')
        network_bin = (network>0).astype(int)
    
        g = sample.DirectedGraph(adjacency=network)

        g.solve_tool(model='CReAMa', method='fixed-point', adjacency=network_bin,  max_steps=1000, verbose=False)

        g.solution_error_CReAMa()

        # test result
        self.assertTrue(g.relative_error_strength < 1e-1)
        self.assertTrue(g.relative_error_strength < 1e-2)
        self.assertTrue((g.weighted_realisation()>=0).all())


    def test_CReAMa_dcm_Newton_random_dense_20(self):
        """test with 3 classes of cardinality 1
        and no zero degrees
        """

        network = sample.random_weighted_matrix_generator_dense(n=20, sup_ext = 10, sym=False, seed=None)
        network_bin = (network>0).astype(int)
    
        g = sample.DirectedGraph(adjacency=network)

        g.solve_tool(model='CReAMa', method='fixed-point', adjacency='dcm',  max_steps=1000, verbose=False)

        g.solution_error()
        g.solution_error_CReAMa()

        # test result
        
        self.assertTrue(g.relative_error_strength < 1e-1)
        self.assertTrue(g.relative_error_strength < 1e-2)
        self.assertTrue((g.weighted_realisation()>=0).all())


    def test_CReAMa_dcm_Newton_random_dense_100(self):
        """test with 3 classes of cardinality 1
        and no zero degrees
        """

        network = sample.random_weighted_matrix_generator_dense(n=100, sup_ext = 10, sym=False, seed=None)
        network_bin = (network>0).astype(int)
    
        g = sample.DirectedGraph(adjacency=network)

        g.solve_tool(model='CReAMa', method='fixed-point', adjacency='dcm',  max_steps=1000, verbose=False)

        g.solution_error()
        g.solution_error_CReAMa()

        # test result
        
        self.assertTrue(g.relative_error_strength < 1e-1)
        self.assertTrue(g.relative_error_strength < 1e-2)
        self.assertTrue((g.weighted_realisation()>=0).all())


    def test_CReAMa_dcm_Newton_random_dense_1000(self):
        """test with 3 classes of cardinality 1
        and no zero degrees
        """

        network = sample.random_weighted_matrix_generator_dense(n=1000, sup_ext = 100, sym=False, seed=None)
        network_bin = (network>0).astype(int)
    
        g = sample.DirectedGraph(adjacency=network)

        g.solve_tool(model='CReAMa', method='fixed-point', adjacency='dcm',  max_steps=1000, verbose=False)

        g.solution_error()
        g.solution_error_CReAMa()

        # test result
        
        self.assertTrue(g.relative_error_strength < 1e-1)
        self.assertTrue(g.relative_error_strength < 1e-2)
        self.assertTrue((g.weighted_realisation()>=0).all())


    def test_CReAMa_dcm_Newton_random_20(self):
        """test with 3 classes of cardinality 1
        and no zero degrees
        """

        network = sample.random_weighted_matrix_generator_dense(n=20, sup_ext = 100, sym=False, seed=None)
        network_bin = (network>0).astype(int)
    
        g = sample.DirectedGraph(adjacency=network)

        g.solve_tool(model='CReAMa', method='fixed-point', adjacency='dcm',  max_steps=1000, verbose=False)

        g.solution_error()
        g.solution_error_CReAMa()

        # test result
        
        self.assertTrue(g.relative_error_strength < 1e-1)
        self.assertTrue(g.relative_error_strength < 1e-2)
        self.assertTrue((g.weighted_realisation()>=0).all())


    def test_CReAMa_dcm_Newton_random_100(self):
        """test with 3 classes of cardinality 1
        and no zero degrees
        """

        network = sample.random_weighted_matrix_generator_dense(n=100, sup_ext = 100, sym=False, seed=None)
        network_bin = (network>0).astype(int)
    
        g = sample.DirectedGraph(adjacency=network)

        g.solve_tool(model='CReAMa', method='fixed-point', adjacency='dcm',  max_steps=1000, verbose=False)

        g.solution_error()
        g.solution_error_CReAMa()

        # test result
        
        self.assertTrue(g.relative_error_strength < 1e-1)
        self.assertTrue(g.relative_error_strength < 1e-2)
        self.assertTrue((g.weighted_realisation()>=0).all())



    def test_CReAMa_dcm_Newton_random_1000(self):
        """test with 3 classes of cardinality 1
        and no zero degrees
        """

        network = sample.random_weighted_matrix_generator_dense(n=1000, sup_ext = 100, sym=False, seed=None)
        network_bin = (network>0).astype(int)
    
        g = sample.DirectedGraph(adjacency=network)

        g.solve_tool(model='CReAMa', method='fixed-point', adjacency='dcm',  max_steps=1000, verbose=False)

        g.solution_error()
        g.solution_error_CReAMa()

        # test result
        
        self.assertTrue(g.relative_error_strength < 1e-1)
        self.assertTrue(g.relative_error_strength < 1e-2)
        self.assertTrue((g.weighted_realisation()>=0).all())







if __name__ == '__main__':
    
    unittest.main()
    
