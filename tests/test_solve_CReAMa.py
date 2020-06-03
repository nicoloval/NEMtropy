import sys
import os
sys.path.append('../')
import Directed_graph_Class as sample
import numpy as np

if __name__ == '__main__':
    
    __file__ = "network.txt"
    __location__ = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__)))
    path = os.path.join(__location__,__file__)
    network = np.loadtxt(path,delimiter=';')
    
    g = sample.DirectedGraph(adjacency=network)

    g.solve_tool(model='CReAMa', method='quasinewton', adjacency='dcm',  max_steps=200, verbose=False)

    # g.solution_error_CReAMa()
    g.solution_error()
        
    print('error on the stregths using "dcm" as binary model and "quasinewton" as method:',g.error_strength)
    print('relative error on the stregths using "dcm" as binary model and "quasinewton" as method:',g.error_strength/np.sum(g.out_strength))
    print()

    g.solve_tool(model='CReAMa', method='quasinewton', adjacency=(network>0).astype(float), max_steps=200, verbose=False)

    # g.solution_error_CReAMa()
    g.solution_error()

    print('error on the stregths using original binary adjacency matrix and "quasinewton" as method:',g.error_strength)
    print('relative error on the stregths using original binary adjacency and "quasinewton" as method:',g.error_strength/np.sum(g.out_strength))
    print()


    g.solve_tool(model='CReAMa', method='fixed-point', adjacency='dcm',  max_steps=200, verbose=False)

    # g.solution_error_CReAMa()
    g.solution_error()
        
    print('error on the stregths using "dcm" as binary model and "fixed-point" as method:',g.error_strength)
    print('relative error on the stregths using "dcm" as binary model and "fixed-point" as method:',g.error_strength/np.sum(g.out_strength))
    print()

    g.solve_tool(model='CReAMa', method='fixed-point', adjacency=(network>0).astype(float), max_steps=200, verbose=False)

    # g.solution_error_CReAMa()
    g.solution_error()

    print('error on the stregths using original binary adjacency matrix and "fixed-point" as method:',g.error_strength)
    print('relative error on the stregths using original binary adjacency and "fixed-point" as method:',g.error_strength/np.sum(g.out_strength))
    print()
    
    g.clean_edges()
    
