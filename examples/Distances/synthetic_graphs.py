from networkx.generators.random_graphs import erdos_renyi_graph
from networkx.generators.classic import cycle_graph
from networkx import grid_graph
from torch_geometric.utils.convert import from_networkx

import networkx as nx
import random
import igraph as ig
import numpy as np
import copy



# https://stackoverflow.com/questions/42591549/add-and-delete-a-random-edge-in-networkx
def random_edge(graph, del_orig=True, verbose=False):
    '''
    Create a new random edge and delete one of its current edge if del_orig is True.
    :param graph: networkx graph
    :param del_orig: bool
    :return: networkx graph
    '''
    edges = list(graph.edges)
    nonedges = list(nx.non_edges(graph))

    chosen_edge = random.choice([x for x in edges if graph.degree[x[0]] < graph.number_of_nodes()])
    chosen_nonedge = random.choice([x for x in nonedges if chosen_edge[0] == x[0] or chosen_edge[0] == x[1]])

    if del_orig:
        # delete chosen edge
        graph.remove_edge(chosen_edge[0], chosen_edge[1])
        if verbose:
            print(f'remove {chosen_edge[0]} -- {chosen_edge[1]}')
    # add new edge
    graph.add_edge(chosen_nonedge[0], chosen_nonedge[1])
    if verbose:
        print(f'add {chosen_nonedge[0]} -- {chosen_nonedge[1]}')

    return graph



def generate_synthetic_graphs(graph_type='random', remove=1):
    np.random.seed(42)
    random.seed(42)
    
    # G = nx.karate_club_graph() # sample graph (undirected, unweighted)
    if graph_type == 'random':
        G = erdos_renyi_graph(50, 0.5)
    elif graph_type == 'cycle':
        G = cycle_graph(50)
    elif graph_type == 'grid':
        G = grid_graph(dim=(5,5,2))
    
    ig_graphs = []
    py_graphs = []
    nx_graphs = []
    for i in range(100):
        
        if remove == 0:
            G = random_edge(G)
        else:
            G = random_edge(G, del_orig=False)
        
        # print(f'{i+1}-th graph {G}')
        
        ig_graph = ig.Graph.from_networkx(copy.deepcopy(G))
        ig_graphs.append(ig_graph)
        
        py_graph = from_networkx(copy.deepcopy(G))
        py_graphs.append(py_graph)
        
        nx_graphs.append(copy.deepcopy(G))
    
    return ig_graphs, py_graphs, nx_graphs
