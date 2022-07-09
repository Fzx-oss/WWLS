# -----------------------------------------------------------------------------
# This file contains the Graph class, which is used to convert the PyG format 
# to the format we need in GOT-WL.
# -----------------------------------------------------------------------------
import sys
import numpy as np
from platform import node
from collections import defaultdict




class Graph(object):
    """A data object describing a homogeneous graph.
    This class converts PyG data objects into the format we need in WWLS.

    Variables:
        edge_index (list): Graph connectivity in COO format with shape :obj:`[2, num_edges]`.
        x (list): Convert Node feature matrix (vector of one hot vectors) to 
                    categorical label vector with shape :obj:`[num_nodes, 1]`.
        y (int): Graph label.
        adj_info (dictionary): Store adjacent nodes of each node in the format  `node id:  adj node id` .
        attInfo (dictionary): Store labels of each node in the format `node id: its label`.
    """

    def __init__(self, data, cleaned=False):
        # link information
        print(data)
        edge_index = data.edge_index.tolist()
        print(edge_index)

        # node list
        # self.x = list(set(edge_index[0]))
        self.x = list(range(data.x.shape[0]))

        # graph label
        self.y = data.y.tolist()[0] 

        # adjacent node information
        self.adj_info = defaultdict(list)
        for node1, node2 in zip(edge_index[0], edge_index[1]):
            self.adj_info[node1].append(node2)
        for key in self.adj_info:
            self.adj_info[key].sort()

        # node label information
        categorical_labels = np.argmax(data.x.tolist(), axis=1)
        self.att_info = dict(zip(self.x, categorical_labels))
        
        # isolated node
        if cleaned == False:
            isolated_nodes =  list(set(range(data.x.shape[0])) - set(self.x))
            if len(isolated_nodes) != 0:
                for node in isolated_nodes:
                    self.adj_info[node]
