# -----------------------------------------------------------------------------
# This file contains the WWLS class, which is used to call WLsubtree to build 
# WL subtrees, embedes graphs, calculate graph Wasserstein distance.
# -----------------------------------------------------------------------------

from tqdm import tqdm
from WLsubtree import WLsubtree
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.metrics.pairwise import pairwise_distances

import ot
import numpy as np



class WWLS:
    def __init__(self, max_iter=3):
        self.wls = WLsubtree(max_iter=max_iter)
        self.num_CST  = 0

    def embedding(self, no, graph):
        graph_feature = []
        for i, node_id in enumerate(graph.x):
            self.wls.input_graph(graph.adj_info, graph.att_info)
            self.wls.build_WLsubtree(node_id, graph.att_info[node_id])
            node_feature = self.wls.get_feature()
            graph_feature.append(node_feature)
            self.wls.clear()
        
        self.num_CST = self.wls.get_num_CST()
        return graph_feature

    def get_num_CST(self):
        return self.num_CST

    def to_sparse(self, graph_features):
        sparse_graph_features = []
        for gf in tqdm(graph_features, desc='Converting to sparse graph features'):
            x_i = []
            y_j = []
            values = []
            for i, nf in enumerate(gf):
                for key, value in nf.items():
                    x_i.append(i)
                    y_j.append(key)
                    values.append(value)
            sparse_gf = coo_matrix((values, (x_i, y_j)), shape=(i+1, self.num_CST))
            sparse_graph_features.append(sparse_gf)
        return sparse_graph_features


    def compute_Wasserstein(self, C, mode='e'):        
        m = C.shape[0]
        n = C.shape[1]
        a = np.ones(m) / m
        b = np.ones(n) / n
        if mode == 'e': # emd
            dis = ot.emd2(a, b, C)
        elif mode == 's': # sinkhorn
            dis = ot.sinkhorn2(a, b, C, reg=10, numItermax=50)
        elif mode == 'r': # relax OT
            R_a = np.amin(C, axis=1).sum() / m
            R_b = np.amin(C, axis=0).sum() / n
            dis = max(R_a, R_b)
        return dis


    def compute_distance_matrices(self, sparse_graph_features, num_graphs):
        D = np.zeros((num_graphs, num_graphs))
        bar = tqdm(total = int(num_graphs * (num_graphs-1) / 2), desc='Computing graph Wasserstein distances')
        for i in range(num_graphs):
            sparse_gf1 = sparse_graph_features[i]
            for j in range(i+1,num_graphs):
                sparse_gf2 = sparse_graph_features[j]
                cost_matrix = pairwise_distances(sparse_gf1, sparse_gf2, metric='l1')
                D[i][j] = self.compute_Wasserstein(cost_matrix, mode='e')
                bar.update(1)
        
        D = D + D.T
        return D
