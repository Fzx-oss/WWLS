from grakel.datasets import fetch_dataset
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from sklearn.model_selection import train_test_split


class WLkernel:
    def __init__(self, h):
        self.wlk = WeisfeilerLehman(n_iter=h, base_graph_kernel=VertexHistogram, normalize=True)
    
    def fit_transform(self, data):
        return self.wlk.fit_transform(data)
