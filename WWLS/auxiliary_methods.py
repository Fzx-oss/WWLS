import math as m

import numpy as np
from scipy import sparse as sp


# Cosine normalization for a gram matrix.
def normalize_gram_matrix(gram_matrix):
    n = gram_matrix.shape[0]
    gram_matrix_norm = np.zeros([n, n], dtype=np.float64)
    for i in range(0, n):
        for j in range(i, n):
            if not (gram_matrix[i][i] == 0.0 or gram_matrix[j][j] == 0.0):
                g = gram_matrix[i][j] / m.sqrt(gram_matrix[i][i] * gram_matrix[j][j])
                gram_matrix_norm[i][j] = g
                gram_matrix_norm[j][i] = g
    return gram_matrix_norm


# Cosine normalization for sparse feature vectors, i.e., \ell_2 normalization.
def normalize_feature_vector(feature_vectors):
    n = feature_vectors.shape[0]
    for i in range(0, n):
        norm = sp.linalg.norm(feature_vectors[i])
        feature_vectors[i] = feature_vectors[i] / norm
    return feature_vectors


""" A function to ensure gram matrix is psd """
def ensure_psd(K, tol=1e-8):
    # Helper function to remove negative eigenvalues
    # from numpy.linalg import eigh
    w, v = np.linalg.eigh(K)
    if (w<-tol).sum() >= 1:
        neg = np.argwhere(w<-tol)
        w[neg] = 0
        Xp = v.dot(np.diag(w)).dot(v.T)
        return Xp
    else:
        return K


""" k-Nearest Neighbor """
def kNN(x_test, y_train, knn_k):
    y_pred = []
    for X in x_test:
        y_indexes = np.argsort(X)
        res = {}
        for k in range(knn_k):
            pred = y_train[y_indexes[k]]
            if pred in res.keys():
                res[pred] += 1
            else:
                res[pred] = 1
        finpred = 0
        votes = 0
        for key in res.keys():
            if res[key] > votes:
                votes = res[key]
                finpred = key
        y_pred.append(finpred)
    return y_pred
