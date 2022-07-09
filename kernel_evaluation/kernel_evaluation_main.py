import os
import sys
import argparse
import numpy as np

from tqdm import tqdm
from kernel_svm_evaluation import kernel_svm_evaluation
# WWLS
sys.path.append("../WWLS")
from WWLS import WWLS
from graph_data import Graph
from dataset_loader import load_dataset




def WWLS_distance(dataset, npy_path, h):
    graphs = [Graph(graph) for graph in dataset]
    wwls = WWLS(h)
    graph_features = [wwls.embedding(i, graph) for i, graph in enumerate(tqdm(graphs, desc='Building WL subtrees'))] 
    sparse_graph_features = wwls.to_sparse(graph_features)
    D = wwls.compute_distance_matrices(sparse_graph_features, len(graphs))
    np.save(npy_path, D)


def main():
    # command line setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="MUTAG", help="dataset name")
    args = parser.parse_args()

    # Number of repetitions of 10-CV.
    num_reps = 10

    print("Loading dataset...")
    dataset = load_dataset(args.dataset)
    print("Done!")
    graphs = [Graph(data) for data in dataset] # graph data
    y = np.array([graph.y for graph in graphs]) # graph labels

    all_matrices = []
    WL_iterations = {}
    for i, current_h in enumerate(range(2, 4)):
        dir_path = '../DistanceMatrices'
        os.makedirs(dir_path, exist_ok=True)
        npy_path = f'{dir_path}/{args.dataset}_{current_h}.npy'            
        if os.path.exists(npy_path) == True:
            print(f'Load {args.dataset}_{current_h}.npy')
            all_matrices.append(np.load(npy_path))
        else:
            print(f"Cannot find {args.dataset}_{current_h}.npy")
            WWLS_distance(dataset, npy_path, current_h)
            print(f'Load {args.dataset}_{current_h}.npy')
            all_matrices.append(np.load(npy_path))
        WL_iterations[i] = current_h
    
    
    acc, std1, std2 = kernel_svm_evaluation(all_matrices, y, WL_iterations, num_repetitions=num_reps, all_std=True)
    print(f"Acc: {acc} +- {std1} (or {std2})")


if __name__ == "__main__":
    main()
