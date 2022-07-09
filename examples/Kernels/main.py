# -----------------------------------------------------------------------------
# This script runs the experiments reported in the WWL paper
#
# October 2019, M. Togninalli, E. Ghisu, B. Rieck
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import argparse
import os

from utilities import read_labels, custom_grid_search_cv
from WWL import *
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, help='Provide the dataset name (MUTAG or Enzymes)')
                            # choices=['MUTAG', 'ENZYMES', 'PTC_MR', 'PROTEINS', 'IMDB_BINARY'])
    parser.add_argument('--sinkhorn', default=False, action='store_true', help='Use sinkhorn approximation')
    parser.add_argument('--h', type = int, required=False, default=2, help = "(Max) number of WL iterations")

    args = parser.parse_args()
    
    np.random.seed(42)
    
    dataset = args.dataset
    h = args.h
    sinkhorn = args.sinkhorn
    
    print(f'The number of the iteration is {h}')
    print(f'Generating results for {dataset}...')
    #---------------------------------
    # Setup
    #---------------------------------
    # Start by making directories for intermediate and final files
    data_path = os.path.join('../data', dataset)
    output_path = os.path.join('output', dataset)
    results_path = os.path.join('results', dataset)
    
    for path in [output_path, results_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    #---------------------------------
    # Embeddings
    #---------------------------------
    # Load the data and generate the embeddings 
    embedding_type = 'discrete'
    print(f'Generating {embedding_type} embeddings for {dataset}.')
    label_sequences = compute_wl_embeddings_discrete(data_path, h)

    # Save embeddings to output folder
    out_name = f'{dataset}_wl_{embedding_type}_embeddings_h{h}.npy'
    np.save(os.path.join(output_path, out_name), label_sequences)
    print(f'Embeddings for {dataset} computed, saved to {os.path.join(output_path, out_name)}.')
    print()

    #---------------------------------
    # Wasserstein & Kernel computations
    #---------------------------------
    # Run Wasserstein distance computation
    print('Computing the Wasserstein distances...')
    wasserstein_distances = compute_wasserstein_distance(label_sequences, h, sinkhorn=sinkhorn, discrete=True)

    # Save Wasserstein distance matrices
    for i, D_w in enumerate(wasserstein_distances):
        filext = 'wasserstein_distance_matrix'
        if sinkhorn:
            filext += '_sinkhorn'
        filext += f'_it{i}.npy'
        np.save(os.path.join(output_path,filext), D_w)
    print('Wasserstein distances computation done. Saved to file.')
    print()

    # Transform to Kernel
    # Here the flags come into play
    gammas = [1, 0.1, 0.01, 0.001, 0.0001]
    
    # Load the dataset
    label_file = os.path.join(data_path, 'Labels.txt')
    y = np.array(read_labels(label_file))

    for i, current_h in enumerate(range(args.h+1)): # 
        if i == 0:
            continue
        
        # Generate the full list of kernel matrices from which to select
        print(f'current h is {i}')
        
        M = wasserstein_distances[current_h]
        kernel_matrices = []
        for g in gammas:
            K = np.exp(-g*M)
            kernel_matrices.append(K)

        #---------------------------------
        # Classification
        #---------------------------------
        cv = StratifiedKFold(n_splits=10, shuffle=True)
        
        mean_accuracies = []
        for _ in range(10): # 10
            accuracy_scores = []
            for train_index, test_index in cv.split(kernel_matrices[0], y): # 10
                best_acc = 0
                
                K_train = [K[train_index][:, train_index] for K in kernel_matrices]
                K_test  = [K[test_index][:, train_index] for K in kernel_matrices]
                y_train, y_test = y[train_index], y[test_index]

                for num in range(len(kernel_matrices)): # gammaの個数分ちがうグラム行列が存在する　
                    for c in [0.1, 1, 10, 100, 1000]:
                        gs = SVC(C=c, kernel='precomputed').fit(K_train[num], y_train)
                        y_pred = gs.predict(K_test[num])
                        acc = accuracy_score(y_test, y_pred)
                        if acc > best_acc:
                            best_acc = acc
                
                accuracy_scores.append(best_acc)
            
            mean_accuracies.append(np.mean(accuracy_scores))

        print('Accuracy: {:2.2f} +- {:2.2f}'.format(
            np.mean(mean_accuracies) * 100, np.std(mean_accuracies) * 100))

if __name__ == '__main__':
    main()