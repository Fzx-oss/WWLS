import os
import sys
import argparse
import numpy as np

sys.path.append("../WWLS")
# WWLS
from WWLS import WWLS
from graph_data import Graph
from dataset_loader import load_dataset
from data_generator import data_generator
import auxiliary_methods as aux

# others
from tqdm import tqdm
from sklearn.svm import SVC
from matplotlib.pyplot import table
from prettytable import PrettyTable
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold



if __name__ == "__main__":
    np.random.seed(42)

    # command line setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="MUTAG", help="dataset name")
    parser.add_argument("--h", default=2, type=int, help="max iteration number")
    parser.add_argument("--mode", default="s", type=str, help="s:SVM, k:k-NN")
    parser.add_argument("--C", default=10, type=float, help="parameter C of SVC")
    parser.add_argument("--gamma", default=0.1, type=float, help="parameter gamma of graph kernel")
    parser.add_argument("--k", default=1, type=int, help="parameter k of k-NN")
    parser.add_argument("--w", default=1, type=int, help="write cost matrices")
    args = parser.parse_args()

    table = PrettyTable()
    if args.mode == 's':
        table.field_names = ['Dataset', 'Number of itarations', 'Classifier', 'Gamma', 'C', 'w']
        table.add_row([args.dataset, args.h, 'SVM', args.gamma, args.C, args.w])
    else:
        table.field_names = ['Dataset', 'Number of itarations', 'Classifier', 'k', 'w']
        table.add_row([args.dataset, args.h, 'k-NN', args.k, args.w])
    print(f'Settings:\n{table}')

# %% -------------------------------------------------------------------------------------------------------
    # load dataset
    print("Loading dataset...")
    name = args.dataset
    dataset = load_dataset(name)
    print("Done!")
    graphs = [Graph(data) for data in dataset] # graph data
    
    y = [graph.y for graph in graphs] # graph labels
    num_graphs = len(y) # number of graphs


# %% -------------------------------------------------------------------------------------------------------
    dir_path = '../DistanceMatrices'
    os.makedirs(dir_path, exist_ok=True)
    npy_path = f'{dir_path}/{args.dataset}_{args.h}.npy'

    if os.path.exists(npy_path) == True:
        D = np.load(npy_path)
    else:
        # GOT-WL
        wwls = WWLS(args.h)
        graph_features = [wwls.embedding(i, graph) for i, graph in enumerate(tqdm(graphs, desc='Building WL subtrees'))] 
        num_CST = wwls.get_num_CST()
        print(f'Number of complete subtree is {num_CST}')
        sparse_graph_features = wwls.to_sparse(graph_features)    
        D = wwls.compute_distance_matrices(sparse_graph_features, num_graphs)
        if args.w == 1:
            np.save(npy_path, D)

    if args.mode == 's':
        Gram = np.exp(-args.gamma*D)
        Gram = aux.normalize_gram_matrix(Gram)
    else:
        Gram = D


# %% -------------------------------------------------------------------------------------------------------
    K_fold = 10
    total = 10 * K_fold
    sfolder = StratifiedKFold(n_splits=K_fold, shuffle=True)
    bar = tqdm(total=total, desc=f'Doing {total} times of graph classification experiments')
    all_acc = []
    mean_accuracies = []
    
    for i in range(10):
        accuracy_scores = []
        for train_index, test_index in sfolder.split(Gram, y):
            X_train, y_train, X_test, y_test = data_generator(y, Gram, train_index, test_index)
            
            if args.mode == 's':
                X_train = aux.ensure_psd(X_train)
                gs = SVC(C=args.C, kernel='precomputed').fit(X_train, y_train)
                y_pred = gs.predict(X_test)
            else:
                y_pred = aux.kNN(X_test, y_train, args.k)
            
            bar.update(1)
            acc = accuracy_score(y_test, y_pred)
            accuracy_scores.append(acc)
            all_acc.append(acc)
        mean_accuracies.append(np.mean(accuracy_scores))
    
    bar.close()
    print('Accuracy: {:2.2f} +- {:2.2f} or {:2.2f}'.format(
            np.mean(mean_accuracies) * 100, np.std(mean_accuracies) * 100, 
            np.std(all_acc) * 100))