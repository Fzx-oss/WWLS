from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

import random
import numpy as np
import sys
sys.path.append("../WWLS")
import auxiliary_methods as aux



def compare(line_a, line_b):
    length = len(line_a)
    cnt1 = np.sum(line_a > line_b)
    cnt2 = np.sum(line_a == line_b)
    cnt3 = length - cnt1 - cnt2
    if cnt1 > cnt3:
        return 'a'
    elif cnt1 < cnt3:
        return 'b'
    else:
        return 'c'


# 10-CV for kernel svm and hyperparameter selection.
def kernel_svm_evaluation(all_matrices, 
                          classes,
                          WL_iterations,
                          num_repetitions=10,
                          C=[10 ** 3, 10 ** 2, 10 ** 1, 10 ** 0, 10 ** -1, 10 ** -2, 10 ** -3], 
                          gamma_list=[0.1, 0.01, 0.001, 0.0001],
                          all_std=False):

    gamma_list=[0.1, 0.01]
    print(f"gamma: {gamma_list}")
    print(f"C: {C}")
    
    # Acc. over all repetitions.
    test_accuracies_all = []
    # All acc. over all folds and repetitions.
    test_accuracies_complete = []

    np.random.seed(42)
    print(f'10 times 10-CV')

    kf = KFold(n_splits=10, shuffle=True)
    sfolder = StratifiedKFold(n_splits=10, shuffle=True)

    total = 0
    for i in range(num_repetitions):
        print(f'{i+1}-th repetition')
        # Test acc. over all folds.
        test_accuracies = []
        
        for Train_index, Test_index in sfolder.split(all_matrices[0], classes):
            
            best_gram_matrix = np.exp(-0.01 * all_matrices[0])
            best_gamma = gamma_list[0]
            best_iteration = 0
            
            results = np.zeros((len(all_matrices), len(gamma_list)))
            # Generate indices of validation and training data for each distance matrix
            for h, distance_matrix in enumerate(all_matrices):
                target_distances = distance_matrix[Test_index, :]
                target_distances = target_distances[:, Train_index]
                num_test_samples = len(target_distances)
                
                pre_val_index = []
                for row_index in range(num_test_samples):
                    col_indices = np.argsort(target_distances[row_index])
                    pre_val_index.append(col_indices[0:1])

                for j in range(1):
                    # Randomly select one index from each row
                    val_index = [random.choice(line) for line in pre_val_index]
                    # training
                    train_index = np.setdiff1d(list(range(len(Train_index))), val_index)
                    
                    c_train = classes[Train_index]
                    c_train = c_train[train_index]
                    c_val = classes[Train_index]
                    c_val = c_val[val_index]
                    
                    for gamma_index, gamma in enumerate(gamma_list):
                        gram_matrix = np.exp(-gamma*distance_matrix)
                        gram_matrix = aux.normalize_gram_matrix(gram_matrix)

                        train = gram_matrix[Train_index, :]
                        train = train[train_index, :]
                        train = train[:, Train_index]
                        train = train[:, train_index]
                        val = gram_matrix[Train_index, :]
                        val = val[val_index, :]
                        val = val[:, Train_index]
                        val = val[:, train_index]

                        clf = SVC(C=10, kernel="precomputed")
                        clf.fit(train, c_train)
                        val_acc = accuracy_score(c_val, clf.predict(val)) * 100.0
                        
                        results[h][gamma_index] = val_acc            
            
            length = len(results)
            cmp_list = np.zeros(length)
            for row_a in range(length):
                line_a = results[row_a]
                for row_b in range(row_a+1, length):
                    line_b = results[row_b]
                    res = compare(line_a, line_b)
                    if res == 'a':
                        cmp_list[row_a] += 1
                    elif res == 'b':
                        cmp_list[row_b] += 1
            
            best_iteration = np.argmax(cmp_list)
            idx = np.argmax(results[best_iteration])
            best_gamma = gamma_list[idx]
            
            
            print(f"| select paremeters: h = {WL_iterations[best_iteration]}, gamma = {best_gamma} ", end='')
            

            best_gram_matrix = np.exp(-best_gamma * all_matrices[best_iteration])

            # test accuracy.
            train = best_gram_matrix[Train_index, :]
            train = train[:, Train_index]
            test = best_gram_matrix[Test_index, :]
            test = test[:, Train_index]

            c_train = classes[Train_index]
            c_test = classes[Test_index]
            
            clf = SVC(C=10, kernel="precomputed")
            clf.fit(train, c_train)
            acc = accuracy_score(c_test, clf.predict(test)) * 100.0
            print(f"| test acc is {acc} | ")

            test_accuracies.append(acc)
            total += 1
            
            if all_std:
                test_accuracies_complete.append(acc)

        test_accuracies_all.append(float(np.array(test_accuracies).mean()))

    if all_std:
        return np.array(test_accuracies_all).mean(), np.array(test_accuracies_all).std(), \
                np.array(test_accuracies_complete).std()
    else:
        return np.array(test_accuracies_all).mean(), np.array(test_accuracies_all).std()


