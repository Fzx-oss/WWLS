

def data_generator(y, distance_matrix, Train_index, Test_index):
    """Divide the dataset into y_train, y_test, X_train and X_test

    Args:
        y ([type]): [description]
        distance_matrix ([type]): [description]
        Train_index ([type]): [description]
        Test_index ([type]): [description]

    Returns:
        [type]: [description]
    """
    y_train = [y[idx] for idx in Train_index]
    y_test = [y[idx] for idx in Test_index]

    X_train = []
    for train_index1 in Train_index:
        Traindata = []
        for train_index2 in Train_index:
            Traindata.append(distance_matrix[train_index1][train_index2])
        X_train.append(Traindata)
    
    X_test = []
    for test_index in Test_index:
        Testdata = []
        for train_index in Train_index:
            Testdata.append(distance_matrix[test_index][train_index])
        X_test.append(Testdata)
    
    return X_train, y_train, X_test, y_test





