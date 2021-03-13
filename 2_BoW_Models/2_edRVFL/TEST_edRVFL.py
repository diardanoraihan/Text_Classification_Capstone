#%%

from __future__ import print_function

import numpy as np
from edRVFL import tune_RVFL, calculate_acc
from Utils import to_categorical
import scipy.io

np.random.seed(813306)


# Function to collect the dataset
def read_mat_data(problem_path):
    mat_data = scipy.io.loadmat(problem_path + '_R.mat')
    x_data_mat = mat_data['dataX']
    y_data_mat = mat_data['dataY'].reshape(mat_data['dataY'].shape[0], )
    train_idx_temp = mat_data['TRAIN_idx']
    test_idx_temp = mat_data['TEST_idx']

    train_idx = np.zeros([train_idx_temp.size, train_idx_temp[0][0].size], dtype=np.int)
    test_idx = np.zeros([test_idx_temp.size, test_idx_temp[0][0].size], dtype=np.int)
    for k in range(train_idx_temp.size):
        train_idx[k, :] = (train_idx_temp[k][0] - 1).reshape(train_idx_temp[k][0].size)
        test_idx[k, :] = (test_idx_temp[k][0] - 1).reshape(test_idx_temp[k][0].size)

    return x_data_mat, y_data_mat, train_idx, test_idx


def read_tune_data(problem_path):
    mat_data = scipy.io.loadmat(problem_path + '_Tune.mat')
    train_tune_temp = mat_data['train_tune']
    test_tune_temp = mat_data['test_tune']

    train_tune = np.zeros(train_tune_temp.size, dtype=np.object)
    test_tune = np.zeros(test_tune_temp.size, dtype=np.object)
    for k in range(train_tune_temp.size):
        train_tune[k] = train_tune_temp[k, 0].astype(np.bool)
        test_tune[k] = test_tune_temp[k, 0].astype(np.bool)

    return train_tune, test_tune


# Settings
input_dir = '../TSC Problems/'
nFolds = 1
processor = "GPU"
GPU = ""
flist = ['Adiac']

for each in flist:        # For each dataset
    # Initialisation
    Train_Acc = np.zeros([nFolds, 1])
    Test_Acc = np.zeros([nFolds, 1])
    Train_Time = np.zeros([nFolds, 1])
    Test_Time = np.zeros([nFolds, 1])
    nb_classes = 0
    batch_size = 0

    # Collects dataset
    file_path = input_dir + each + '/' + each
    x_data, y_data, TRAIN_idx, TEST_idx = read_mat_data(file_path)
    TRAIN_Tune_idx, TEST_Tune_idx = read_tune_data(file_path)

    y_data, _ = to_categorical(y_data)

    kFold = TRAIN_Tune_idx[0].shape[0]

    for i in range(nFolds):          # For each experimental fold
        # Splits into training and testing data
        x_train = x_data[TRAIN_idx[i, :], :]
        y_train = y_data[TRAIN_idx[i, :], :]
        x_test = x_data[TEST_idx[i, :], :]
        y_test = y_data[TEST_idx[i, :], :]

        nb_classes = len(np.unique(y_test))  # Number of Classes

        # Label normalisation - set values to 0 ~ Number of classes
        y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min()) * (
                nb_classes - 1)  # Label Normalisation
        y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min()) * (nb_classes - 1)  # Label Normalisation

        # Normalise x_train
        x_train_mean = x_train.mean(0)
        x_train_std = x_train.std(0)
        x_train = (x_train - x_train_mean) / x_train_std
        # Normalise x_test (Note: uses mean and std of training samples)
        x_test = (x_test - x_train_mean) / x_train_std

        x_train_val = np.zeros(kFold, dtype=np.object)
        y_train_val = np.zeros(kFold, dtype=np.object)
        x_test_val = np.zeros(kFold, dtype=np.object)
        y_test_val = np.zeros(kFold, dtype=np.object)
        for j in range(kFold):
            x_train_val[j] = train_feas_tune[i, j, 2]
            y_train_val[j] = y_train[TRAIN_Tune_idx[i][j, :], :]
            x_test_val[j] = test_feas_tune[i, j, 2]
            y_test_val[j] = y_train[TEST_Tune_idx[i][j, :], :]

        # TRAINING
        ModelT, Val_Acc, Results = tune_RVFL(x_train_val, y_train_val, x_test_val, y_test_val)

        # TRAINING
        Train_Time[i] = ModelT.train(x_train, y_train)
        Train_Scores = ModelT.predict(x_train)[0]
        Train_Acc[i] = calculate_acc(Train_Scores, y_train)

        # TESTING
        Test_Scores, Test_Time[i] = ModelT.predict(x_test)
        Test_Acc[i] = calculate_acc(Test_Scores, y_test)

        # Displays results for each fold
        print('Run', i, 'Train Accuracy =', Train_Acc[i, 0], 'Test Accuracy =', Test_Acc[i, 0],
              'Training Time =', Train_Time[i, 0], 's', end="\n")

    # Calculate mean results
    MeanTrainAcc = Train_Acc.mean()
    MeanTestAcc = Test_Acc.mean()
    MeanTrainTime = Train_Time.mean()

    # Displays mean results
    print('Mean Train Accuracy =', MeanTrainAcc, 'Mean Test Accuracy =', MeanTestAcc)

# %%
