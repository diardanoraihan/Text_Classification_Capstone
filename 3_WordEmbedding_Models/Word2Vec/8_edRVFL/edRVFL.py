import numpy as np
import time


class edRVFL:
    def __init__(self, L=10, N=100, C=0.1, scale=1, seed=0):
        self.L = L
        self.N = N
        self.C = C
        self.scale = scale
        self.w = []
        self.b = []
        self.beta = []
        self.mu = []
        self.sigma = []

        np.random.seed(seed)

    def train(self, x_train, y_train):
        start_time = time.perf_counter()

        n_sample = x_train.shape[0]
        beta = np.empty((self.L,), dtype=np.object)
        weights = np.empty((self.L,), dtype=np.object)
        biases = np.empty((self.L,), dtype=np.object)
        mu = np.empty((self.L,), dtype=np.object)
        sigma = np.empty((self.L,), dtype=np.object)

        a_input = x_train

        for i in range(self.L):

            w = self.scale * (2 * np.random.random([a_input.shape[1], self.N]) - 1)
            b = self.scale * np.random.random([1, self.N])

            a1 = np.dot(a_input, w) + b
            mu1 = a1.mean(0)
            sigma1 = a1.std(0)
            sigma1 = np.maximum(sigma1, 0.0001)  # for numerical stability
            a1 = (a1 - mu1) / sigma1

            a1 = relu(a1)

            a1_temp = np.concatenate((x_train, a1, np.ones((n_sample, 1))), axis=1)
            beta1 = l2weights(a1_temp, y_train, self.C)

            weights[i] = w
            biases[i] = b
            beta[i] = beta1
            mu[i] = mu1
            sigma[i] = sigma1

            a_input = np.concatenate((x_train, a1), axis=1)

        train_time = time.perf_counter() - start_time

        self.w = weights
        self.b = biases
        self.beta = beta
        self.mu = mu
        self.sigma = sigma

        return train_time

    def predict(self, x_test):
        n_sample = x_test.shape[0]
        n_layer = self.L
        beta = self.beta
        weights = self.w
        biases = self.b
        mu = self.mu
        sigma = self.sigma
        prob_scores = np.empty((n_layer,), dtype=np.object)
        n_class = beta[0].shape[1]

        start_time = time.perf_counter()
        a_input = x_test

        for i in range(n_layer):
            w = weights[i]
            b = biases[i]

            a1 = np.dot(a_input, w) + b
            mu1 = mu[i]
            sigma1 = sigma[i]
            a1 = (a1 - mu1) / sigma1

            a1 = relu(a1)

            a1_temp = np.concatenate((x_test, a1, np.ones((n_sample, 1))), axis=1)

            beta1 = beta[i]
            y_test_temp = a1_temp.dot(beta1)
            y_test_temp1 = y_test_temp - np.tile(y_test_temp.max(1), (n_class, 1)).transpose()
            num = np.exp(y_test_temp1)
            dem = num.sum(1)
            prob_scores_temp = num / np.tile(dem, (n_class, 1)).transpose()
            prob_scores[i] = prob_scores_temp

            a_input = np.concatenate((x_test, a1), axis=1)

        test_time = time.perf_counter() - start_time

        return prob_scores, test_time


def l2weights(x, y, c):
    [n_sample, n_feature] = x.shape

    if n_feature < n_sample:
        beta = np.linalg.pinv((np.eye(n_feature) / c + x.transpose().dot(x))).dot(x.transpose()).dot(y)

    else:
        beta = x.transpose().dot(np.linalg.pinv(np.eye(n_sample) / c + x.dot(x.transpose()))).dot(y)

    return beta


def relu(x):
    y = np.maximum(x, 0)

    return y


def sigmoid(x):
    y = 1 / (1 + np.exp(-1 * (x)))

    return y


def calculate_acc(prob_scores, y_test):
    sum_prob_scores = 0

    n_scores = len(prob_scores)
    for scores in prob_scores:
        sum_prob_scores = sum_prob_scores + scores

    mean_prob_scores = sum_prob_scores / n_scores

    correct_idx = np.argmax(y_test, 1)
    pred_idx = np.argmax(mean_prob_scores, 1)

    acc = np.mean(correct_idx == pred_idx)

    return acc


def tune_RVFL(x_train, y_train, x_test, y_test):
    n_range = np.array(range(3, 204, 20))
    c_range = 2. ** np.array(range(-5, 15))

    k_fold = len(x_train)

    r1_data_type = np.dtype(
        [('N', np.int), ('C', np.float), ('MeanValAcc', np.float)])
    results_t = np.empty((len(n_range), len(c_range)), dtype=r1_data_type)

    best_acc = 0
    best_n = n_range[0]
    best_c = c_range[0]

    for p1 in range(len(n_range)):
        for p2 in range(len(c_range)):
            test_acc = np.zeros((k_fold,))

            for k in range(k_fold):
                x_train_fold = x_train[k]
                y_train_fold = y_train[k]
                x_test_fold = x_test[k]
                y_test_fold = y_test[k]

                mu = x_train_fold.mean(0)
                sd = x_train_fold.std(0)

                x_train_fold = (x_train_fold - mu) / sd
                x_test_fold = (x_test_fold - mu) / sd

                model_temp = edRVFL(L=10, N=n_range[p1], C=c_range[p2])
                model_temp.train(x_train_fold, y_train_fold)
                scores = model_temp.predict(x_test_fold)[0]
                test_acc[k] = calculate_acc(scores, y_test_fold)

            mean_acc = test_acc.mean()

            if mean_acc > best_acc:
                best_acc = mean_acc
                best_n = n_range[p1]
                best_c = c_range[p2]

            results_t[p1, p2] = (n_range[p1], c_range[p2], mean_acc)

    results = results_t

    val_acc = best_acc

    model = edRVFL(L=10, N=best_n, C=best_c)

    return model, val_acc, results
