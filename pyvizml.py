# -*- coding: utf-8 -*-
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

__author__ = '{Yao-Jen Kuo}'
__copyright__ = 'Copyright {2020}, {py-viz-ml-book}'
__license__ = '{MIT}'
__version__ = '{1}.{0}.{0}'
__maintainer__ = '{Yao-Jen Kuo}'
__email__ = '{tonykuoyj@gmail.com}'

class CreateNBAData:
    """
    This class scrapes NBA.com offical api: data.nba.net.
    See https://data.nba.net/10s/prod/v1/today.json
    Args:
        season_year (int): Use the first year to specify season, e.g. specify 2019 for the 2019-2020 season.
    """
    def __init__(self, season_year):
        self._season_year = str(season_year)

    def create_players_df(self):
        """
        This function returns the DataFrame of player information.
        """
        request_url = "https://data.nba.net/prod/v1/{}/players.json".format(self._season_year)
        resp_dict = requests.get(request_url).json()
        players_list = resp_dict['league']['standard']
        players_list_dict = []
        print("Creating players df...")
        for p in players_list:
            player_dict = {}
            for k, v in p.items():
                if isinstance(v, str) or isinstance(v, bool):
                    player_dict[k] = v
            players_list_dict.append(player_dict)
        df = pd.DataFrame(players_list_dict)
        filtered_df = df[(df['isActive']) & (df['heightMeters'] != '')]
        filtered_df = filtered_df.reset_index(drop=True)
        self._person_ids = filtered_df['personId'].values
        return filtered_df

    def create_stats_df(self):
        """
        This function returns the DataFrame of player career statistics.
        """
        self.create_players_df()
        career_summaries = []
        print("Creating player stats df...")
        for pid in self._person_ids:
            request_url = "https://data.nba.net/prod/v1/{}/players/{}_profile.json".format(self._season_year, pid)
            response = requests.get(request_url)
            profile_json = response.json()
            career_summary = profile_json['league']['standard']['stats']['careerSummary']
            career_summaries.append(career_summary)
        stats_df = pd.DataFrame(career_summaries)
        stats_df.insert(0, 'personId', self._person_ids)
        return stats_df
    
    def create_player_stats_df(self):
        """
        This function returns the DataFrame merged from players_df and stats_df.
        """
        players = self.create_players_df()
        stats = self.create_stats_df()
        player_stats = pd.merge(players, stats, left_on='personId', right_on='personId')
        return player_stats
    
class ImshowSubplots:
    """
    This class plots 2d-arrays with subplots.
    Args:
        rows (int): The number of rows of axes.
        cols (int): The number of columns of axes.
        fig_size (tuple): Figure size.
    """
    def __init__(self, rows, cols, fig_size):
        self._rows = rows
        self._cols = cols
        self._fig_size = fig_size
    def im_show(self, X, y, label_dict=None):
        """
        This function plots 2d-arrays with subplots.
        Args:
            X (ndarray): 2d-arrays.
            y (ndarray): Labels for 2d-arrays.
            label_dict (dict): Str labels for y if any.
        """
        n_pics = self._rows*self._cols
        first_n_pics = X[:n_pics, :, :]
        first_n_labels = y[:n_pics]
        fig, axes = plt.subplots(self._rows, self._cols, figsize=self._fig_size)
        for i in range(n_pics):
            row_idx = i % self._rows
            col_idx = i // self._rows
            axes[row_idx, col_idx].imshow(first_n_pics[i], cmap="Greys")
            if label_dict is not None:
                axes[row_idx, col_idx].set_title("Label: {}".format(label_dict[first_n_labels[i]]))
            else:
                axes[row_idx, col_idx].set_title("Label: {}".format(first_n_labels[i]))
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])
        plt.tight_layout()
        plt.show()
        
class NormalEquation:
    """
    This class defines the Normal equation for linear regression.
    Args:
        fit_intercept (bool): Whether to add intercept for this model.
    """
    def __init__(self, fit_intercept=True):
        self._fit_intercept = fit_intercept
    def fit(self, X_train, y_train):
        """
        This function uses Normal equation to solve for weights of this model.
        Args:
            X_train (ndarray): 2d-array for feature matrix of training data.
            y_train (ndarray): 1d-array for target vector of training data.
        """
        self._X_train = X_train.copy()
        self._y_train = y_train.copy()
        m = self._X_train.shape[0]
        if self._fit_intercept:
            X0 = np.ones((m, 1), dtype=float)
            self._X_train = np.concatenate([X0, self._X_train], axis=1)
        X_train_T = np.transpose(self._X_train)
        left_matrix = np.dot(X_train_T, self._X_train)
        right_matrix = np.dot(X_train_T, self._y_train)
        left_matrix_inv = np.linalg.inv(left_matrix)
        w = np.dot(left_matrix_inv, right_matrix)
        w_ravel = w.ravel().copy()
        self._w = w
        self.intercept_ = w_ravel[0]
        self.coef_ = w_ravel[1:]
    def predict(self, X_test):
        """
        This function returns predicted values with weights of this model.
        Args:
            X_test (ndarray): 2d-array for feature matrix of test data.
        """
        self._X_test = X_test.copy()
        m = self._X_test.shape[0]
        if self._fit_intercept:
            X0 = np.ones((m, 1), dtype=float)
            self._X_test = np.concatenate([X0, self._X_test], axis=1)
        y_pred = np.dot(self._X_test, self._w)
        return y_pred
    
class GradientDescent:
    """
    This class defines the vanilla gradient descent algorithm for linear regression.
    Args:
        fit_intercept (bool): Whether to add intercept for this model.
    """
    def __init__(self, fit_intercept=True):
        self._fit_intercept = fit_intercept
    def find_gradient(self):
        """
        This function returns the gradient given certain model weights.
        """
        y_hat = np.dot(self._X_train, self._w)
        gradient = (2/self._m) * np.dot(self._X_train.T, y_hat - self._y_train)
        return gradient
    def mean_squared_error(self):
        """
        This function returns the mean squared error given certain model weights.
        """
        y_hat = np.dot(self._X_train, self._w)
        mse = ((y_hat - self._y_train).T.dot(y_hat - self._y_train)) / self._m
        return mse
    def fit(self, X_train, y_train, epochs=10000, learning_rate=0.001):
        """
        This function uses vanilla gradient descent to solve for weights of this model.
        Args:
            X_train (ndarray): 2d-array for feature matrix of training data.
            y_train (ndarray): 1d-array for target vector of training data.
            epochs (int): The number of iterations to update the model weights.
            learning_rate (float): The learning rate of gradient descent.
        """
        self._X_train = X_train.copy()
        self._y_train = y_train.copy()
        self._m = self._X_train.shape[0]
        if self._fit_intercept:
            X0 = np.ones((self._m, 1), dtype=float)
            self._X_train = np.concatenate([X0, self._X_train], axis=1)
        n = self._X_train.shape[1]
        self._w = np.random.rand(n)
        n_prints = 10
        print_iter = epochs // n_prints
        w_history = dict()
        for i in range(epochs):
            current_w = self._w.copy()
            w_history[i] = current_w
            mse = self.mean_squared_error()
            gradient = self.find_gradient()
            if i % print_iter == 0:
                print("epoch: {:6} - loss: {:.6f}".format(i, mse))
            self._w -= learning_rate*gradient
        w_ravel = self._w.copy().ravel()
        self.intercept_ = w_ravel[0]
        self.coef_ = w_ravel[1:]
        self._w_history = w_history
    def predict(self, X_test):
        """
        This function returns predicted values with weights of this model.
        Args:
            X_test (ndarray): 2d-array for feature matrix of test data.
        """
        self._X_test = X_test
        m = self._X_test.shape[0]
        if self._fit_intercept:
            X0 = np.ones((m, 1), dtype=float)
            self._X_test = np.concatenate([X0, self._X_test], axis=1)
        y_pred = np.dot(self._X_test, self._w)
        return y_pred
    
class AdaGrad(GradientDescent):
    """
    This class defines the Adaptive Gradient Descent algorithm for linear regression.
    """
    def fit(self, X_train, y_train, epochs=10000, learning_rate=0.01, epsilon=1e-06):
        self._X_train = X_train.copy()
        self._y_train = y_train.copy()
        self._m = self._X_train.shape[0]
        if self._fit_intercept:
            X0 = np.ones((self._m, 1), dtype=float)
            self._X_train = np.concatenate([X0, self._X_train], axis=1)
        n = self._X_train.shape[1]
        self._w = np.random.rand(n)
        # 初始化 ssg
        ssg = np.zeros(n, dtype=float)
        n_prints = 10
        print_iter = epochs // n_prints
        w_history = dict()
        for i in range(epochs):
            current_w = self._w.copy()
            w_history[i] = current_w
            mse = self.mean_squared_error()
            gradient = self.find_gradient()
            ssg += gradient**2
            ada_grad = gradient / (epsilon + ssg**0.5)
            if i % print_iter == 0:
                print("epoch: {:6} - loss: {:.6f}".format(i, mse))
            # 以 adaptive gradient 更新 w
            self._w -= learning_rate*ada_grad
        w_ravel = self._w.copy().ravel()
        self.intercept_ = w_ravel[0]
        self.coef_ = w_ravel[1:]
        self._w_history = w_history
        
class LogitReg:
    """
    This class defines the vanilla descent algorithm for logistic regression.
    Args:
        fit_intercept (bool): Whether to add intercept for this model.
    """
    def __init__(self, fit_intercept=True):
        self._fit_intercept = fit_intercept
    def sigmoid(self, X):
        """
        This function returns the Sigmoid output as a probability given certain model weights.
        """
        X_w = np.dot(X, self._w)
        p_hat = 1 / (1 + np.exp(-X_w))
        return p_hat
    def find_gradient(self):
        """
        This function returns the gradient given certain model weights.
        """
        m = self._m
        p_hat = self.sigmoid(self._X_train)
        X_train_T = np.transpose(self._X_train)
        gradient = (1/m) * np.dot(X_train_T, p_hat - self._y_train)
        return gradient
    def cross_entropy(self, epsilon=1e-06):
        """
        This function returns the cross entropy given certain model weights.
        """
        m = self._m
        p_hat = self.sigmoid(self._X_train)
        cost_y1 = -np.dot(self._y_train, np.log(p_hat + epsilon))
        cost_y0 = -np.dot(1 - self._y_train, np.log(1 - p_hat + epsilon))
        cross_entropy = (cost_y1 + cost_y0) / m
        return cross_entropy
    def fit(self, X_train, y_train, epochs=10000, learning_rate=0.001):
        """
        This function uses vanilla gradient descent to solve for weights of this model.
        Args:
            X_train (ndarray): 2d-array for feature matrix of training data.
            y_train (ndarray): 1d-array for target vector of training data.
            epochs (int): The number of iterations to update the model weights.
            learning_rate (float): The learning rate of gradient descent.
        """
        self._X_train = X_train.copy()
        self._y_train = y_train.copy()
        m = self._X_train.shape[0]
        self._m = m
        if self._fit_intercept:
            X0 = np.ones((self._m, 1), dtype=float)
            self._X_train = np.concatenate([X0, self._X_train], axis=1)
        n = self._X_train.shape[1]
        self._w = np.random.rand(n)
        n_prints = 10
        print_iter = epochs // n_prints
        for i in range(epochs):
            cross_entropy = self.cross_entropy()
            gradient = self.find_gradient()
            if i % print_iter == 0:
                print("epoch: {:6} - loss: {:.6f}".format(i, cross_entropy))
            self._w -= learning_rate*gradient
        w_ravel = self._w.ravel().copy()
        self.intercept_ = w_ravel[0]
        self.coef_ = w_ravel[1:].reshape(1, -1)
    def predict_proba(self, X_test):
        """
        This function returns predicted probability with weights of this model.
        Args:
            X_test (ndarray): 2d-array for feature matrix of test data.
        """
        m = X_test.shape[0]
        if self._fit_intercept:
            X0 = np.ones((m, 1), dtype=float)
            self._X_test = np.concatenate([X0, X_test], axis=1)
        p_hat_1 = self.sigmoid(self._X_test).reshape(-1, 1)
        p_hat_0 = 1 - p_hat_1
        proba = np.concatenate([p_hat_0, p_hat_1], axis=1)
        return proba
    def predict(self, X_test):
        """
        This function returns predicted label with weights of this model.
        Args:
            X_test (ndarray): 2d-array for feature matrix of test data.
        """
        proba = self.predict_proba(X_test)
        y_pred = np.argmax(proba, axis=1)
        return y_pred
    
class ClfMetrics:
    """
    This class calculates some of the metrics of classifier including accuracy, precision, recall, f1 according to confusion matrix.
    Args:
        y_true (ndarray): 1d-array for true target vector.
        y_pred (ndarray): 1d-array for predicted target vector.
    """
    def __init__(self, y_true, y_pred):
        self._y_true = y_true
        self._y_pred = y_pred
    def confusion_matrix(self):
        """
        This function returns the confusion matrix given true/predicted target vectors.
        """
        n_unique = np.unique(self._y_true).size
        cm = np.zeros((n_unique, n_unique), dtype=int)
        for i in range(n_unique):
            for j in range(n_unique):
                n_obs = np.sum(np.logical_and(self._y_true == i, self._y_pred == j))
                cm[i, j] = n_obs
        self._tn = cm[0, 0]
        self._tp = cm[1, 1]
        self._fn = cm[1, 0]
        self._fp = cm[0, 1]
        return cm
    def accuracy_score(self):
        """
        This function returns the accuracy score given true/predicted target vectors.
        """
        cm = self.confusion_matrix()
        accuracy = (self._tn + self._tp) / np.sum(cm)
        return accuracy
    def precision_score(self):
        """
        This function returns the precision score given true/predicted target vectors.
        """
        precision = self._tp / (self._tp + self._fp)
        return precision  
    def recall_score(self):
        """
        This function returns the recall score given true/predicted target vectors.
        """
        recall = self._tp / (self._tp + self._fn)
        return recall
    def f1_score(self, beta=1):
        """
        This function returns the f1 score given true/predicted target vectors.
        Args:
            beta (int, float): Can be used to generalize from f1 score to f score.
        """
        precision = self.precision_score()
        recall = self.recall_score()
        f1 = (1 + beta**2)*precision*recall / ((beta**2 * precision) + recall)
        return f1

class DeepLearning:
    """
    This class defines the vanilla optimization of a deep learning model.
    Args:
        layer_of_units (list): A list to specify the number of units in each layer.
    """
    def __init__(self, layer_of_units):
        self._n_layers = len(layer_of_units)
        parameters = {}
        for i in range(self._n_layers - 1):
            parameters['W{}'.format(i + 1)] = np.random.rand(layer_of_units[i + 1], layer_of_units[i])
            parameters['B{}'.format(i + 1)] = np.random.rand(layer_of_units[i + 1], 1)
        self._parameters = parameters
    def sigmoid(self, Z):
        """
        This function returns the Sigmoid output.
        Args:
            Z (ndarray): The multiplication of weights and output from previous layer.
        """
        return 1/(1 + np.exp(-Z))
    def single_layer_forward_propagation(self, A_previous, W_current, B_current):
        """
        This function returns the output of a single layer of forward propagation.
        Args:
            A_previous (ndarray): The Sigmoid output from previous layer.
            W_current (ndarray): The weights of current layer.
            B_current (ndarray): The bias of current layer.
        """
        Z_current = np.dot(W_current, A_previous) + B_current
        A_current = self.sigmoid(Z_current)
        return A_current, Z_current
    def forward_propagation(self):
        """
        This function returns the output of a complete round of forward propagation.
        """
        self._m = self._X_train.shape[0]
        X_train_T = self._X_train.copy().T
        cache = {}
        A_current = X_train_T
        for i in range(self._n_layers - 1):
            A_previous = A_current
            W_current = self._parameters["W{}".format(i + 1)]
            B_current = self._parameters["B{}".format(i + 1)]
            A_current, Z_current = self.single_layer_forward_propagation(A_previous, W_current, B_current)
            cache["A{}".format(i)] = A_previous
            cache["Z{}".format(i + 1)] = Z_current
        self._cache = cache
        self._A_current = A_current
    def derivative_sigmoid(self, Z):
        """
        This function returns the output of the derivative of Sigmoid function.
        Args:
            Z (ndarray): The multiplication of weights, bias and output from previous layer.
        """
        sig = self.sigmoid(Z)
        return sig * (1 - sig)
    def single_layer_backward_propagation(self, dA_current, W_current, B_current, Z_current, A_previous):
        """
        This function returns the output of a single layer of backward propagation.
        Args:
            dA_current (ndarray): The output of the derivative of Sigmoid function from previous layer.
            W_current (ndarray): The weights of current layer.
            B_current (ndarray): The bias of current layer.
            Z_current (ndarray): The multiplication of weights, bias and output from previous layer.
            A_previous (ndarray): The Sigmoid output from previous layer.
        """
        dZ_current = dA_current * self.derivative_sigmoid(Z_current)
        dW_current = np.dot(dZ_current, A_previous.T) / self._m
        dB_current = np.sum(dZ_current, axis=1, keepdims=True) / self._m
        dA_previous = np.dot(W_current.T, dZ_current)
        return dA_previous, dW_current, dB_current
    def backward_propagation(self):
        """
        This function performs a complete round of backward propagation to update weights and bias.
        """
        gradients = {}
        self.forward_propagation()
        Y_hat = self._A_current.copy()
        Y_train = self._y_train.copy().reshape(1, self._m)
        dA_previous = - (np.divide(Y_train, Y_hat) - np.divide(1 - Y_train, 1 - Y_hat))
        for i in reversed(range(self._n_layers - 1)):
            dA_current = dA_previous
            A_previous = self._cache["A{}".format(i)]
            Z_current = self._cache["Z{}".format(i+1)]
            W_current = self._parameters["W{}".format(i+1)]
            B_current = self._parameters["B{}".format(i+1)]
            dA_previous, dW_current, dB_current = self.single_layer_backward_propagation(dA_current, W_current, B_current, Z_current, A_previous)
            gradients["dW{}".format(i + 1)] = dW_current
            gradients["dB{}".format(i + 1)] = dB_current
        self._gradients = gradients
    def cross_entropy(self):
        """
        This function returns the cross entropy given weights and bias.
        """
        Y_hat = self._A_current.copy()
        self._Y_hat = Y_hat
        Y_train = self._y_train.copy().reshape(1, self._m)
        ce = -1 / self._m * (np.dot(Y_train, np.log(Y_hat).T) + np.dot(1 - Y_train, np.log(1 - Y_hat).T))
        return ce[0, 0]
    def accuracy_score(self):
        """
        This function returns the accuracy score given weights and bias.
        """
        p_pred = self._Y_hat.ravel()
        y_pred = np.where(p_pred > 0.5, 1, 0)
        y_true = self._y_train
        accuracy = (y_pred == y_true).sum() / y_pred.size
        return accuracy
    def gradient_descent(self):
        """
        This function performs vanilla gradient descent to update weights and bias.
        """
        for i in range(self._n_layers - 1):
            self._parameters["W{}".format(i + 1)] -= self._learning_rate * self._gradients["dW{}".format(i + 1)]
            self._parameters["B{}".format(i + 1)] -= self._learning_rate * self._gradients["dB{}".format(i + 1)]
    def fit(self, X_train, y_train, epochs=100000, learning_rate=0.001):
        """
        This function uses multiple rounds of forward propagations and backward propagations to optimize weights and bias.
        Args:
            X_train (ndarray): 2d-array for feature matrix of training data.
            y_train (ndarray): 1d-array for target vector of training data.
            epochs (int): The number of iterations to update the model weights.
            learning_rate (float): The learning rate of gradient descent.
        """
        self._X_train = X_train.copy()
        self._y_train = y_train.copy()
        self._learning_rate = learning_rate
        loss_history = []
        accuracy_history = []
        n_prints = 10
        print_iter = epochs // n_prints
        for i in range(epochs):
            self.forward_propagation()
            ce = self.cross_entropy()
            accuracy = self.accuracy_score()
            loss_history.append(ce)
            accuracy_history.append(accuracy)
            self.backward_propagation()
            self.gradient_descent()
            if i % print_iter == 0:
                print("Iteration: {:6} - cost: {:.6f} - accuracy: {:.2f}%".format(i, ce, accuracy * 100))
        self._loss_history = loss_history
        self._accuracy_history = accuracy_history
    def predict_proba(self, X_test):
        """
        This function returns predicted probability for class 1 with weights of this model.
        Args:
            X_test (ndarray): 2d-array for feature matrix of test data.
        """
        X_test_T = X_test.copy().T
        A_current = X_test_T
        for i in range(self._n_layers - 1):
            A_previous = A_current
            W_current = self._parameters["W{}".format(i + 1)]
            B_current = self._parameters["B{}".format(i + 1)]
            A_current, Z_current = self.single_layer_forward_propagation(A_previous, W_current, B_current)
            self._cache["A{}".format(i)] = A_previous
            self._cache["Z{}".format(i + 1)] = Z_current
        p_hat_1 = A_current.copy().ravel()
        return p_hat_1
    def predict(self, X_test):
        p_hat_1 = self.predict_proba(X_test)
        return np.where(p_hat_1 >= 0.5, 1, 0)