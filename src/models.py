import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import copy
from collections.abc import Callable

class CrossValidation:
    def __init__(
        self,
        X : np.ndarray, 
        Y : np.ndarray, 
        epochs : int, 
        num_folds : int, 
        c_value : float,

        learning_rate_range : tuple[int, int], 
        batch_size_2_pow_values : list[int], 
        K_values : list[int],
        S_values : list[float], 
        b1_and_b2_values : list[tuple[float ,float]], 
        L2_values : list[float],
        M_values : list[list[int]] = [],
        h_values : list[list[int]] = [],
    ):
        self.X : np.ndarray = X
        self.Y : np.ndarray = Y
        self.epochs : int = epochs
        self.num_folds : int = num_folds
        self.learning_rate : tuple[int, int] = learning_rate_range
        self.c_value = c_value

        self.batch_size_2_pow_values : list[int] = batch_size_2_pow_values 
        self.K_values : list[int] = K_values
        self.S_values : list[float] = S_values
        self.b1_and_b2_values : list[tuple[float, float]] = b1_and_b2_values
        self.L2_values : list[float] = L2_values
        self.M_values : list[list[int]] = M_values
        self.h_values : list[list[str]] = h_values

    def evaluate_hiperparameters(self, M : list[int], h : list[str]) -> list[dict]:
        fold_size : int = len(self.X) // self.num_folds
        model_score : list[dict] = []
        indices = np.arange(len(self.X))
        print(f"Cantidad de iteraciones estimada: {len(self.batch_size_2_pow_values) * len(self.L2_values) * 2 * len(self.b1_and_b2_values) * len(self.K_values) * len(self.S_values)}")
        model_index : int = 0
        folds_x_train : list[np.ndarray] = []
        folds_y_train : list[np.ndarray] = []
        folds_x_val : list[np.ndarray] = []
        folds_y_val : list[np.ndarray] = []
        for fold in range(self.num_folds):
            val_indices = indices[fold * fold_size : (fold + 1) * fold_size]
            train_indices = np.setdiff1d(indices, val_indices, assume_unique=True)

            X_train, X_val = self.X[train_indices], self.X[val_indices]
            Y_train, Y_val = self.Y[train_indices], self.Y[val_indices]

            folds_x_train.append(self.X[train_indices])
            folds_y_train.append(self.Y[train_indices])
            folds_x_val.append(self.X[val_indices])
            folds_y_val.append(self.Y[val_indices])
        for batch_size_2 in self.batch_size_2_pow_values:
            for l2 in self.L2_values:
                for use_adam in [True, False]:
                    for b1_b2 in self.b1_and_b2_values:
                        for K in self.K_values:
                            for S in self.S_values:
                                if (K != 0 and (S != 0)):
                                    continue
                                accuracies_per_fold : list[float] = []
                                for fold in range(self.num_folds):
                                    X_train, X_val = folds_x_train[fold], folds_x_val[fold]
                                    Y_train, Y_val = folds_y_train[fold], folds_y_val[fold]

                                    model : RedNeuronal = RedNeuronal(M, h)
                                    model.stochastic_gradient_descent(
                                        X_train,
                                        Y_train,
                                        epochs=self.epochs,
                                        learning_rate=self.learning_rate,
                                        batch_size_2_pow=batch_size_2,
                                        K=K,
                                        c=self.c_value,
                                        S=S,
                                        use_adam=use_adam,
                                        b1=b1_b2[0],
                                        b2=b1_b2[1],
                                        L2=l2,
                                    )
                                    pred = model.forward_pass(X_val, model.W, model.w_0)
                                    # loss = model.cross_entropy(Y_val, preds)
                                    accuracy = model.get_accuracy(Y_val, pred)
                                    accuracies_per_fold.append(np.mean(accuracy))
                                model_score.append(
                                    {
                                        'accuracy' : np.mean(accuracies_per_fold),
                                        'model_index' : model_index,
                                        'lr_range' : self.learning_rate,
                                        'batch_size_2' : batch_size_2,
                                        'l2' : l2,
                                        'use_adam' : use_adam,
                                        'b1_b2' : b1_b2,
                                        'K' : K,
                                        'c' : self.c_value,
                                        'S' : S,
                                    }
                                )
                                # print(batch_size_2)
                                print(f"model index: {model_index}")
                                model_index += 1
        return model_score
    
    def print_n_scores(self, model_score: list[dict], n: int) -> None:
        top_n = sorted(model_score, key=lambda x: x['accuracy'], reverse=True)[:n]
        for entry in top_n:
            print(entry)


class RedNeuronal:
    ## M: estructura sin input layer
    def __init__(
            self, 
            M : list[int], 
            h : list[str], 
        ):
        self.M : list[int] = M
        # self.M.append(48)
        # self.M.insert(0, 784)
        self.h : list[str] = h # funciones de activación, incluyendo última capa
        self.L : int = len(M) # cantidad de capas, incluyendo input y output.
        self.z : list[np.ndarray] = []
        self.a : list[np.ndarray] = []
        self.W : list[np.ndarray] = []
        self.w_0 : list[np.ndarray] = []
        self.pred : np.ndarray = np.array([])
    
    def relu(self, x : np.ndarray, gradient : bool = False) -> np.ndarray:
        if gradient:
            return np.where(x > 0, 1, 0)
        return np.where(x > 0, x, 0)
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        # Restar el valor máximo de cada fila para estabilizar la función softmax
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)
    
    def cross_entropy(self, y_true : np.ndarray, y_pred : np.ndarray) -> float:
        epsilon = 1e-10
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        # return - (y_true.T @ np.log(y_pred))
        return -np.sum(y_true * np.log(y_pred), axis=1)
    
    def cross_entropy_gradient(self, y_true : np.ndarray, y_pred : np.ndarray) -> np.ndarray:
        epsilon = 1e-10
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return - (y_true / y_pred)
    
    def initialize_weights(self) -> np.ndarray:
        W : np.ndarray = [
            np.random.randn(self.M[l-1], self.M[l]) * np.sqrt(2 / (self.M[l-1] + self.M[l]))
            for l in range(1, len(self.M))
        ]
        w_0 : np.ndarray = [
            np.zeros(self.M[l], dtype=float) for l in range(1, len(self.M))
        ]
        return W, w_0
    
    def forward_pass(self, x, W, w_0) -> np.ndarray:
        # self.z.append(x)    # Cada fila de x es una muestra
        self.z = [x]
        self.a = []
        
        for l in range(0, self.L-1):
            self.a.append((self.z[l] @ W[l]) + w_0[l])
            self.z.append(self.relu(self.a[l]) if self.h[l] == 'relu' else self.softmax(self.a[l]))
        return self.z[self.L-1]
    
    def backward_pass(self, y, pred, W, w_0, L2 : float = 0.0) -> np.ndarray:
        delta : list[np.ndarray] = [np.zeros_like(a) for a in self.a]
        delta[self.L-2] = pred - y      # derivada de la cross-entropy con softmax
        dW : list[np.ndarray] = [np.zeros_like(W[l]) for l in range(self.L - 1)]
        dw_0 : list[np.ndarray] = [np.zeros_like(w_0[l]) for l in range(self.L - 1)]
        dW[self.L - 2] = (self.z[self.L - 2].T @ delta[self.L - 2]) + (L2 * W[self.L - 2])
        dw_0[self.L - 2] = np.sum(delta[self.L - 2], axis=0)
        for l in range(self.L - 3, -1, -1):
            activation_grad : np.ndarray = self.relu(self.a[l], True) if self.h[l] == 'relu' else self.softmax(self.a[l], True)
            delta[l] = activation_grad * (delta[l+1] @ self.W[l+1].T)
            dW[l] = self.z[l].T @ delta[l]
            # dw_0[l] = dw_0[l] = np.sum(delta[l], axis=0)
            dw_0[l] = np.sum(delta[l], axis=0)
        return dW, dw_0
    
    def computar_gradiente(self, x : np.ndarray, y : np.ndarray, W : np.ndarray, w_0 : np.ndarray, L2 : float = 0.0) -> np.ndarray:
        # Forward pass
        pred : np.ndarray = self.forward_pass(x, W, w_0)
        loss : np.ndarray = self.cross_entropy(y, pred)
        # Backward-pass
        dW : np.ndarray
        dw_0 : np.ndarray
        dW, dw_0 = self.backward_pass(y, pred, W, w_0, L2=L2)
        # print("")
        # print("W[self.L-2].shape: ", W[self.L-2].shape)
        # print("w_0[self.L-2].shape: ", w_0[self.L-2].shape)
        # print("")
        # print("pred.shape: ", pred.shape)
        # print("loss.shape: ", loss.shape)
        # print("dW.shape[self.L-2]: ", dW[self.L-2].shape)
        # print("dW0.shape[self.L-2]: ", dw_0[self.L-2].shape)
        # print("")
        return pred, loss, dW, dw_0
    
    def batch_gradient_descent(self, X : np.ndarray, Y : np.ndarray, epochs : int, learning_rate : tuple[float, float], K : int = 0, c : float = 0, S : float = 0, L2 : float = 0.0, print_results_rate : int = -1) -> None:
        # K: K steps LR schedule
        # c: exponential LR schedule
        # S: exponential LR schedule
        self.W, self.w_0 = self.initialize_weights()
        # pred, loss, dW, dw_0 = self.computar_gradiente(X, Y, self.W, self.w_0)
        pred : np.ndarray
        loss : np.ndarray
        dW : list[np.ndarray] 
        dw_0 : list[np.ndarray]
        last_loss_mean : float = np.inf
        for epoch in range(1, epochs+1):
            pred, loss, dW, dw_0 = self.computar_gradiente(X, Y, self.W, self.w_0, L2=L2)
            
            # actualizaciones
            for i in range(self.L - 1):
                lr_t : float
                # Rate scheduling
                if c > 0 and S > 0:
                    lr_t = learning_rate[0] * ((1 + ((epoch-1) / S))**c)
                elif K > 0:
                        if epoch <= K:
                            lr_t = ((1 - ((epoch - 1) / K)) * learning_rate[0]) + (((epoch - 1) / K) * learning_rate[1])
                        else:
                            lr_t = learning_rate[1]
                else:
                    lr_t = learning_rate[0] - (((learning_rate[0] - learning_rate[1]) / epochs) * (epoch-1))

                self.W[i] -= lr_t * dW[i]
                self.w_0[i] -= lr_t * dw_0[i]
            
            # print("loss.shape: ", loss.shape)
            if (epoch) % print_results_rate == 0 and print_results_rate != -1:
                loss_mean : float = np.mean(loss)
                print(f"Epoch {epoch}\n-> Loss = {loss}\n-> Loss Mean = {loss_mean}")
                if last_loss_mean != np.inf:
                    print(f"-> Difference = {'+' if loss_mean - last_loss_mean >= 0 else ''}{loss_mean - last_loss_mean}")
                last_loss_mean = loss_mean
        self.pred = pred

    def stochastic_gradient_descent(
            self, 
            X : np.ndarray, 
            Y : np.ndarray, 
            epochs : int, 
            learning_rate : tuple[float, float], 
            batch_size_2_pow : int = 0, 
            K : int = 0,
            c : float = 0.0, 
            S : float = 0.0, 
            use_adam : bool = False, 
            b1 : float = 0.0, 
            b2 : float = 0.0, 
            L2 : float = 0.0,
            print_results_rate : int = -1
        ) -> None:

        # K: K steps LR lineal schedule
        # c: exponential LR schedule
        # S: exponential LR schedule
        self.W, self.w_0 = self.initialize_weights()
        # pred, loss, dW, dw_0 = self.computar_gradiente(X, Y, self.W, self.w_0)
        pred : np.ndarray
        loss : np.ndarray
        dW : list[np.ndarray] 
        dw_0 : list[np.ndarray]
        last_loss_mean : float = np.inf

        # Adam parameters
        m_t: list[np.ndarray] = [np.zeros_like(w) for w in self.W]
        v_t: list[np.ndarray] = [np.zeros_like(w) for w in self.W]
        m_t_0: list[np.ndarray] = [np.zeros_like(w) for w in self.w_0]
        v_t_0: list[np.ndarray] = [np.zeros_like(w) for w in self.w_0]
        epsilon : float = 1e-8
        
        for epoch in range(1, epochs+1):
            batch_size : int = 2 ** batch_size_2_pow if 2 ** batch_size_2_pow < len(X) else len(X)
            for batch in range(len(X) // batch_size):
                X_b : np.ndarray = X[batch * batch_size : (batch + 1) * batch_size]
                Y_b : np.ndarray = Y[batch * batch_size : (batch + 1) * batch_size]
                pred, loss, dW, dw_0 = self.computar_gradiente(X_b, Y_b, self.W, self.w_0, L2=L2)
                
                # actualizaciones
                for i in range(self.L - 1):
                    lr_t : float
                    # Rate scheduling
                    if c > 0 and S > 0:
                        lr_t = learning_rate[0] * ((1 + ((epoch-1) / S))**c)
                    elif K > 0:
                        # lr_t = ((1 - ((epoch-1) / K))*learning_rate[0]) + (((epoch-1) / K)*learning_rate[1])
                        if epoch <= K:
                            lr_t = ((1 - ((epoch - 1) / K)) * learning_rate[0]) + (((epoch - 1) / K) * learning_rate[1])
                        else:
                            lr_t = learning_rate[1]
                    else:
                        lr_t = learning_rate[0] - (((learning_rate[0] - learning_rate[1]) / epochs) * (epoch-1))

                    if use_adam:
                        # Adam updates
                        m_t[i] = b1 * m_t[i] + (1 - b1) * dW[i]
                        v_t[i] = b2 * v_t[i] + (1 - b2) * (dW[i] ** 2)

                        # Bias correction
                        m_hat = m_t[i] / (1 - b1 ** epoch)
                        v_hat = v_t[i] / (1 - b2 ** epoch)

                        # Update weights
                        self.W[i] -= lr_t * m_hat / (np.sqrt(v_hat) + epsilon)

                        # Repeat for bias
                        m_t_0[i] = b1 * m_t_0[i] + (1 - b1) * dw_0[i]
                        v_t_0[i] = b2 * v_t_0[i] + (1 - b2) * (dw_0[i] ** 2)
                        m_hat_0 = m_t_0[i] / (1 - b1 ** epoch)
                        v_hat_0 = v_t_0[i] / (1 - b2 ** epoch)

                        self.w_0[i] -= lr_t * m_hat_0 / (np.sqrt(v_hat_0) + epsilon)
                    else:
                        # Standard gradient descent
                        self.W[i] -= lr_t * dW[i]
                        self.w_0[i] -= lr_t * dw_0[i]
            
            # print("loss.shape: ", loss.shape)
            if (epoch) % print_results_rate == 0 and print_results_rate != -1:
                loss_mean : float = np.mean(loss)
                # print(f"Epoch {epoch}\n-> Loss = {loss}\n-> Loss Mean = {loss_mean}")
                print(f"Epoch {epoch}\n-> Loss Mean = {loss_mean}")
                if last_loss_mean != np.inf:
                    print(f"-> Difference = {'+' if loss_mean - last_loss_mean >= 0 else ''}{loss_mean - last_loss_mean}")
                last_loss_mean = loss_mean
        self.pred = pred

    def get_train_accuracy(self, y_ground_truth: np.ndarray) -> float:
        pred_labels = np.argmax(self.pred, axis=1)
        true_labels = np.argmax(y_ground_truth, axis=1)
        return np.mean(pred_labels == true_labels)

    def get_train_confusion_matrix(self, y_ground_truth: np.ndarray) -> np.ndarray:
        num_classes : int = y_ground_truth.shape[1]
        pred_labels : np.ndarray = np.argmax(self.pred, axis=1)
        true_labels : np.ndarray = np.argmax(y_ground_truth, axis=1)
        cm : np.ndarray = np.zeros((num_classes, num_classes), dtype=int)
        for true, self.pred in zip(true_labels, pred_labels):
            cm[true, self.pred] += 1
        return cm

    def get_accuracy(self, y_ground_truth: np.ndarray, pred : np.ndarray) -> float:
        pred_labels = np.argmax(pred, axis=1)
        true_labels = np.argmax(y_ground_truth, axis=1)
        return np.mean(pred_labels == true_labels)
    
    def get_prediction(self, x : np.ndarray) -> np.ndarray:
        return self.forward_pass(x, self.W, self.w_0)