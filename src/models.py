import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from collections.abc import Callable

class CrossValidation:
    def __init__(
        self,
        X : np.ndarray, 
        Y : np.ndarray, 
        num_folds : int, 
        epochs : int, 
        learning_rate_values : list[tuple[int, int]], 
        batch_size_2_pow_values : list[int], 
        K : list[int],
        c : list[float], 
        S : list[float], 
        b1 : list[float], 
        b2 : list[float], 
        L2 : list[float],
    ):
        self.X : np.ndarray = X
        self.Y : np.ndarray = Y
        self.num_folds : int = num_folds
        self.epochs : int = epochs
        self.learning_rate_values : list[tuple[int, int]] = learning_rate_values 
        self.batch_size_2_pow_values : list[int] = batch_size_2_pow_values 
        self.K : list[int] = K
        self.c : list[float] = c
        self.S : list[float] = S
        self.b1 : list[float] = b1
        self.b2 : list[float] = b2
        self.L2 : list[float] = L2

    def evaluate_hiperparameters(self, M : list[int], h : list[str]):
        fold_size : int = len(self.X) // self.num_folds
        indices = np.arange(len(self.X))
        fold_scores = []
        for fold in range(self.num_folds):
            val_indices = indices[fold * fold_size : (fold + 1) * fold_size]
            train_indices = np.setdiff1d(indices, val_indices, assume_unique=True)

            X_train, X_val = self.X[train_indices], self.X[val_indices]
            Y_train, Y_val = self.Y[train_indices], self.Y[val_indices]

            M : RedNeuronal = RedNeuronal(M, h)
            M.stochastic_gradient_descent(
                X_train,
                Y_train,
                epochs=self.epochs,
            )

class RedNeuronal:
    ## M: estructura sin input layer
    def __init__(
            self, 
            M : list[int], 
            h : list[str], 
        ):
        self.M : list[int] = M
        self.M.append(48)
        self.M.insert(0, 784)
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