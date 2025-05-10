import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from collections.abc import Callable

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
        # print("M: ", M)
        self.h : list[str] = h # Funciones de activación, incluyendo última capa
        self.L : int = len(M) # cantidad de capas, incluyendo input y output.
        self.z : list[np.ndarray] = []
        self.a : list[np.ndarray] = []
        self.W : list[np.ndarray] = []
        self.w_0 : list[np.ndarray] = []
    
    def relu(self, x : np.ndarray, gradient : bool = False) -> np.ndarray:
        if gradient:
            return np.where(x > 0, 1, 0)
        return np.where(x > 0, x, 0)
    
    # def softmax(self, x : np.ndarray) -> np.ndarray:
    #     e_x : np.ndarray = np.exp(x - np.max(x))
    #     # return e_x / e_x.sum(axis=1)
    #     return e_x / e_x.sum(axis=1, keepdims=True)
    def softmax(self, x: np.ndarray) -> np.ndarray:
        # Restar el valor máximo de cada fila para estabilizar la función softmax
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)
    
    def cross_entropy(self, y_true : np.ndarray, y_pred : np.ndarray) -> float:
        epsilon = 1e-10
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        # return - (y_true.T @ np.log(y_pred))
        return -np.sum(y_true * np.log(y_pred), axis=1)
    
    # def cross_entropy_gradient(self, y_true : np.ndarray, y_pred : np.ndarray) -> np.ndarray:
    #     epsilon = 1e-10
    #     y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    #     # return - (y_true / y_pred)
    #     return -np.sum(y_true / y_pred, axis=1, keepdims=True)
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
    
    def backward_pass(self, y, pred, W, w_0) -> np.ndarray:
        delta : list[np.ndarray] = [np.zeros_like(a) for a in self.a]
        delta[self.L-2] = pred - y      # derivada de la cross-entropy con softmax
        dW : list[np.ndarray] = [np.zeros_like(W[l]) for l in range(self.L - 1)]
        dw_0 : list[np.ndarray] = [np.zeros_like(w_0[l]) for l in range(self.L - 1)]
        dW[self.L - 2] = self.z[self.L - 2].T @ delta[self.L - 2]
        dw_0[self.L - 2] = np.sum(delta[self.L - 2], axis=0)
        for l in range(self.L - 3, -1, -1):
            activation_grad : np.ndarray = self.relu(self.a[l], True) if self.h[l] == 'relu' else self.softmax(self.a[l], True)
            delta[l] = activation_grad * (delta[l+1] @ self.W[l+1].T)
            dW[l] = self.z[l].T @ delta[l]
            # dw_0[l] = dw_0[l] = np.sum(delta[l], axis=0)
            dw_0[l] = np.sum(delta[l], axis=0)
        return dW, dw_0
    
    def computar_gradiente(self, x : np.ndarray, y : np.ndarray, W : np.ndarray, w_0 : np.ndarray) -> np.ndarray:
        # Forward pass
        pred : np.ndarray = self.forward_pass(x, W, w_0)
        loss : np.ndarray = self.cross_entropy(y, pred)
        # Backward-pass
        dW : np.ndarray
        dw_0 : np.ndarray
        dW, dw_0 = self.backward_pass(y, pred, W, w_0)
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

    def gradient_descent(self, X, Y, epochs, learning_rate):
        # Inicialización
        self.W, self.w_0 = self.initialize_weights()
        # pred, loss, dW, dw_0 = self.computar_gradiente(X, Y, self.W, self.w_0)
        
        for epoch in range(epochs):
            pred : np.ndarray
            loss : np.ndarray
            dW : list[np.ndarray] 
            dw_0 : list[np.ndarray]
            pred, loss, dW, dw_0 = self.computar_gradiente(X, Y, self.W, self.w_0)
            # if epoch % 100 == 0:
            #     print("pred: ", pred)
            #     print("np.sum(pred[self.L-2]): ", np.sum(pred[self.L-2]))
            
            # Actualización de pesos y sesgos
            for i in range(self.L - 1):
                # if epoch % 100 == 0:
                #     print(f"EPOCH {epoch}, LAYER {i}")
                #     print("W[self.L-2].shape: ", self.W[i].shape)
                #     print("w_0[self.L-2].shape: ", self.w_0[i].shape)
                #     print("dW.shape[self.L-2]: ", dW[i].shape)
                #     print("dw_0.shape[self.L-2]: ", dw_0[i].shape)
                #     print("")
                self.W[i] -= learning_rate * dW[i]
                self.w_0[i] -= learning_rate * dw_0[i]
            
            # print("loss.shape: ", loss.shape)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

        return self.W, self.w_0

# red = RedNeuronal([2, 7], [RedNeuronal.relu, RedNeuronal.relu])
# x = np.array([
#     [0,1,0],
#     [0,1,0],
#     [0,0,1],
# ])
# y = np.array([
#     [0.2,0.7,0.1],
#     [0.05,0.9,0.05],
#     [0,0.1,0.9],
# ])
# a = np.array([[
#     0,
#     1,
#     0,
# ]])
# b = np.array([[
#     0.2,
#     0.7,
#     0.1,
# ]])
# print(red.relu())
# print("")
# print("x: ", x)
# print("y: ", y)
# print("")
# print("a: ", a)
# print("b", b)
# print("")
# print(red.cross_entropy(a, b))
# red.train(x, y)

# CADA ROW ES UNA MUESTRA
# axis=1
# FILAS
# PRIMER INDEX