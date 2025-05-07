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
            initial_weights : int = 0
        ):
        self.M : list[int] = M
        self.h : list[str] = h
        self.initial_weights : int = initial_weights
        self.L : int = len(M)
        self.z : list[np.ndarray] = [np.array(
            [0 for _ in range(M[i])], dtype=float
        ) for i in range(len(M))] # Falta layer de input en z[0]
        self.a : list[np.ndarray] = [np.array(
            [0 for _ in range(M[i])], dtype=float
        ) for i in range(len(M))]
        self.w : list[np.ndarray] = np.array([]) # len(M) matrices de pesos, I en w[0]
        self.w_0 : list[np.ndarray] = [np.array(
            [0 for _ in range(M[i])], dtype=float
        ) for i in range(len(M))] # len(M)
    
    def relu(self, x : np.ndarray, gradient : bool = False) -> np.ndarray:
        if gradient:
            return np.where(x > 0, 1, 0)
        return np.where(x > 0, x, 0)
    
    def softmax(self, x : np.ndarray, gradient : bool = False) -> np.ndarray:
        e_x : np.ndarray = np.exp(x - np.max(x))
        if gradient:
            pass
        return e_x / e_x.sum(axis=0)
    
    def cross_entropy(self, y_true : np.ndarray, y_pred : np.ndarray, gradient : bool = False) -> float:
        if gradient:
            return - (y_true / y_pred)
        return - (y_true.T @ np.log(y_pred))

    def backpropagation(self, y_i : np.ndarray, W : np.ndarray, w_0 : np.ndarray) -> np.ndarray:
        # forward pass
        suma = 0
        for l in range(1, self.L + 1):
            suma += 1
            self.a[l] = (W[l-1] @ self.z[l-1]) + w_0[l-1]
            self.z[l] = self.relu(self.a) if self.h[l] == 'relu' else self.softmax(self.a[l])
        
        pred : np.ndarray = self.z[self.L]
        loss_i : float = self.cross_entropy(y_i, pred)

        # Backward-pass
        delta : np.ndarray = np.array([0 for _ in range(self.L)])
        delta[self.L] = [self.h[l](self.a[l], True)] * self.cross_entropy(y_i, pred, gradient=True)
        al_aw : np.ndarray = delta @ self.z[self.L - 1].T
        al_aw0 : np.ndarray = np.array([])
        for l in range(1, self.L - 1 + 1, -1):
            delta[l] = (self.relu(self.a, gradient=True) if self.h[l] == 'relu' else self.softmax(self.a[l], gradient=True)) * (W[l].T @ delta[l+1])
            al_aw = delta[l] @ self.z[l-1].T
            al_aw0 = delta[l]
        
        return pred, loss_i, al_aw, al_aw0


    def train(self, x, y) -> None:
        self.z.insert(0, x[0])
        # print("z: ", len(self.z[0]))
        self.w = [
            np.array([
                [
                    self.initial_weights for _ in range(len(self.z[i]))
                ] for _ in range(self.M[i])
            ], dtype=float) for i in range(self.L)
        ]
        # print(self.w[1].shape)
        pred, loss_i, al_aw, al_aw0 = self.backpropagation(y[0], self.w, self.w_0)

# red = RedNeuronal([2, 7, 5], [RedNeuronal.relu, RedNeuronal.relu])
# print(red.L)
# print(red.M)
# print(red.z)
# print("")
# print(red.softmax(red.w[2]))

# x = np.array([0.3, 0.5, 0.2, 4, 0.1])
# print(red.softmax(red.w[2]).T @ x)
# print(list(range(10, 1, -1)))