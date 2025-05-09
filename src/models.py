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
        self.L : int = len(M) # cantidad de layers sin contar input
        
        self.z : list[np.ndarray] = [np.zeros(M[i], dtype=float) for i in range(self.L)] # Falta layer de input en z[0]
        # self.a : list[np.ndarray] = [np.array(
        #     [0 for _ in range(M[i])], dtype=float
        # ) for i in range(len(M))]
        self.a : list[np.ndarray] = [np.zeros(M[i], dtype=float) for i in range(self.L)]
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
    
    def cross_entropy(self, y_true : np.ndarray, y_pred : np.ndarray) -> float:
        epsilon = 1e-10
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return - (y_true.T @ np.log(y_pred))
    
    def cross_entropy_gradient(self, y_true : np.ndarray, y_pred : np.ndarray) -> np.ndarray:
        epsilon = 1e-10
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return - (y_true / y_pred)

    def computar_gradiente(self, y_i : np.ndarray, W : np.ndarray, w_0 : np.ndarray) -> np.ndarray:
        # Forward pass
        suma = 0
        for l in range(0, self.L):
            suma += 1
            self.a[l] = (W[l] @ self.z[l]) + w_0[l]
            self.z[l+1] = self.relu(self.a[l]) if self.h[l] == 'relu' else self.softmax(self.a[l])
        
        pred : np.ndarray = self.z[self.L]
        loss_i : float = self.cross_entropy(y_i, pred)
        # Backward-pass
        delta : list[np.ndarray] = [np.zeros_like(a) for a in self.a]
        delta[self.L-1] = self.relu(self.a[self.L-1], True) if self.h[self.L-1] == 'relu' else self.softmax(self.a[self.L-1], True) * self.cross_entropy_gradient(y_i, pred)
        al_aw = [np.zeros_like(W[l]) for l in range(self.L)]
        al_aw0 = [np.zeros_like(w_0[l]) for l in range(self.L)]
        al_aw[self.L - 1] = np.outer(delta[self.L - 1], self.z[self.L - 1])
        al_aw0[self.L - 1] = delta[self.L - 1]
        for l in range(self.L - 1, 0, -1):
            activation_grad = self.relu(self.a[l], True) if self.h[l] == 'relu' else self.softmax(self.a[l], True)
            delta[l] = activation_grad * (W[l + 1].T @ delta[l + 1])
            al_aw[l] = np.outer(delta[l], self.z[l])
            al_aw0[l] = delta[l]
        
        return pred, loss_i, al_aw, al_aw0
        # return


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
        pred, loss_i, al_aw, al_aw0 = self.computar_gradiente(y[0], self.w, self.w_0)
        print("pred: ", pred)
        print("loss_i: ", loss_i)
        print("al_aw: ", al_aw)
        print("al_aw0: ", al_aw0)

# red = RedNeuronal([2, 7, 5], [RedNeuronal.relu, RedNeuronal.relu])
# print(red.L)
# print(red.M)
# print(red.z)
# print("")
# print(red.softmax(red.w[2]))

# x = np.array([0.3, 0.5, 0.2, 4, 0.1])
# print(red.softmax(red.w[2]).T @ x)
# print(list(range(10, 1, -1)))