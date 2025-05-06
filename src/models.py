import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

class RedNeuronal:
    def __init__(self, M : list[int], initial_weights : int = 0):
        self.L = len(M)
        self.M = M
        self.w = [np.array([initial_weights for _ in range(M[i])], dtype=float) for i in range(len(M))]

    def relu(self, x : float) -> float:
        return max(0, x)
    
    def softmax(self, x : np.ndarray[float]) -> np.float32:
        e_x : np.ndarray[float] = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    
    def cross_entropy(self, y_true : np.ndarray[float], y_pred : np.ndarray[float]) -> float:
        return y_true.T @ np.log(y_pred) 
    
    def train(self, x, y) -> None:
        pass

red = RedNeuronal([2, 7, 5], initial_weights=1)
print(red.L)
print(red.M)
print(red.w)
# print("")
# print(red.softmax(red.w[2]))

# x = np.array([0.3, 0.5, 0.2, 4, 0.1])
# print(red.softmax(red.w[2]).T @ x)