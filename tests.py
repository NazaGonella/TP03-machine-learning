import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import src.data_handler as data_handler
import src.models as models
import os

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
CANTIDAD_DE_CLASES = 48

X_images : np.ndarray[float] = np.load(f"{project_root}/TP03/data/X_images.npy")
y_images : np.ndarray[float] = np.load(f"{project_root}/TP03/data/y_images.npy")

# print(X_images)
# print(set(y_images))


X_images = X_images / 255
X_train : pd.DataFrame
X_validation : pd.DataFrame
X_test : pd.DataFrame
X_train, X_validation, X_test = data_handler.get_splitted_dataset(pd.DataFrame(X_images))
y_images = np.array([[0 if y_images[x] != i else 1 for i in range(CANTIDAD_DE_CLASES)] for x in range(len(X_images))], dtype=float)
red = models.RedNeuronal([2,4], ['relu', 'relu', 'softmax'])
# print(X_images.shape)
np.random.seed(42)
red.gradient_descent(X_images, y_images, 1, 0.5)
# print(np.array([[1,1,1],[1,1,1],[1,1,1],[1,1,1]]))

# print(list(range(9, -1, -1)))