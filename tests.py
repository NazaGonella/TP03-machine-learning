import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import src.data_handler as data_handler
import os

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))

X_images : np.ndarray[float] = np.load(f"{project_root}/data/X_images.npy")
y_images : np.ndarray[float] = np.load(f"{project_root}/data/y_images.npy")

print(X_images)
print(set(y_images))