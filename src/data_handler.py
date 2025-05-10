import pandas as pd
import numpy as np
import os

project_root : str = os.path.abspath(os.path.join(os.getcwd(), ".."))

def get_splitted_dataset(
        df_x : pd.DataFrame, 
        df_y : pd.DataFrame, 
        train_fraction : float = 0.8, 
        val_fraction : float = 0.15, 
        test_fraction : float = 0.15, 
        seed : int = 42
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    # train : pd.DataFrame = df.sample(frac=train_fraction, random_state=seed)
    # validation : pd.DataFrame = df.drop(train.index).sample(frac=val_fraction/(val_fraction + test_fraction), random_state=seed)
    # test : pd.DataFrame = df.drop(validation.index)
    # print("train_fraction: ", train_fraction)
    # print("val_fraction: ", val_fraction)
    # print("test_fraction: ", test_fraction)
    
    # Shuffle data
    indices = np.arange(len(df_x))
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    train_end = int(train_fraction * len(indices))
    val_end = train_end + int(val_fraction * len(indices))
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    # Split data
    X_train, y_train = df_x.iloc[train_indices], df_y.iloc[train_indices]
    X_val, y_val = df_x.iloc[val_indices], df_y.iloc[val_indices]
    X_test, y_test = df_x.iloc[test_indices], df_y.iloc[test_indices]
    
    return X_train, X_val, X_test, y_train, y_val, y_test
    # return train, validation, test
