import pandas as pd
import os

project_root : str = os.path.abspath(os.path.join(os.getcwd(), ".."))

def get_splitted_dataset(
        df : pd.DataFrame, 
        train_fraction : float = 0.8, 
        val_fraction : float = 0.15, 
        test_fraction : float = 0.15, 
        seed : int = 42
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    train : pd.DataFrame = df.sample(frac=train_fraction, random_state=seed)
    validation : pd.DataFrame = df.drop(train.index).sample(frac=val_fraction/(val_fraction + test_fraction), random_state=seed)
    test : pd.DataFrame = df.drop(validation.index)
    return train, validation, test
