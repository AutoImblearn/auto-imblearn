import os
import pandas as pd
import numpy as np
import re
import pickle

from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans


class DataLoader:
    """ DataLoader object to load train, valid and test data from dataset.
        Args:
            dataset (str): Name os the dataset
    """

    def __init__(self,
                 dataset: str, is_notebook=False) -> None:
        self.item_set = set()
        self.path = os.path.join("..", 'data', 'raw', dataset)
        if is_notebook:
            self.path = os.path.join("../..", self.path)
        self.header = []
        self.name = dataset

    def train_loader(self) -> pd.DataFrame:
        """ Load training data
        :returns:
            df: whole data
        """
        null_values = ['', ' ']

        filetype = re.search("[^\.]*$", self.path).group()

        if filetype == "csv":
            df = pd.read_csv(self.path, na_values=null_values)
        elif filetype == "xlsx":
            df = pd.read_excel(self.path, na_values=null_values)
        self.header = list(df.columns.values)
        return df


class Samplar:
    """ Samplar oject to split data
        Args:
            X (np.ndarray):
            Y (np.ndarray):
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray):

        self.X = X
        self.Y = Y


    def apply_kfold(self, split_num):
        """Apply stratified cross validation to dataset"""
        skf = StratifiedKFold(n_splits=split_num, shuffle=True, random_state=42)
        for train_index, test_index in skf.split(self.X, self.Y):
            X_train, X_test = self.X[train_index], self.X[test_index]
            Y_train, Y_test = self.Y[train_index], self.Y[test_index]
            yield (X_train, Y_train, X_test, Y_test)


def param_loader():
    # Load resampling strategy find manually
    param_file = os.path.join("..", 'data', 'interim', "params.p")
    with open(param_file, "rb") as f:
        params = pickle.load(f)
    return params


if __name__ == "__main__":
    a = param_loader()
    print(a)

    # dataloader = DataLoader("2022-09-13 NHANES 1999-2010 for Raymond v2.csv")
    #
    # print(dataloader.path)
    # dataloader.train_loader()
