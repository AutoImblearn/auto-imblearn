import os
from sys import exit

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from hyperimpute.plugins.imputers import Imputers

import pandas as pd

imps = ["median", "knn", "ii", "gain", "MIRACLE", "MIWAE"]


class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self, method="median", aggregation=None):
        self.method = method
        self.aggregation = aggregation

        self.data = None
        self.header_X = None
        self.feature2drop = []

    def apply_rounding(self):
        # Add rounding to categorical features after imputation
        self.category_columns = [
            "DMDEDUC2",
            "DMDMARTL",
            "educationfor20",
            "RIDRETH1",
            "SDDSRVYR",
            "smoking",
        ]
        self.category_columns = [i for i in self.category_columns if i in self.data.columns.values]
        for column in self.category_columns:
            self.data[column] = self.data[column].round(0)


    def handle_missing(self):
        # Apply imputation to data
        if self.method == "dropna":
            self.data.dropna(inplace=True)

        elif self.method == "median":
            medians = self.data.median()
            self.data = self.data.fillna(medians)

        elif self.method == "mean":
            means = self.data.mean()
            self.data = self.data.fillna(means)

        elif self.method == "knn":
            data_file_name = "knnimputer.csv"
            file_path = os.path.join("..", "data", "interim", data_file_name)
            if os.path.isfile(file_path):
                self.data[:] = pd.read_csv(file_path)
            else:
                impute = KNNImputer(weights='distance', n_neighbors=1)
                self.data[:] = impute.fit_transform(self.data)
                self.apply_rounding()
                del impute

        elif self.method == "ii":
            data_file_name = "iiimputer.csv"
            file_path = os.path.join("..", "data", "interim", data_file_name)
            if os.path.isfile(file_path):
                self.data[:] = pd.read_csv(file_path)
            else:
                impute = IterativeImputer(
                )
                self.data[:] = impute.fit_transform(self.data)
                self.apply_rounding()
                del impute

        elif self.method in ['gain', 'MIRACLE']:
            dict_types = dict(self.data.dtypes)
            old_columns = self.data.columns.values

            impute = Imputers().get(self.method.lower())
            self.data = impute.fit_transform(self.data.astype('float32').copy())
            # Change back to old column names
            rename_dict = dict(map(lambda i, j: (i, j), self.data.columns.values, old_columns))
            self.data.rename(rename_dict, axis=1, inplace=True)

            # Change back to old column dtypes
            self.data = self.data.astype(dict_types)
            self.apply_rounding()
            del impute

        elif self.method in ['MIWAE']:
            dict_types = dict(self.data.dtypes)
            old_columns = self.data.columns.values

            impute = Imputers().get(self.method.lower(), random_state=42, batch_size = 128)
            self.data = impute.fit_transform(self.data.astype('float32').copy())
            # Change back to old column names
            rename_dict = dict(map(lambda i, j: (i, j), self.data.columns.values, old_columns))
            self.data.rename(rename_dict, axis=1, inplace=True)

            # Change back to old column dtypes
            self.data = self.data.astype(dict_types)
            self.apply_rounding()
            del impute

        else:
            print("Error with handling missing value")
            exit()

    def data_scaling(self):
        """ Scale Features """
        min_max_columns = [
            "RIDAGEYR",
            "DMDHHSIZ",
            "INDFMPIR",
            "BMXWT",
            "BMXHT",
            "BMXLEG",
            "BMXARML",
            "BMXARMC",
            "BMXWAIST",
            "BMXTRI",
            "BMXSUB",
            "URXUMA",
            "urine_creatinine_corrected",
            "LBDSALSI",
            "LBDSBUSI",
            "LBXSCA",
            "LBXSC3SI",
            "sermcreatinine_corrected",
            "LBXSIR",
            "LBXSLDSI",
            "LBXSPH",
            "LBXSTP",
            "LBDSTBSI",
            "LBDSUASI",
            "LBXSNASI",
            "LBXSKSI",
            "LBXSCLSI",
            "LBXSOSSI",
            "LBXSGB",
            "LBXSAPSI",
            "LBXSATSI",
            "LBXSASSI",
            "LBXSGTSI",
            "BPXSAR_Leung",
            "BPXDAR_Leung",
            "HTdrug",
            "LBXTC",
            "LBDHDD",
            "LBXTR",
            "LBDLDL",
            "LBXGH",
            "fastglucose_ModP",
            "fastinsulin_Mercodia",
            "HOMAIR",
            "LBXSGL",
            "lipiddrug_no_pm",
            "HTdrug_no_pm",
            "DMdrug_no_pm",
            "BMXBMI",
            "UACR",
            "eGFR",
            "LBXCRP",
            "permth_exm",
            "LBXWBCSI",
            "LBXRBCSI",
            "LBXHGB",
            "LBXHCT",
            "LBXMCVSI",
            "LBXMCHSI",
            "LBXMC",
            "LBXRDW",
            "LBXPLTSI",
            "LBXMPSI",
            'dm_flag',
        ]
        min_max_columns = [item for item in min_max_columns if item in self.data.columns.values]

        self.data[min_max_columns] = MinMaxScaler().fit_transform(self.data[min_max_columns])

    def data_categorical(self):
        """ Categorical Data """
        # Making sex 0 and 1, instead of 1 and 2.
        # TODO check if any 1 and 2 for binary data
        self.data.loc[self.data['RIAGENDR'] == 1, 'RIAGENDR'] = 0
        self.data.loc[self.data['RIAGENDR'] == 2, 'RIAGENDR'] = 1

        # TODO try make smoking not categorical
        # Perform OneHotEncoding for Categorical data
        if self.args.aggregation == "categorical":
            self.category_columns.append("dm_flag")

        self.category_columns = [
            "DMDEDUC2",
            "DMDMARTL",
            "educationfor20",
            "RIDRETH1",
            "SDDSRVYR",
            "smoking",
        ]

        self.category_columns = [i for i in self.category_columns if i in self.data.columns.values]

        self.data = pd.get_dummies(data=self.data, columns=self.category_columns)

    def fit(self, X, y = None):
        self.data = X
        return self

    def transform(self, X):
        self.handle_missing()

        self.data_categorical()

        # Scale data to the same interval
        self.data_scaling()

        self.header_X = self.data.columns.values

        return self.data
