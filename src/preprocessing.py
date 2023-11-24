import logging
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2

class DataPreprocess:
    def __init__(self, data: pd.DataFrame, args):
        self.data = data
        self.X = None
        self.y = None
        self.header_X = None
        self.sel_header = None
        self.feature2drop = []
        self.args = args

    def data_sort(self, value):
        self.data.sort_values(by=value, inplace=True)

    def split_data(self):
        header = list(self.data.columns.values)
        header.remove('Status')

        self.X = self.data[header].copy()
        self.y = self.data['Status'].copy()
        logging.info("\t Total number of training data: {}, and CVD death {}".format(
            self.y.shape[0], np.sum(self.y)))

        self.header_X = list(self.X.columns.values)

    def feature_selection(self, n_features=40):
        #     sel = VarianceThreshold(threshold=0.8*(1-0.8))
        sel = SelectKBest(chi2, k=n_features)
        sel.fit_transform(self.X, self.y)
        self.sel_header = self.X.columns.values[sel.get_support()]
        self.X = self.X[self.sel_header]

    def drop_columns(self, df):
        return df.drop([column for column in self.feature2drop if column in df.columns], axis=1)

    def data_aggregation(self):
        """ Drop columns that are not useful and aggregation data to reduce dimension
        """
        # Generate the CVD death flag
        self.data = self.data.assign(Status=self.data['mortstat'] & (self.data['ucod_leading'] == 1))

        # Diabetes Flag generation direction:
        #   1, Categorical variable of the following:
        #      0: Non-diabetes,
        #      1: Pre-diabetes,
        #      2: Diabetes
        #   2, Binary flag of diabetes
        def determine_value(fg, hba1c, dmdrug):
            """ For glycemicstatus feature
            """
            if fg < 100 and hba1c < 5.7:
                return 0
            elif 100 <= fg <= 125 or 5.7 <= hba1c <= 6.4:
                return 1
            elif fg >= 126 or hba1c >= 6.5 or dmdrug:
                return 2
            else:
                return np.nan

        self.data['glycemicstatus'] = self.data.apply(
            lambda x: determine_value(x['fastglucose_ModP'], x['LBXGH'], x['DMdrug']),
            axis=1)

        def combine_dm(self_report, exam, binary=False):
            """ For dm_flag feature
            """
            if not binary:
                # if the desire output is categorical
                if self_report == 1 or exam == 2:
                    return 2
                elif exam == 1:
                    return 1
                elif exam == 0:
                    return 0
                elif np.isnan(exam):
                    return np.nan
            else:
                # if the desire output is binary
                if self_report == 1 or exam == 2:
                    return 1
                elif self_report == 0 or exam == 0:
                    return 0
                else:
                    return np.nan

        # TODO very next thing to do
        if self.args.aggregation == "categorical":
            self.data['dm_flag'] = self.data.apply(lambda x: combine_dm(x['DMdiagnosishistory'], x['glycemicstatus']),
                                                   axis=1)
        elif self.args.aggregation == "binary":
            self.data['dm_flag'] = self.data.apply(
                lambda x: combine_dm(x['DMdiagnosishistory'], x['glycemicstatus'], binary=True),
                axis=1)
        self.feature2drop.append('glycemicstatus')

    def delete_features(self):

        # Delete participant IDs
        self.feature2drop.append('SEQN')

        # Delete because duplication with "educationfor20"
        self.feature2drop.append('DMDEDUC2')

        # Not useful as 'LBXGH' is already used in generating 'dm_flag'
        self.feature2drop.append('HBA1c_le65')

        # Delete because of data aggregation for 'dm_flag'
        self.feature2drop.append('fastglucose_ModP')
        self.feature2drop.append('LBXGH')
        self.feature2drop.append('DMdrug')

        # Delete unused prediction direction
        self.feature2drop.append('mortstat')
        self.feature2drop.append('ucod_leading')
        self.feature2drop.append('diabetes')
        self.feature2drop.append('hyperten')

        # TODO check if need to keep with Dr Leung -- result: delete
        self.feature2drop.append('LBXSGL')
        self.feature2drop.append("glucose_nonfast_le200")

        self.data = self.drop_columns(self.data)

    def only_positive_cases(self):
        self.data = self.data.drop(self.data[self.data['cvddeath'] == 0].index)

    def preprocess(self):
        # Sort data by "SEQN" for better performance in debugging
        self.data_sort(["SEQN"])

        self.data_aggregation()

        self.delete_features()

        # Split labels from data
        self.split_data()

        yield self.X, self.y

