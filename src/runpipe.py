import os
import pickle
import logging
import numpy as np
import pandas as pd

from utils import DataLoader, Samplar, param_loader
from preprocessing import DataPreprocess
from customrsp import CustomResamplar
from customimputation import CustomImputer
from customclf import CustomClassifier


def average(lst):
    return sum(lst) / len(lst)

class RunPipe:
    def __init__(self, args=None):
        self.preprocessor = None
        self.args = args
        self.X = None
        self.y = None
        self.proportion = None

    def loadData(self, proportion=None):

        # Load data
        logging.info("Loading Start")
        dataloader = DataLoader(self.args.dataset)
        data = dataloader.train_loader()
        logging.info("Loading Done")

        # Proprocess data
        logging.info("Preprocessing Start")
        self.preprocessor = DataPreprocess(data, self.args)

    def stratifiedSample(self, X, y, train_ratio):
        data = pd.concat([X, y], axis=1)
        new_data = data.groupby('Status', group_keys=False).apply(lambda x: x.sample(frac=train_ratio))
        new_data.sort_index(inplace=True)
        new_data.reset_index(inplace=True, drop=True)
        columns = list(new_data.columns.values)
        columns.remove("Status")
        X = new_data[columns].copy()
        y = new_data["Status"].copy()
        return X, y

    def fit(self, pipe, train_ratio=1.0):
        # Run the pipeline
        imp, rsp, clf = pipe
        #
        # Imputation level
        #
        imput_file_name = os.path.join("..", "data", "interim", "imp_" + imp + "_.p")

        if os.path.exists(imput_file_name):
            # Load Saved imputation files
            with open(imput_file_name, "rb") as f:
                X, y = pickle.load(f)
        else:
            # Save imputated data
            X, y = self.preprocessor.preprocess()
            imputer = CustomImputer(imp)
            X[:] = imputer.fit_transform(X, y)
            # Save imputation results
            data2save = (X, y)
            print(X.shape)
            with open(imput_file_name, "wb") as f:
                pickle.dump(data2save, f)

        if train_ratio != 1.0:
            X, y = self.stratifiedSample(X, y, train_ratio)

        logging.info("Preprocessing Done")
        train_sampler = Samplar(np.array(X), np.array(y))

        precisions = []
        recalls = []
        w_precisions = []
        w_recalls = []
        aurocs = []
        auprcs = []
        f1s = []
        for X_train, Y_train, X_test, Y_test in train_sampler.apply_kfold(self.args.n_splits):
            logging.info("\t Fold {}".format(self.args.repeat))

            #
            # Resampling level
            #
            resamplar = CustomResamplar(X_train, Y_train)
            params = param_loader()
            sam_ratio = params[imp][rsp][clf] / 10 if params[imp][rsp][clf] else 1
            if resamplar.need_resample(sam_ratio):
                logging.info("\t Re-Sampling Started")
                X_train, Y_train = resamplar.resample(self.args, rsp=rsp, ratio=sam_ratio)
                logging.info("\t Re-Sampling Done")

            #
            # Classification level
            #
            logging.info("\t Training in fold {} Start".format(self.args.repeat))
            trainer = CustomClassifier(self.args)
            trainer.train(X_train, Y_train, clf=clf)
            logging.info("\t Training in fold {} Done".format(self.args.repeat))

            # Validation of the result
            trainer.predict(X_test, Y_test)

            f1s.append(trainer.f1)
            precisions.append(trainer.precision)
            recalls.append(trainer.recall)
            w_precisions.append(trainer.w_precision)
            w_recalls.append(trainer.w_recall)
            if hasattr(trainer.clf, 'predict_proba'):
                aurocs.append(trainer.auroc)
                auprcs.append(trainer.auprc)

            del trainer

            self.args.repeat += 1
        if self.args.metric == "auroc":
            return average(aurocs)
        elif self.args.metric == "macro_f1":
            return average(f1s)
        else:
            raise ValueError("Metric {} not yet supported".format(self.args.metric))


if __name__ == "__main__":
    class Args:
        def __init__(self):
            self.train_ratio=0.2
            self.n_splits = 10
            self.repeat = 0
    args = Args()
    run_pipe = RunPipe(args)
    run_pipe.fit("MIRACLE", "mwmote", "lr")