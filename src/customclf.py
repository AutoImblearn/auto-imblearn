import logging

import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, classification_report, \
    average_precision_score

from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier

from customrsp import value_counter

clfs = {
    "lr": LogisticRegression(random_state=42, max_iter=500, solver='lbfgs'),
    "mlp": MLPClassifier(alpha=1, random_state=42, max_iter=1000),
    "ada": AdaBoostClassifier(random_state=42),
    "svm": svm.SVC(random_state=42, probability=True, kernel='linear'),
    # "SVM": svm.SVC(random_state=42, probability=True, kernel='poly'),
    # "rf": RandomForestClassifier(random_state=42),
    # "ensemble": XGBClassifier(learning_rate=1.0, max_depth=10, min_child_weight=15, n_estimators=100, n_jobs=1, subsample=0.8, verbosity=0),
    # "bst": GradientBoostingClassifier(random_state=42),
}

class CustomClassifier:
    def __init__(self, args):
        # def __init__(self, X: pd.DataFrame, Y: pd.DataFrame):
        self.args = args
        self.f1 = None
        self.precision = None
        self.recall = None
        self.w_precision = None
        self.w_recall = None
        self.auroc = None
        self.auprc = None

        self.c_index = None
        self.clf = None

    def train(self, X_train: np.ndarray, Y_train: np.ndarray, clf=None):
        # Train classifier
        if clf in clfs.keys():
            self.clf = clfs[clf]
        else:
            raise "Model {} not defined in model.py".format(clf)

        self.clf.fit(X_train, Y_train)

    def predict(self, X_test: np.ndarray, Y_test: np.ndarray):
        # Test model performance
        logging.info("\t\t Data distribution for testing")
        value_counter(Y_test)

        Y_pred = self.clf.predict(X_test)

        logging.info("\t\t Data distribution for prediction")
        value_counter(Y_pred)

        precision, recall, _, _ = (
            precision_recall_fscore_support(Y_test, Y_pred, average='binary'))
        w_precision, w_recall, _, _ = (
            precision_recall_fscore_support(Y_test, Y_pred, average='weighted'))
        _, _, f1, _ = (
            precision_recall_fscore_support(Y_test, Y_pred, average='macro'))

        self.precision = precision
        self.recall = recall
        self.w_precision = w_precision
        self.w_recall = w_recall
        self.f1 = f1

        if hasattr(self.clf, 'predict_proba'):
            Y_proba = self.clf.predict_proba(X_test)[:, 1]
            auroc = roc_auc_score(Y_test, Y_proba)
            # auprc = average_precision_score(Y_test, Y_proba)
            # self.auprc = auprc
            self.auroc = auroc

            logging.info(
                "\t AUROC {:.4f}. Precision {:.2f}% and Recall {:.2f}%. Real precision: {:.2f}% and recall: {:.2f}%. Number of true death: {}, predicted death: {}".format(
                    auroc, w_precision * 100, w_recall * 100, precision * 100, recall * 100,
                    Y_test.sum(), Y_pred.sum()))
        else:
            logging.info(
                "\t Precision {:.2f}% and Recall {:.2f}%. Real precision: {:.2f}% and recall: {:.2f}%. Number of true death: {}, predicted death: {}".format(
                    w_precision * 100, w_recall * 100, precision * 100, recall * 100,
                    Y_test.sum(), Y_pred.sum()))

        if np.sum(Y_pred) == 0:
            logging.debug("\t No one is predicted die because of CVD")
