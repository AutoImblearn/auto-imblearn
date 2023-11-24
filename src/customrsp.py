from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, OneSidedSelection, CondensedNearestNeighbour
from imblearn.combine import SMOTEENN

import logging
import numpy as np
import pandas as pd

from resamplers.mwmote import MWMOTE

rsps = {
    'under': RandomUnderSampler(random_state=42),
    'over': RandomOverSampler(random_state=42),
    'smote': SMOTE(random_state=42),
    'mwmote': MWMOTE(random_state=42),
    # 'combined': SMOTEENN(random_state=42, n_jobs=-1),
}

def value_counter(Y: np.ndarray):
    values, counts = np.unique(Y, return_counts=True)
    for value, count in zip(values, counts):
        dist = count / Y.shape[0] * 100
        logging.info("\t\t Class={}, n={},\t ({:.2f}%)".format(value, count, dist))


class CustomResamplar:
    """ Resamplar oject to re-sample imbalanced data
        Args:
            X (np.ndarray):
            Y (np.ndarray):
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray):

        self.X = X
        self.Y = Y

    def need_resample(self, samratio):
        """ Test if need re-sample
        """
        _, counts = np.unique(self.Y, return_counts=True)
        ratio = counts[1] / counts[0]

        return ratio < samratio

    def resample(self, args,  rsp=None, ratio=None):
        logging.info("\t Before Re-Sampling")
        value_counter(self.Y)
        if rsp in rsps.keys():
            resampler = rsps[rsp]
            if ratio is not None and hasattr(resampler, 'set_params'):
                resampler.set_params(**{"sampling_strategy": ratio})
            else:
                raise ValueError("can't set resampling ratio for {}".format(rsp))
        else:
            raise ValueError("Re-sampling method {} is not defined".format(rsp))

        self.X, self.Y = resampler.fit_resample(self.X, self.Y)
        logging.info("\t After Re-Sampling")
        value_counter(self.Y)

        return self.X, self.Y
