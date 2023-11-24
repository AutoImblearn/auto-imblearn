import os

import numpy as np

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import logging

from resamplers.autosmote.data_loading import get_data
from resamplers.autosmote.classifiers import get_clf
from resamplers.autosmote.rl.training import train

class Arguments:
    def __init__(self):
        self.dataset = "imp_knn_.p"
        self.seed = 1
        self.clf = "svm"
        self.metric = "auroc"

        self.device = "cpu"
        self.cuda = "0"

        self.xpid = "AutoSMOTE"
        self.undersample_ratio = 100

        self.num_instance_specific_actions = 10
        self.num_max_neighbors = 30
        self.cross_instance_scale = 4
        self.savedir = "logs"
        self.num_actors = 40
        self.total_steps = 500
        self.batch_size = 8
        self.cross_instance_unroll_length = 2
        self.instance_specific_unroll_length = 300
        self.low_level_unroll_length = 300
        self.num_buffers = 20

        # Loss settings.
        self.entropy_cost = 0.0006
        self.baseline_cost = 0.5
        self.discounting = 1.0

        # Optimizer settings.
        self.learning_rate = 0.005
        # self.learning_rate = 0.0005
        self.grad_norm_clipping = 40.0

class RunAutoSmote:
    def __init__(self):
        self.flags = Arguments()
        os.environ["CUDA_VISIBLE_DEVICES"] = self.flags.cuda
        np.random.seed(self.flags.seed)

    def fit(self, clf="ada", imp="gain", metric="auroc", train_ratio=1.0):
        imp = "imp_" + imp + "_.p"
        self.flags.clf = clf
        self.flags.dataset = imp
        self.flags.metric = metric

        train_X, train_y, val_X, val_y, test_X, test_y = get_data(
            name=self.flags.dataset,
            val_ratio=0.2,
            test_raito=0.2,
            undersample_ratio=self.flags.undersample_ratio,
            train_ratio=train_ratio
        )
        clf = get_clf(self.flags.clf)

        # Search space for ratios
        self.flags.ratio_map = [0.0, 0.25, 0.5, 0.75, 1.0]

        # Start training
        score = train(self.flags, train_X, train_y, val_X, val_y, test_X, test_y, clf)

        # print("Results:", self.flags.dataset, score)
        return score

    # def runpipe(self, clf=None, imp=None, metric="auroc"):
    #     imp = "imp_" + imp + "_.p"
    # command = ['python', 'train.py', '--device=cpu', '--clf={}'.format(clf), '--dataset={}'.format(imp),
    #            '--metric={}'.format(metric)]
    # command = " python train.py --device=cpu --clf={} --dataset={} --metric={}".format(clf, imp, metric)
    # command = "cd /home/hongkuan/Projects/AutoImblearn/src/resamplers && source /home/hongkuan/anaconda3/etc/profile.d/conda.sh && conda activate NHANES && python train.py --device=cpu --clf={} --dataset={} --metric={}".format(clf, imp, metric)
    # os.system(command)

    # work_dir = os.path.join(os.getcwd(), "resamplers")
    # p = subprocess.Popen(command, cwd=work_dir, shell=True, executable='/bin/bash')
    # process = subprocess.Popen('/bin/bash', stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=work_dir)
    # out, err = process.communicate(command)

    # p.wait()
    # out, _ = p.communicate()

if __name__ == "__main__":
    logging.basicConfig(filename='cvd.log', level=logging.DEBUG,
                        format='%(asctime)s:%(levelname)s:%(message)s')

    warnings.filterwarnings("ignore")
    run_autosmote = RunAutoSmote()
    run_autosmote.fit(clf="mlp", imp="MIRACLE", metric="auroc", train_ratio=0.2)
