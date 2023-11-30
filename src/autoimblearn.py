import secrets
from runpipe import RunPipe
from customimputation import imps
from customclf import clfs
from customrsp import rsps
from runautosmote import RunAutoSmote
import os
import pickle


def is_checked(pipe, checked):
    imp, rsp, clf = pipe
    if imp not in checked:
        checked[imp] = {}
    if rsp not in checked[imp]:
        checked[imp][rsp] = {}
    if clf not in checked[imp][rsp]:
        checked[imp][rsp][clf] = None
    if checked[imp][rsp][clf] is None:
        return False
    else:
        return True

class Result:
    def __init__(self, train_ratio, metric):
        saved_file_name = "saved_pipe_{}_{}.p".format(metric, str(train_ratio))
        self.saved_file_path = os.path.join("..", "data", "processed", saved_file_name)
        self.saved_result = None

    def load_saved_result(self):
        # load data from file
        if os.path.isfile(self.saved_file_path):
            with open(self.saved_file_path, "rb") as f:
                self.saved_result = pickle.load(f)
        else:
            self.saved_result = {}

    def is_in(self, pipe):
        # Check if pipe in result
        imp, rsp, clf = pipe
        if imp not in self.saved_result:
            self.saved_result[imp] = {}
        if rsp not in self.saved_result[imp]:
            self.saved_result[imp][rsp] = {}
        if clf not in self.saved_result[imp][rsp]:
            self.saved_result[imp][rsp][clf] = None
        if self.saved_result[imp][rsp][clf] is None:
            return False
        else:
            return True

    def append(self, pipe, score):
        # append pipeline into result
        imp, rsp, clf = pipe
        self.saved_result[imp][rsp][clf] = score
        self.save2file()

    def get(self, pipe):
        # get the result
        imp, rsp, clf = pipe
        return self.saved_result[imp][rsp][clf]

    def save2file(self):
        # save result into file
        if self.saved_result is None:
            raise ValueError("Please create saved_result first before save.")
        with open(self.saved_file_path, "wb") as f:
            pickle.dump(self.saved_result, f)


class AutoImblearn:
    def __init__(self, run_pipe: RunPipe, metric):
        self.resamplers = list(rsps.keys())
        self.resamplers.append("autosmote")
        self.classifiers = list(clfs.keys())
        self.imputers = imps
        self.run_pipe = run_pipe
        self.metric = metric

    def find_best(self, checked=None, train_ratio=1.0):
        # Get the saved result
        saver = Result(train_ratio, self.metric)
        saver.load_saved_result()

        # find the best pipeline
        tmp_pipe = [None, None, None]
        if checked is None:
            checked = {}
        if tmp_pipe[0] is None:
            tmp_pipe[0] = secrets.choice(self.imputers)
        if tmp_pipe[1] is None:
            tmp_pipe[1] = secrets.choice(self.resamplers)
        if tmp_pipe[2] is None:
            tmp_pipe[2] = secrets.choice(self.classifiers)
        counter = 0
        # TODO add best check
        best_pipe = set([])
        best_score = 0
        final_result = set([])
        while True:

            # Brute force method
            # Choose imputation
            for imp in self.imputers:
                pipe = [imp, tmp_pipe[1], tmp_pipe[2]]
                print(pipe)
                if is_checked(pipe, checked):
                    tmp = checked[imp][tmp_pipe[1]][tmp_pipe[2]]
                else:
                    if saver.is_in(pipe):
                        tmp = saver.get(pipe)
                    else:
                        if tmp_pipe[1] == "autosmote":
                            run_autosmote = RunAutoSmote()
                            tmp = run_autosmote.fit(clf=tmp_pipe[2], imp=imp, metric=self.metric, train_ratio=train_ratio)
                        else:
                            tmp = self.run_pipe.fit(pipe)
                        saver.append(pipe, tmp)
                    checked[imp][tmp_pipe[1]][tmp_pipe[2]] = tmp
                    counter += 1

                if tmp > best_score:
                    tmp_pipe[0] = imp
                    best_pipe = set(tmp_pipe)
                    best_score = tmp
                print("Current pipe: {}, counter: {}, best pipe: {}, best result: {}".format(tmp, counter, best_pipe, best_score))

            # Choose resampler
            for resampler in self.resamplers:
                pipe = [tmp_pipe[0], resampler, tmp_pipe[2]]
                print(pipe)
                if is_checked(pipe, checked):
                    tmp = checked[tmp_pipe[0]][resampler][tmp_pipe[2]]
                else:
                    if saver.is_in(pipe):
                        tmp = saver.get(pipe)
                    else:
                        if resampler == "autosmote":
                            run_autosmote = RunAutoSmote()
                            tmp = run_autosmote.fit(clf=tmp_pipe[2], imp=tmp_pipe[0], metric=self.metric, train_ratio=train_ratio)
                        else:
                            tmp = self.run_pipe.fit(pipe)
                        saver.append(pipe, tmp)
                    checked[tmp_pipe[0]][resampler][tmp_pipe[2]] = tmp
                    counter += 1

                if tmp > best_score:
                    tmp_pipe[1] = resampler
                    best_pipe = set(tmp_pipe)
                    best_score = tmp
                print("Current pipe: {}, counter: {}, best pipe: {}, best result: {}".format(tmp, counter, best_pipe, best_score))

            # Choose classifier
            for classifier in self.classifiers:
                pipe = [tmp_pipe[0], tmp_pipe[1], classifier]
                print(pipe)
                if is_checked(pipe, checked):
                    tmp = checked[tmp_pipe[0]][tmp_pipe[1]][classifier]
                else:
                    if saver.is_in(pipe):
                        tmp = saver.get(pipe)
                    else:
                        if tmp_pipe[1] == "autosmote":
                            run_autosmote = RunAutoSmote()
                            tmp = run_autosmote.fit(clf=classifier, imp=tmp_pipe[0], metric=self.metric, train_ratio=train_ratio)
                        else:
                            tmp = self.run_pipe.fit(pipe)
                        saver.append(pipe, tmp)
                    checked[tmp_pipe[0]][tmp_pipe[1]][classifier] = tmp
                    counter += 1

                if tmp > best_score:
                    tmp_pipe[2] = classifier
                    best_pipe = set(tmp_pipe)
                    best_score = tmp
                print("Current pipe: {}, counter: {}, best pipe: {}, best result: {}".format(tmp, counter, best_pipe, best_score))

            if best_pipe == final_result:
                break
            else:
                final_result = best_pipe

        pipe = []
        for imp in self.imputers:
            if imp in best_pipe:
                pipe.append(imp)
                break
        for rsp in self.resamplers:
            if rsp in best_pipe:
                pipe.append(rsp)
                break
        for clf in self.classifiers:
            if clf in best_pipe:
                pipe.append(clf)
                break
        if len(pipe) != 3:
            raise ValueError("Some elements in pipeline {} is not yet supported".format(best_pipe))
        best_pipe = pipe
        saver.save2file()
        return best_pipe, counter, best_score

    def run_best(self, pipeline=None):
        # Re-run the best pipeline found with 100% of data
        saver = Result(1.0, self.metric)
        saver.load_saved_result()
        if saver.is_in(pipeline):
            result = saver.get(pipeline)
        else:
            if pipeline[1] == "autosmote":
                run_autosmote = RunAutoSmote()
                result = run_autosmote.fit(clf=pipeline[2], imp=pipeline[0], metric=self.metric, train_ratio=1.0)
            else:
                result = self.run_pipe.fit(pipeline, train_ratio=1.0)
            saver.append(pipeline, result)
        return result


    def count_pipe(self, pipeline=None):
        # Find the optimal and count how many pipelines to check
        counters = []
        for _ in range(100):
            checked = []
            final, count, best_score = self.find_best(checked)
            while final != set(pipeline):
                final, count, best_score = self.find_best(checked)
            counters.append(count)
        return counters
