from sklearn.svm import LinearSVC
import numpy as np
from tqdm import tqdm

from utils import compute_metrics


class LocalSVM:
    def __init__(self, datasets_train, datasets_test, datasets_val, seed=54):
        self.train_set = datasets_train
        self.test_set = datasets_test
        self.val_set = datasets_val
        np.random.seed(seed)
        self.seed = seed
        self.svm_classifiers = [None] * len(self.train_set)
        for idx, train_data in enumerate(self.train_set):
            x, y = train_data
            clf = None
            if len(np.unique(y)) > 1:
                clf = LinearSVC(dual=False, random_state=self.seed)
            self.svm_classifiers[idx] = clf

    def fit_all_client(self):
        pbar = tqdm(total=len(self.svm_classifiers))
        for idx, clf in enumerate(self.svm_classifiers):
            if clf is None:
                pbar.update(1)
                continue
            x, y = self.train_set[idx]
            clf.fit(x, y)
            pbar.update(1)

    def calculate_measures(self, cross_evaluate=True):
        all_train_measures = {'accuracy': [], 'fpr': [], 'tpr': [], 'ber': [], 'loss': []}
        all_val_measures = {'accuracy': [], 'fpr': [], 'tpr': [], 'ber': [], 'loss': []}

        if cross_evaluate:
            count = 0
            for clf in self.svm_classifiers:
                if clf is None:
                    continue
                count += 1
                for x_train, y_train in self.train_set:
                    y_pred_train = clf.predict(x_train)
                    accuracy, loss, fpr, tpr, ber = compute_metrics(y_train, y_pred_train)
                    all_train_measures['accuracy'].append(accuracy)
                    all_train_measures['loss'].append(loss)
                    all_train_measures['fpr'].append(fpr)
                    all_train_measures['tpr'].append(tpr)
                    all_train_measures['ber'].append(ber)

                for x_val, y_val in self.val_set:
                    y_pred_val = clf.predict(x_val)
                    accuracy, loss, fpr, tpr, ber = compute_metrics(y_val, y_pred_val)
                    all_val_measures['accuracy'].append(accuracy)
                    all_val_measures['loss'].append(loss)
                    all_val_measures['fpr'].append(fpr)
                    all_val_measures['tpr'].append(tpr)
                    all_val_measures['ber'].append(ber)

            global_train_measures = {}
            global_val_measures = {}
            for key in all_train_measures:
                global_train_measures[key] = (sum(all_train_measures[key][idx] for idx, _ in
                                                  enumerate(all_train_measures['accuracy']))
                                              / (len(self.train_set) * count))
                global_val_measures[key] = (sum(all_val_measures[key][idx] for idx, _ in
                                                enumerate(all_val_measures['accuracy'])) / (len(self.val_set) * count))
        else:
            for idx, clf in enumerate(self.svm_classifiers):
                if clf is None:
                    continue
                x_train, y_train = self.train_set[idx]
                y_pred_train = clf.predict(x_train)
                y_prob_train = clf._predict_proba_lr(x_train)
                accuracy, loss, fpr, tpr, ber = compute_metrics(y_train, y_pred_train, y_prob_train)
                all_train_measures['accuracy'].append(accuracy)
                all_train_measures['loss'].append(loss)
                all_train_measures['fpr'].append(fpr)
                all_train_measures['tpr'].append(tpr)
                all_train_measures['ber'].append(ber)

            for idx, clf in enumerate(self.svm_classifiers):
                if clf is None:
                    continue
                x_val, y_val = self.val_set[idx]
                y_pred_val = clf.predict(x_val)
                y_prob_val = clf._predict_proba_lr(x_train)
                accuracy, loss, fpr, tpr, ber = compute_metrics(y_val, y_pred_val, y_prob_val)
                all_val_measures['accuracy'].append(accuracy)
                all_val_measures['loss'].append(loss)
                all_val_measures['fpr'].append(fpr)
                all_val_measures['tpr'].append(tpr)
                all_val_measures['ber'].append(ber)

            print("max:", max(all_val_measures['accuracy']))
            print("min:", min(all_val_measures['accuracy']))
            min_v = min(all_val_measures['accuracy'])
            index = all_val_measures['accuracy'].index(min_v)
            print("min_index", index)

            global_train_measures = {}
            global_val_measures = {}
            for key in all_train_measures:
                global_train_measures[key] = sum(all_train_measures[key][idx] for idx, _ in
                                                 enumerate(all_train_measures['accuracy'])) / len(self.train_set)
                global_val_measures[key] = sum(all_val_measures[key][idx] for idx, _ in
                                               enumerate(all_val_measures['accuracy'])) / len(self.val_set)



        print("Global Train Set Loss:", global_train_measures['loss'])
        print("Global Train Set Accuracy:", global_train_measures['accuracy'])
        print("Global Train Set FPR:", global_train_measures['fpr'])
        print("Global Train Set TPR:", global_train_measures['tpr'])
        print("Global Train Set BER:", global_train_measures['ber'])
        print("Global Validation Set Loss:", global_val_measures['loss'])
        print("Global Validation Set Accuracy:", global_val_measures['accuracy'])
        print("Global Validation Set FPR:", global_val_measures['fpr'])
        print("Global Validation Set TPR:", global_val_measures['tpr'])
        print("Global Validation Set BER:", global_val_measures['ber'])

        return global_train_measures, global_val_measures

    def calculate_measures_test_set(self, cross_evaluate=True):
        all_test_measures = {'accuracy': [], 'fpr': [], 'tpr': [], 'ber': [], 'loss': []}
        if cross_evaluate:
            count = 0
            for clf in self.svm_classifiers:
                if clf is None:
                    continue
                count += 1
                for x_val, y_val in self.val_set:
                    y_pred_val = clf.predict(x_val)
                    accuracy, loss, fpr, tpr, ber = compute_metrics(y_val, y_pred_val)
                    all_test_measures['accuracy'].append(accuracy)
                    all_test_measures['loss'].append(loss)
                    all_test_measures['fpr'].append(fpr)
                    all_test_measures['tpr'].append(tpr)
                    all_test_measures['ber'].append(ber)
                global_test_measures = {}
                for key in all_test_measures:
                    global_test_measures[key] = (sum(all_test_measures[key][idx] for idx, _ in
                                                     enumerate(all_test_measures['accuracy']))
                                                 / (len(self.test_set) * count))
        else:
            for idx, clf in enumerate(self.svm_classifiers):
                if clf is None:
                    continue
                x_val, y_val = self.val_set[idx]
                y_pred_val = clf.predict(x_val)
                accuracy, loss, fpr, tpr, ber = compute_metrics(y_val, y_pred_val)
                all_test_measures['accuracy'].append(accuracy)
                all_test_measures['loss'].append(loss)
                all_test_measures['fpr'].append(fpr)
                all_test_measures['tpr'].append(tpr)
                all_test_measures['ber'].append(ber)

            global_test_measures = {}
            for key in all_test_measures:
                global_test_measures[key] = sum(all_test_measures[key][idx] for idx, _ in
                                                enumerate(all_test_measures['accuracy'])) / len(self.test_set)

        return global_test_measures
