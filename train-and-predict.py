#!/usr/bin/env python3
import random
import math
import itertools
import logging
import sys
from statistics import median, mode

from train_data_explore import read_data, feature_labels
from parse_chinese_data import (sample_dict, verify_training,
                                generate_roc_graph, tree_as_pdf)
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)
random.seed(42)


time_logger = logging.getLogger('parse_emails')
time_logger.setLevel(logging.WARNING)

def sample_matrix(m, keep_portion=0.5):
    height, _w = m.shape
    closest_sublen = math.ceil(height * keep_portion)
    m_as_list = m.tolist()
    selected = m_as_list[:closest_sublen]
    deselected = m_as_list[closest_sublen:]
    return np.matrix(selected), np.matrix(deselected)


def split_data(data):
    broken, ok = [], []

    for row in data:
        columns = []
        for k, v in sorted(row.items()):
            if not k == 'is_broken':
                columns.append(v)
        if row['is_broken'] == 1:
            broken.append(columns)
        else:
            ok.append(columns)
    return np.matrix(broken), np.matrix(ok)


def predict(broken, ok_disks, keep_broken, keep_nonbroken,
            classifier=tree.DecisionTreeClassifier()):
    training_broken, witheld_broken = sample_matrix(broken,
                                                    keep_portion=keep_broken)
    training_ok, witheld_ok = sample_matrix(ok_disks,
                                            keep_portion=keep_nonbroken)
    training_set = np.append(training_ok, training_broken, axis=0)
    verification_set = np.append(witheld_broken, witheld_ok, axis=0)

    expected_labels = list((witheld_broken.shape[0] * [-1])
                           + (witheld_ok.shape[0] * [1]))


    labels = list((training_ok.shape[0] * [1])
                  + (training_broken.shape[0] * [-1]))
    classifier = classifier.fit(training_set, labels)

    return verify_training(classifier, verification_set, expected_labels)


def find_best_training_proportion(broken, ok):
    best_tpr = 0
    best_tpr_far = float("inf")
    best_far = float("inf")
    best_tpr_combination = None
    best_far_combination = None
    keep_ok_p = range(1, 76, 1)
    keep_broken_p = range(19, 76, 1)

    highest_tolerable_far = 0.1
    lowest_tolerable_tpr = 0.5

    for ok_p, broken_p in itertools.product(keep_ok_p, keep_broken_p):
        tpr, far, _tree = predict(broken, ok, keep_broken=broken_p/100,
                                  keep_nonbroken=ok_p/100)
        if far <= 0.001 or tpr >= 0.999:
            # Ignoring perfect outcome
            continue

        if tpr > best_tpr or abs(tpr - best_tpr) < 0.01 and far <= highest_tolerable_far:
            if tpr > best_tpr or far < best_tpr_far:
                best_tpr_far = far
                best_tpr = tpr
                best_tpr_combination = ok_p, broken_p
                print("New best TPR! {:.3f} OK% {} Broken% {} (FAR was {:.3f})"
                      .format(tpr, ok_p, broken_p, far))

        if far < best_far and tpr >= lowest_tolerable_tpr:
            best_far = far
            best_far_combination = ok_p, broken_p
            print("New best FAR! {:.3f} OK% {} Broken% {} (TPR was {:.3f})"
                  .format(far, ok_p, broken_p, tpr))

    print("Best TPR was {:.2f}, with arguments ok/broken {}"
          .format(best_tpr, best_tpr_combination))
    print("Best FAR was {:.2f}, with arguments ok/broken {}"
          .format(best_far, best_far_combination))
    return best_tpr_combination


if __name__ == '__main__':
    task = sys.argv[1]

    disk_data = read_data()
    random.shuffle(disk_data)
    broken, ok = split_data(disk_data)

    if task == "best_settings":
        ok_p, broken_p = find_best_training_proportion(broken, ok)
        tpr, far, _tree = predict(broken, ok, keep_broken=broken_p/100,
                                  keep_nonbroken=ok_p/100)
        print("Result: TPR: {}, FAR {} at mix {}% from failed set, {}% from OK set"
              .format(tpr, far, broken_p, ok_p))
    elif task == "roc_graph":
        generate_roc_graph(broken, ok, target_file="../Report/Graphs/roc_graph_disks.tex",
                           predict=predict)
    elif task == "predict":
        broken_p = int(sys.argv[2])
        ok_p = int(sys.argv[3])
        tpr, far, _tree = predict(broken, ok, keep_broken=broken_p/100,
                                  keep_nonbroken=ok_p/100)
        print("Result: TPR: {}, FAR {} at mix {}% from failed set, {}% from OK set"
              .format(tpr, far, broken_p, ok_p))

    elif task == "random_forest":
        broken_p = int(sys.argv[2])
        ok_p = int(sys.argv[3])
        tpr, far, _tree = predict(broken, ok, keep_broken=broken_p/100,
                                  keep_nonbroken=ok_p/100,
                                  classifier=RandomForestClassifier())
        print("Result: TPR: {}, FAR {} at mix {}% from failed set, {}% from OK set"
              .format(tpr, far, broken_p, ok_p))
    elif task == "tree":
        try:
            broken_p = int(sys.argv[2])
            ok_p = int(sys.argv[3])
        except (IndexError, ValueError):
            broken_p = 67
            ok_p = 1
        tpr, far, t = predict(broken, ok, keep_broken=broken_p/100,
                                keep_nonbroken=ok_p/100)

        target_file = "../Report/Graphs/disks_tree.pdf"

        labels = feature_labels()
        labels.remove('is_broken')
        tree_as_pdf(t, target_file, feature_names=labels,
                    class_names=["Failed", "OK"])
        print("Rendered tree with TPR: {}, FAR {} at mix {}% from failed set, {}% from OK set"
              .format(tpr, far, broken_p, ok_p))
