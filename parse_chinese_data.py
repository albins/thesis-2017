#!/usr/bin/env python3
import common

import csv
import random
import math
import sys
import logging

from collections import defaultdict
from parse_emails import timed

import numpy as np
from sklearn import tree
# from sklearn.ensemble import RandomForestClassifier
# import matplotlib.pyplot as plt

log = logging.getLogger(__name__)


FILENAME = "../Data/Disk_SMART_dataset.csv"
BROKEN_DISKS_RETAIN_RATE = 0.70
NON_BROKEN_DISKS_RETAIN_RATE = 0.05

FEATURE_LABELS = [
    "Raw Read Error Rate",
    "Spin Up Time",
    "Reallocated Sectors Count",
    "Seek Error Rate",
    "Power On Hours",
    "Reported Uncorrectable Errors",
    "High Fly Writes",
    "Temperature Celsius",
    "Hardware ECC Recovered",
    "Current Pending Sector Count",
    "Reallocated Sectors Count",
    "Current Pending Sector Count"
]

CLASS_NAMES = ["Failed", "OK"]


TEX_PREAMBLE = """
\\begin{tikzpicture}
  \\begin{axis}[
   xmode=log,
   % ymode=log,
    xlabel={False Positive Rate},
    ylabel={True Positive Rate},
    cycle list/Set1-6,
            % define fill color for the marker
    mark list fill={.!75!white}, cycle multiindex* list={ Set1-6
      \\nextlist [3 of]linestyles \\nextlist very thick \\nextlist mark=o
      solid, mark=square solid, mark=triangle, mark=*, mark=+ }, ]

    \\addplot[
    scatter,
    only marks,
    nodes near coords,
    point meta=explicit symbolic]
    coordinates {
"""
TEX_POST_STUFF = """
    };
  \\end{axis}
\\end{tikzpicture}
"""

random.seed(93)


def sample_dict(d, keep_portion=0.5):
    keys = list(d.keys())
    total = len(keys)
    closest_sublen = math.ceil(total * keep_portion)
    selected_keys = keys[:closest_sublen]
    nonselected_keys = keys[closest_sublen:]

    result = []
    rest = []
    for key in selected_keys:
        for row in d[key]:
            result.append(row)
    for key in nonselected_keys:
        for row in d[key]:
            rest.append(row)
    return result, rest


def read_line(line, broken, ok):
    formatted_line = [float(i) for i in line]
    serial = line[0]
    if formatted_line[1] < 0:
        broken[serial].append(line)
    else:
        ok[serial].append(line)


def clean_line(line):
    # remove broken/non-broken and serial
    return line[2:]


def verify_training(clf, verification_set, expected_labels):
    false_positives = 0
    false_negatives = 0
    true_positives = 0
    true_negatives = 0

    with timed(task_name="verification"):
        for prediction, expected in zip(clf.predict(verification_set),
                                        expected_labels):
            if prediction == expected:
                #log.debug("Found correct prediction of %s",
                #          common.NUM_TO_PREDICTION[prediction])
                if expected == common.PREDICT_FAIL:
                    true_positives += 1
                else:
                    true_negatives += 1
            else:
                #log.debug("Found misprediction, got %s expected %s",
                #          common.NUM_TO_PREDICTION[prediction],
                #          common.NUM_TO_PREDICTION[expected])
                if expected == common.PREDICT_FAIL:
                    false_negatives += 1
                else:
                    false_positives += 1

    tpr, far = common.calculate_tpr_far(true_positives, true_negatives,
                                        false_positives, false_negatives)
    log.debug("False positives: %d, false negatives: %d, true positives: %d, true negatives: %d",
             false_positives, false_negatives, true_positives, true_negatives)
    return tpr, far, clf


def predict(broken, ok_disks, keep_broken, keep_nonbroken,
            classifier=tree.DecisionTreeClassifier):
    time_record = []
    with timed(task_name="Partition results", time_record=time_record, printer=print):
        training_broken, witheld_broken = sample_dict(broken,
                                                      keep_portion=keep_broken)
        training_ok, witheld_ok = sample_dict(ok_disks,
                                              keep_portion=keep_nonbroken)
        training_set = np.asarray(list(map(clean_line, training_broken))
                                  + list(map(clean_line, training_ok)))
        verification_set = np.asarray(list(map(clean_line, witheld_broken))
                                      + list(map(clean_line, witheld_ok)))

    expected_labels = list((len(witheld_broken) * [-1])
                           + (len(witheld_ok) * [1]))

    with timed(task_name="Training", time_record=time_record, printer=print):
        labels = list((len(training_broken) * [-1]) + (len(training_ok) * [1]))
        clf = classifier()
        clf = clf.fit(training_set, labels)

    return verify_training(clf, verification_set, expected_labels)


def generate_roc_graph(broken, ok,
                       target_file="../Report/Graphs/roc_graph.tex",
                       predict=predict, start_percentage=1,
                       stop_percentage=75,
                       step_size=5,
                       broken_percent=75):
    xs_and_ys = []

    for n in range(start_percentage, stop_percentage + 1, step_size):
        log.debug("Calculating ROC values for %d%%", n)
        xs_and_ys.append(calculate_roc_point(n, broken, ok,
                                             broken_percent=broken_percent,
                                             predict=predict))

    print(sorted(xs_and_ys))
    ys, xs, _vals = zip(*xs_and_ys)

    with open(target_file, 'w') as f:
       f.write(TEX_PREAMBLE)
       for tpr, far, value in xs_and_ys:
           line = "({}, {}) [{}]\n".format(far, tpr, round(value * 100))
           f.write(line)
       f.write(TEX_POST_STUFF)


def tree_as_pdf(t, target_file, feature_names, class_names):
    import pydotplus
    dot_data = tree.export_graphviz(t, out_file=None,
                                    feature_names=feature_names,
                                    class_names=class_names,
                                    rounded=True,
                                    filled=True,
                                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf(target_file)


def render_tree_pdf(broken, ok_disks, keep_nonbroken):
    target_file = "../Report/Graphs/sample_tree.pdf"
    tpr, far, t = predict(broken, ok_disks, keep_broken=0.75,
                          keep_nonbroken=keep_nonbroken)
    tree_as_pdf(t, target_file, FEATURE_LABELS, CLASS_NAMES)


def calculate_roc_point(n, broken, ok, classifier=tree.DecisionTreeClassifier,
                        predict=predict, broken_percent=75):
    percentage = 0.01 * n
    tpr, far, _tree = predict(broken, ok, keep_broken=broken_percent/100,
                              keep_nonbroken=percentage,
                              classifier=classifier)
    return tpr, far, percentage


if __name__ == '__main__':
    broken = defaultdict(list)
    ok = defaultdict(list)
    time_record = []
    task = sys.argv[1]
    #p = multiprocessing.Pool(4)

    with timed(task_name="Parse file", time_record=time_record):
        with open(FILENAME, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for i, row in enumerate(reader):
                read_line(row, broken, ok)

    print("Read {} lines in {}s".format(i, time_record[0]))

    if task == "tree" or task == "all":
        render_tree_pdf(broken, ok, 0.16)
    if task == "roc_graph" or task == "all":
        generate_roc_graph(broken, ok)
