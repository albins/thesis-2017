#!/usr/bin/env python3
import random
import math
import itertools
import logging
import sys
from statistics import median, mode, stdev
import argparse

from parse_chinese_data import (sample_dict, verify_training,
                                generate_roc_graph, tree_as_pdf)
import common
from common import sample_matrix

import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import daiquiri
log = daiquiri.getLogger()


np.random.seed(42)
random.seed(42)


def predict_best(*args, **kwargs):
    # Sort on highest TPR, then lowest FAR:
    results = predict(*args, **kwargs)
    results.sort(key=lambda tup: tup[1], reverse=False)
    results.sort(reverse=True, key=lambda tup: tup[0])
    return results[0]



def predict_worst(*args, **kwargs):
    results = predict(*args, **kwargs)
    results.sort(key=lambda tup: tup[1], reverse=True)
    results.sort(reverse=False, key=lambda tup: tup[0])
    return results[0]


def predict(broken, ok_disks, keep_broken, keep_nonbroken,
            classifier=tree.DecisionTreeClassifier, nrounds=None):
    training_broken, witheld_broken = sample_matrix(broken,
                                                    keep_portion=keep_broken)
    training_ok, witheld_ok = sample_matrix(ok_disks,
                                            keep_portion=keep_nonbroken)

    training_set = np.append(training_ok, training_broken, axis=0)
    verification_set = np.append(witheld_broken, witheld_ok, axis=0)
    log.debug("Training set size %dx%d (%d ok, %d broken)", *training_set.shape,
             training_ok.shape[0], training_broken.shape[0])
    log.debug("Verification set size %dx%d (%d ok, %d broken)",
             *verification_set.shape,
             witheld_ok.shape[0], witheld_broken.shape[0])

    expected_labels = list((witheld_broken.shape[0] * [common.PREDICT_FAIL])
                           + (witheld_ok.shape[0] * [common.PREDICT_OK]))

    labels = list((training_ok.shape[0] * [common.PREDICT_OK])
                  + (training_broken.shape[0] * [common.PREDICT_FAIL]))

    if not nrounds:
        with common.timed(task_name="training"):
            c = classifier(random_state=42)
            model = c.fit(training_set, labels)

            return verify_training(model, verification_set, expected_labels)
    else:
        results = []
        for i in range(0, nrounds):
            c = classifier(random_state=i)
            model = c.fit(training_set, labels)
            results.append(verify_training(model, verification_set,
                                           expected_labels))
        return results


def find_best_training_proportion(broken, ok, nrounds, broken_start, ok_start):
    MIN_DATAPOINTS_BROKEN = 3

    num_broken_drives, _ = broken.shape
    num_ok_drives, _ = ok.shape

    min_broken_percent = math.ceil(MIN_DATAPOINTS_BROKEN/num_broken_drives * 100)
    max_broken_percent = math.floor(100 - MIN_DATAPOINTS_BROKEN/ \
                                    num_broken_drives * 100)
    log.info("Determined OK broken percent range is [%d, %d]",
              min_broken_percent, max_broken_percent)

    min_ok_percent = math.ceil(100/num_ok_drives)
    log.info("Determined OK non-broken percent min is %d",
             min_ok_percent)

    best_tpr = 0
    best_tpr_far = float("inf")
    best_far = float("inf")
    best_tpr_combination = None
    best_far_combination = None
    keep_ok_p = range(max(min_ok_percent, ok_start), 76, 10)
    keep_broken_p = range(max(broken_start, min_broken_percent),
                          max_broken_percent + 1, 10)

    highest_tolerable_far = 0.1
    lowest_tolerable_tpr = 0.5

    for ok_p, broken_p in itertools.product(keep_ok_p, keep_broken_p):
        log.debug("Testing %d%% ok, %d%% broken data", ok_p, broken_p)
        tpr, far, _tree = predict_worst(broken, ok, keep_broken=broken_p/100,
                                        keep_nonbroken=ok_p/100, nrounds=nrounds)
        if far <= 0.001 or tpr >= 0.999:
            # Ignoring perfect outcome
            continue

        if tpr > best_tpr or abs(tpr - best_tpr) < 0.01 and far <= highest_tolerable_far:
            if tpr > best_tpr or far < best_tpr_far:
                best_tpr_far = far
                best_tpr = tpr
                best_tpr_combination = ok_p, broken_p
                log.info("New best TPR! {:.3f} OK% {} Broken% {} (FAR was {:.3f})"
                      .format(tpr, ok_p, broken_p, far))

        if far < best_far and tpr >= lowest_tolerable_tpr:
            best_far = far
            best_far_combination = ok_p, broken_p
            log.info("New best FAR! {:.3f} OK% {} Broken% {} (TPR was {:.3f})"
                  .format(far, ok_p, broken_p, tpr))

    print("Best TPR was {:.2f}, with arguments ok/broken {}"
          .format(best_tpr, best_tpr_combination))
    print("Best FAR was {:.2f}, with arguments ok/broken {}"
          .format(best_far, best_far_combination))
    return best_tpr_combination


def best_settings(ok, broken, args):
    ok_p, broken_p = find_best_training_proportion(broken, ok, args.nrounds,
                                                   broken_start=args.broken_start,
                                                   ok_start=args.ok_start)
    tpr, far, _tree = predict(broken, ok, keep_broken=broken_p/100,
                              keep_nonbroken=ok_p/100)
    print("Result: TPR: {}, FAR {} at mix {}% from failed set, {}% from OK set"
          .format(tpr, far, broken_p, ok_p))


def try_predict(ok, broken, args):
    broken_p = args.percent_broken
    ok_p = args.percent_ok
    if not args.do_random_forest:
        log.debug("Using normal classification tree")
        if not args.nrounds:
            tpr, far, _tree = predict(broken, ok, keep_broken=broken_p/100,
                                      keep_nonbroken=ok_p/100)
        else:
            results = predict(broken, ok, keep_broken=broken_p/100,
                              keep_nonbroken=ok_p/100, nrounds=args.nrounds)

    else:
        log.debug("Using random forest")
        tpr, far, _tree = predict(broken, ok, keep_broken=broken_p/100,
                                  keep_nonbroken=ok_p/100,
                                  classifier=RandomForestClassifier)
    if args.nrounds and not args.do_random_forest:
        tprs = [x[0] for x in results]
        fars = [x[1] for x in results]

        def int_mode(xs):
            return mode([int(x * 100)/100 for x in tprs])

        print("Result: TPR: [{}, {}] (stdev={}, mode={}, median={}), FAR [{}, {}] (stedv={}, mode={}, median={}) at mix {}% from failed set, {}% from OK set"
                  .format(min(tprs), max(tprs), stdev(tprs), int_mode(tprs), median(tprs),
                          min(fars),  max(fars), stdev(tprs), int_mode(tprs), median(tprs),
                          broken_p, ok_p))
    else:
        print("Result: TPR: {}, FAR {} at mix {}% from failed set, {}% from OK set"
              .format(tpr, far, broken_p, ok_p))


def make_tree(ok, broken, args):
    broken_p = args.percent_broken
    ok_p = args.percent_ok
    if args.nrounds:
        tpr, far, t = predict_best(broken, ok, keep_broken=broken_p/100,
                                   keep_nonbroken=ok_p/100, nrounds=args.nrounds)
    else:
        tpr, far, t = predict(broken, ok, keep_broken=broken_p/100,
                              keep_nonbroken=ok_p/100)

    target_file = args.writefile

    labels = args.feature_labels
    tree_as_pdf(t, target_file, feature_names=labels,
                class_names=["Failed", "OK"])
    log.info("Rendered tree with TPR: {}, FAR {} at mix {}% from failed set, {}% from OK set"
             .format(tpr, far, broken_p, ok_p))

def make_roc_graph(ok, broken, args):
    generate_roc_graph(broken, ok,
                       target_file=args.writefile,
                       predict=predict,
                       start_percentage=args.roc_start_p,
                       broken_percent=args.roc_broken_p)


def make_kmeans_graph(ok, broken, args):
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import scale
    from sklearn.decomposition import PCA

    all_data = np.append(ok, broken, axis=0)

    if args.use_at_most:
        use_at_most = min(args.use_at_most, all_data.shape[0])
    else:
        use_at_most = all_data.shape[0]

    if use_at_most < all_data.shape[0]:
        log.debug("Limiting data set to %d entries", use_at_most)
        np.random.shuffle(all_data)
        all_data = all_data[:use_at_most]

    from matplotlib.backends.backend_pdf import PdfPages
    pp = PdfPages(args.writefile)

    scaled_data = scale(all_data)
    reduced_data = PCA(n_components=2).fit_transform(scaled_data)
    kmeans = KMeans(init='k-means++', n_clusters=2, n_init=10)
    kmeans.fit(reduced_data)
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')
    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    #plt.title('K-means clustering disk dataset (PCA-reduced data)\n'
    #          'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    #plt.show()
    pp.savefig()
    pp.close()
    log.info("Wrote to {}".format(args.writefile))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    common.setup_verbosity_flags(parser)
    parser.add_argument("--normalise", "-n", action="store_true",
                        help="normalise data before processing",
                        dest="do_normalise", default=False)
    parser.add_argument("--as-is", "-a", action="store_true",
                        help="disable shuffling and use data as-is",
                        dest="dont_shuffle", default=False)
    parser.add_argument("--dataset", "-d",
                        dest="use_dataset",
                        type=str,
                        help="use this dataset",
                        default="cern_disks",
                        choices=["cern_disks", "zhu_disks", "random",
                                 "cern_bad_blocks"])
    common.add_subcommands(parent=parser,
                           descriptions=[
                               ('best_settings',
                                "Determine the best settings",
                                [
                                    (['-n','--nrounds'],
                                     {'type': int,
                                      'help': "Run n iterations in stead of one",
                                      'dest': 'nrounds',
                                      'default': 1}),
                                    (['-b','--broken-start-percent'],
                                     {'type': int,
                                      'help': "Start trying at this percentage",
                                      'dest': 'broken_start',
                                      'default': 1}),
                                    (['-o','--ok-start-percent'],
                                     {'type': int,
                                      'help': "Start trying at this percentage",
                                      'dest': 'ok_start',
                                      'default': 1}),
                                ],
                                best_settings),
                               ('predict',
                                "Try predicting failures from the set",
                                [
                                    (['percent_broken'], {'type': int}),
                                    (['percent_ok'], {'type': int}),
                                    (['--random-forest', '-r'],
                                     {'action': 'store_true',
                                      'dest': "do_random_forest",
                                      'help': "use a random forest",
                                      'default': False}),
                                    (['-n','--nrounds'],
                                     {'type': int,
                                      'help': "Run n iterations in stead of one",
                                      'dest': 'nrounds',
                                      'default': None}),
                                ],
                                try_predict),
                               ('tree',
                                "Generate a tree graph visualisation",
                                [
                                    (['percent_broken'], {'type': int}),
                                    (['percent_ok'], {'type': int}),
                                    (['-w','--writefile'],
                                     {'type': str,
                                      'default': '../Report/Graphs/disks_tree.pdf'}),
                                    (['-n','--nrounds'],
                                     {'type': int,
                                      'help': "Run n iterations in stead of one",
                                      'dest': 'nrounds',
                                      'default': None}),
                                ],
                                make_tree),
                               ('roc_graph',
                                "Generate a ROC graph",
                                [
                                    (['-w','--writefile'],
                                     {'type': str,
                                      'default': '../Report/Graphs/roc_graph_disks.tex'}),
                                ],
                                make_roc_graph),
                               ('kmeans_graph',
                                "Generate a K-means++ visualisation of data",
                                [
                                    (['-w','--writefile'],
                                     {'type': str,
                                      'default': '../Report/Graphs/kmeans-PCA-dataset.pdf'}),
                                    (['-u','--use-at-most'],
                                     {'type': int,
                                      'dest': 'use_at_most',
                                      'default': None}),
                                ],
                                make_kmeans_graph),

                               ])

    args = parser.parse_args()
    daiquiri.setup()
    common.set_log_level_from_args(args, log)

    log.info("Using dataset %s", args.use_dataset)

    if args.use_dataset == "cern_disks":
        train_data = common.disk_training_set()
        args.feature_labels = common.disk_training_set_feature_labels()
        args.feature_labels.remove('is_broken')
        args.roc_start_p = 1
        args.roc_broken_p = 41
    elif args.use_dataset == "zhu_disks":
        train_data = common.zhu_2013_training_set()
        args.feature_labels = common.ZHU_FEATURE_LABELS[2:]
        args.roc_start_p = 1
        args.roc_broken_p = 75
    elif args.use_dataset == "random":
        train_data = common.random_training_set(4000, 4)
        args.feature_labels = ["random%d" %i for i in range(1, train_data.shape[1])]
        args.roc_start_p = 1
        args.roc_broken_p = 75
    elif args.use_dataset == "cern_bad_blocks":
        train_data = common.bad_block_training_set()
        args.feature_labels = common.bad_block_training_set_feature_labels()
        args.feature_labels.remove('is_broken')
        args.roc_start_p = 1
        args.roc_broken_p = 41
    else:
        print("That dataset is not supported yet")
        exit(1)

    # Note -1 because the data set contains the target label column and
    # our feature labels doesn't.
    assert len(args.feature_labels) == (train_data.shape[1] - 1)

    log.info("Loaded dataset")

    if args.do_normalise:
        log.debug("Normalising data...")
        train_data = common.zhu_2013_normalise(train_data)
        log.debug("Done normalising data")

    ok, broken = common.split_disk_data(train_data)
    log.info("Split disk data")
    if not args.dont_shuffle:
        np.random.shuffle(ok)
        np.random.shuffle(broken)
    ok = common.remove_labels(ok)
    broken = common.remove_labels(broken)
    log.info("Finished pre-processing data")

    args.func(ok=ok, broken=broken, args=args)
