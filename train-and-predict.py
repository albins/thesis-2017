#!/usr/bin/env python3
import random
import math
import itertools
import logging
import sys
from statistics import median, stdev
import argparse

import common
from common import sample_matrix, tree_as_pdf, verify_training

import numpy as np
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, chi2
from sklearn.feature_selection import SelectKBest
from sklearn.externals import joblib
from scipy.stats import mode
import daiquiri
import palettable
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
            classifier=tree.DecisionTreeClassifier, nrounds=None,
            max_depth=None):

    if keep_broken == 0 and keep_nonbroken == 0:
        log.info("Using equal proportions broken/non-broken")
        # Use equal proportions
        half_broken_length = int(broken.shape[0]/2)

        training_broken = broken[:half_broken_length]
        witheld_broken = broken[half_broken_length:]

        training_ok = ok_disks[:half_broken_length]
        witheld_ok = ok_disks[half_broken_length:]
    else:
        training_broken, witheld_broken = sample_matrix(
            broken,
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
            if classifier is not svm.SVC:
                c = classifier(random_state=42, max_depth=max_depth)
            else:
                c = classifier(random_state=42)

            model = c.fit(training_set, labels)

            return verify_training(model, verification_set, expected_labels)
    else:
        results = []
        for i in range(0, nrounds):
            if classifier is not svm.SVC:
                c = classifier(random_state=i, max_depth=max_depth)
            else:
                c = classifier(random_state=i)
            model = c.fit(training_set, labels)
            results.append(verify_training(model, verification_set,
                                           expected_labels))
        return results


def find_best_training_proportion(broken, ok, nrounds, broken_start, ok_start, max_depth):
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
    best_tpr_combination = (best_tpr, best_far)
    best_far_combination = None
    keep_ok_p = range(max(min_ok_percent, ok_start), 76, 10)
    keep_broken_p = range(max(broken_start, min_broken_percent),
                          max_broken_percent + 1, 10)

    highest_tolerable_far = 0.1
    lowest_tolerable_tpr = 0.5

    for ok_p, broken_p in itertools.product(keep_ok_p, keep_broken_p):
        log.debug("Testing %d%% ok, %d%% broken data", ok_p, broken_p)
        tpr, far, _tree = predict_worst(broken, ok, keep_broken=broken_p/100,
                                        keep_nonbroken=ok_p/100, nrounds=nrounds,
                                        max_depth=max_depth)
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
                                                   ok_start=args.ok_start,
                                                   max_depth=args.max_depth)
    tpr, far, _tree = predict(broken, ok, keep_broken=broken_p/100,
                              keep_nonbroken=ok_p/100,
                              max_depth=args.max_depth)
    print("Result: TPR: {}, FAR {} at mix {}% from failed set, {}% from OK set"
          .format(tpr, far, broken_p, ok_p))


def try_predict(ok, broken, args):
    broken_p = args.percent_broken
    ok_p = args.percent_ok
    if args.classifier == "tree":
        log.debug("Using normal classification tree")
        if not args.nrounds >= 2:
            tpr, far, tree = predict(broken, ok, keep_broken=broken_p/100,
                                     keep_nonbroken=ok_p/100,
                                     max_depth=args.max_depth)
        else:
            results = predict(broken, ok, keep_broken=broken_p/100,
                              keep_nonbroken=ok_p/100, nrounds=args.nrounds,
                              max_depth=args.max_depth)

    elif args.classifier == "random_forest":
        log.debug("Using random forest")
        tpr, far, tree = predict(broken, ok, keep_broken=broken_p/100,
                                  keep_nonbroken=ok_p/100,
                                  classifier=RandomForestClassifier)
    elif args.classifier == "svm":
        log.debug("Using SVM!")
        results = predict(broken, ok, keep_broken=broken_p/100,
                          keep_nonbroken=ok_p/100, nrounds=args.nrounds,
                          max_depth=args.max_depth,
                          classifier=svm.SVC)

    if args.nrounds >= 2:
        tprs = [x[0] for x in results]
        fars = [x[1] for x in results]

        # Use the best result (sorted first by TPR high to low, then FAR
        # low to high
        results.sort(key=lambda tup: tup[1], reverse=False)
        results.sort(reverse=True, key=lambda tup: tup[0])
        _, _, tree = results[0]

        def int_mode(xs):
            return mode([int(x * 100)/100 for x in xs]).mode[0]

        print("Result: TPR: [{}, {}] (stdev={}, mode={}, median={}), FAR [{}, {}] (stedv={}, mode={}, median={}) at mix {}% from failed set, {}% from OK set"
              .format(min(tprs), max(tprs), stdev(tprs), int_mode(tprs), median(tprs),
                      min(fars),  max(fars), stdev(fars), int_mode(fars), median(fars),
                      broken_p, ok_p))
    else:
        print("Result: TPR: {}, FAR {} at mix {}% from failed set, {}% from OK set"
              .format(tpr, far, broken_p, ok_p))

    if args.dump_model_file:
        log.info("Dumping model to %s", args.dump_model_file)

        from sklearn.externals import joblib
        joblib.dump(tree, args.dump_model_file)


def make_tree(ok, broken, args):
    broken_p = args.percent_broken
    ok_p = args.percent_ok
    if args.from_model:
        t = joblib.load(args.from_model)
        tpr, far = -1, -1

    elif args.nrounds:
        tpr, far, t = predict_best(broken, ok, keep_broken=broken_p/100,
                                   keep_nonbroken=ok_p/100,
                                   nrounds=args.nrounds,
                                   max_depth=args.max_depth)
    else:
        tpr, far, t = predict(broken, ok, keep_broken=broken_p/100,
                              keep_nonbroken=ok_p/100,
                              max_depth=args.max_depth)

    target_file = args.writefile

    labels = args.feature_labels
    tree_as_pdf(t, target_file, feature_names=labels,
                class_names=["Failed", "OK"])
    log.info("Rendered tree with TPR: {}, FAR {} at mix {}% from failed set, {}% from OK set"
             .format(tpr, far, broken_p, ok_p))


def make_roc_graph(ok, broken, args):
    def predictor(*predictor_args, **kwargs):
        return predict_worst(*predictor_args, nrounds=args.nrounds, **kwargs)

    xs, ys, vals = common.make_roc_measurements(
        broken,
        ok,
        predict=predictor,
        start_percentage=args.roc_start_p,
        broken_percent=args.roc_broken_p,
        step_size=args.roc_step_size,
        stop_percentage=args.roc_stop_p)

    common.render_pyplot_scatter_plot(xs,
                                      ys,
                                      vals,
                                      file_name=args.writefile,
                                      x_label="False-positive rate",
                                      y_label="True-positive rate")


def reduce_features(dataset, labels, k):
    """
    Reduce to k features, returning the column indices to keep.

    Use with m[:,return value]
    """

    selector = SelectKBest(chi2, k=k)

    # chi2: go through each column, add -1 * smallest number, if
    # negative, as it can't handle negative numbers.
    def normalise_column(col):
        x_min = col.min()
        if x_min < 0:
            col += (-1 * x_min)
        return col

    all_data = np.apply_along_axis(normalise_column, 0, dataset)
    _ = selector.fit_transform(all_data, labels)
    return selector.get_support(indices=True)



def try_feature_reduction(ok, broken, args):
    all_data = np.append(ok, broken, axis=0)

    labels = list((ok.shape[0] * [common.PREDICT_OK])
                  + (broken.shape[0] * [common.PREDICT_FAIL]))

    keep_columns = reduce_features(all_data, labels, args.k)

    for x in keep_columns:
        print(args.feature_labels[x])


def make_kmeans_graph(ok, broken, args):
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import scale
    from sklearn.decomposition import PCA

    if args.use_at_most == -1:
        log.info("Using equal proportions of broken/non-broken data!")
        half_broken_length = int(broken.shape[0]/2)
        ok_subset = ok[:half_broken_length]
        all_data = np.append(ok_subset, broken, axis=0)
    else:
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
               cmap=palettable.wesanderson.Moonrise1_5.mpl_colormap,
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
    parser.add_argument("--reduce-features", "-f",
                        dest="reduce_features",
                        type=int,
                        help="reduce to this number of features",
                        default=None)

    common.add_subcommands(parent=parser,
                           descriptions=[
                               ('best_settings',
                                "Determine the best settings",
                                [
                                    (['-n', '--nrounds'],
                                     {'type': int,
                                      'help': "Run n iterations in stead of one",
                                      'dest': 'nrounds',
                                      'default': 1}),
                                    (['-b', '--broken-start-percent'],
                                     {'type': int,
                                      'help': "Start trying at this percentage",
                                      'dest': 'broken_start',
                                      'default': 1}),
                                    (['-o', '--ok-start-percent'],
                                     {'type': int,
                                      'help': "Start trying at this percentage",
                                      'dest': 'ok_start',
                                      'default': 1}),
                                    (['-d','--max-depth'],
                                     {'type': int,
                                      'help': "Force tree to be of this maximum depth",
                                      'dest': 'max_depth',
                                      'default': None}),
                                ],
                                best_settings),
                               ('predict',
                                "Try predicting failures from the set",
                                [
                                    (['percent_broken'], {'type': int}),
                                    (['percent_ok'], {'type': int}),
                                    (['--use-classifier', '-c'],
                                     {'type': str,
                                      'dest': "classifier",
                                      'help': "use this classifier",
                                      'default': "tree",
                                      'choices': ["svm", "tree", "random_forest"]}),
                                    (['-n', '--nrounds'],
                                     {'type': int,
                                      'help': "Run n iterations in stead of one",
                                      'dest': 'nrounds',
                                      'default': 1}),
                                    (['-m', '--dump-model-file'],
                                     {'type': str,
                                      'help': "Dump the model to this file afterwards",
                                      'dest': 'dump_model_file',
                                      'default': None}),
                                    (['-d','--max-depth'],
                                     {'type': int,
                                      'help': "Force tree to be of this maximum depth",
                                      'dest': 'max_depth',
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
                                    (['-d','--max-depth'],
                                     {'type': int,
                                      'help': "Force tree to be of this maximum depth",
                                      'dest': 'max_depth',
                                      'default': None}),
                                    (['-m','--from-model'],
                                     {'type': str,
                                      'help': "Use this model file",
                                      'dest': 'from_model',
                                      'default': None}),
                                ],
                                make_tree),
                               ('roc_graph',
                                "Generate a ROC graph",
                                [
                                    (['-w','--writefile'],
                                     {'type': str,
                                      'default': '../Report/Graphs/roc_graph_disks.pdf'}),
                                    (['-n','--nrounds'],
                                     {'type': int,
                                      'help': "Run n iterations in stead of one",
                                      'dest': 'nrounds',
                                      'default': 3}),
                                    (['-s','--start-percent'],
                                     {'type': int,
                                      'help': "Start at N percent",
                                      'dest': 'roc_start_p',
                                      'default': 1}),
                                    (['-t','--stop-percent'],
                                     {'type': int,
                                      'help': "Start at N percent",
                                      'dest': 'roc_stop_p',
                                      'default': 75}),
                                    (['-b', '--broken-percent'],
                                     {'type': int,
                                      'help': "Use N percent of the records in the broken set",
                                      'dest': 'roc_broken_p',
                                      'default': 75}),
                                    (['-e', '--step-size'],
                                     {'type': int,
                                      'help': "Step N percent at a time",
                                      'dest': 'roc_step_size',
                                      'default': 5}),
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

                               ('feature_reductions',
                                "Try out various feature reduction techniques",
                                [
                                    (['-k','--target-feature-count'],
                                     {'type': int,
                                      'dest': 'k',
                                      'default': 2}),
                                ],
                                try_feature_reduction),
                               ])

    args = parser.parse_args()
    daiquiri.setup()
    common.set_log_level_from_args(args, log)

    log.info("Using dataset %s", args.use_dataset)

    if args.use_dataset == "cern_disks":
        train_data = common.disk_training_set()
        args.feature_labels = common.disk_training_set_feature_labels()
        args.feature_labels.remove('is_broken')
    elif args.use_dataset == "zhu_disks":
        train_data = common.zhu_2013_training_set()
        args.feature_labels = common.ZHU_FEATURE_LABELS[2:]
    elif args.use_dataset == "random":
        train_data = common.random_training_set(4000, 4)
        args.feature_labels = ["random%d" % i for i in
                               range(1, train_data.shape[1])]
    elif args.use_dataset == "cern_bad_blocks":
        train_data = common.bad_block_training_set()
        args.feature_labels = common.bad_block_training_set_feature_labels()
        args.feature_labels.remove('is_broken')
    else:
        print("That dataset is not supported yet")
        exit(1)

    # Note -1 because the data set contains the target label column and
    # our feature labels doesn't.
    assert len(args.feature_labels) == (train_data.shape[1] - 1)

    log.info("Loaded dataset")

    if args.do_normalise:
        log.debug("Normalising data...")
        with common.timed(task_name="normalisation"):
            train_data = common.zhu_2013_normalise_fast(train_data)
        log.debug("Done normalising data")

    ok, broken = common.split_disk_data(train_data)
    log.info("Split disk data %d (%d%%) broken entries, %d ok (%d%%)",
             broken.shape[0],
             broken.shape[0]/train_data.shape[0]*100,
             ok.shape[0],
             ok.shape[0]/train_data.shape[0]*100)

    if not args.dont_shuffle:
        np.random.shuffle(ok)
        np.random.shuffle(broken)
        log.info("Shuffled dataset")

    assert set(ok[:,0]) == set([0]), "Non-OK labeled drives in OK set"
    assert set(broken[:,0]) == set([1]), "Only broken drives in broken set"

    ok = common.remove_labels(ok)
    broken = common.remove_labels(broken)

    if args.reduce_features and args.reduce_features <= broken.shape[1]:
        log.info("Applying Chi-squared feature reduction to %d features",
                 args.reduce_features)
        labels = list((ok.shape[0] * [common.PREDICT_OK])
                      + (broken.shape[0] * [common.PREDICT_FAIL]))
        feature_indices = reduce_features(np.append(ok, broken, axis=0), labels,
                                          args.reduce_features)
        log.debug("Got the following feature indices: %s",
                  ", ".join([str(x) for x in feature_indices]))
        args.feature_labels = list(np.array(args.feature_labels)[feature_indices])
        log.info("Reduced to using the following features %s",
                 ", ".join(args.feature_labels))
        ok = ok[:, feature_indices]
        broken = broken[:, feature_indices]

    log.info("Finished pre-processing data")
    log.debug("Using features %s", ", ".join(args.feature_labels))

    args.func(ok=ok, broken=broken, args=args)
