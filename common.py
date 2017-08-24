import argparse
from contextlib import contextmanager
import time
import logging
import pandas
import math
import random
import os
import urllib3

import numpy as np
from elasticsearch import Elasticsearch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import palettable

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

log = logging.getLogger(__name__)

ELASTIC_ADDRESS = "es-itdb.cern.ch:9203"
ZHU_FEATURE_LABELS = [
    "Disk ID",
    "Is broken",
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
    "Reallocated Sectors Count RAW",
    "Current Pending Sector Count RAW"
]

PREDICT_FAIL = -1
PREDICT_OK = 1
NUM_TO_PREDICTION = {PREDICT_OK: "OK", PREDICT_FAIL: "FAIL"}

def add_elasticsearch_options(parser, default_address):
    parser.add_argument('--timeout_s', '-t', nargs='?', dest='timeout',
                        default=10,
                        type=int, help="Database operation timeout in seconds")
    parser.add_argument('--es-node', '-s', nargs='?', type=str,
                        dest='es_nodes',
                        help="Elasticsearch node to connect to",
                        default=default_address)


def es_conn_from_args(args):
    log.debug("Setting up ES connection using %s", args)
    es = Elasticsearch([args.es_nodes],
                       timeout=args.timeout,
                       use_ssl=True,
                       verify_certs=False,
                       http_auth=(os.environ['ES_USERNAME'],
                                  os.environ['ES_PASSWORD']),
                       #retry_on_timeout=True,
    )
    return es


def set_log_level_from_args(args, logger):
    log_level = (max(3 - args.verbose_count, 0) * 10)
    logger.setLevel(log_level)


def setup_verbosity_flags(parser):
    parser.add_argument('--verbose', '-v', action='count', dest='verbose_count',
                        default=0)

def make_es_base_parser():
    parser = argparse.ArgumentParser(description="")
    setup_verbosity_flags(parser)
    add_elasticsearch_options(parser, ELASTIC_ADDRESS)
    return parser


def add_subcommands(parent, descriptions):
    subparser_master = parent.add_subparsers(title='operations',
                                             help="valid operations",
                                             dest='cmd')
    subparser_master.required = True
    added_parsers = [subparser_master]
    for description in descriptions:
        name, hlp, args, func = description
        parser = subparser_master.add_parser(name, help=hlp)
        parser.set_defaults(func=func)
        for arg_set in args:
            fun_args, kwd_args = arg_set
            parser.add_argument(*fun_args, **kwd_args)
        added_parsers.append(parser)
    return added_parsers



def format_counter(counter):
    return ", ".join(["{} ({})".format(k, c) for k, c in
                      sorted(counter.items(),
                             key=lambda tpl: tpl[1],
                             reverse=True)])


@contextmanager
def timed(task_name, time_record=None, printer=log.debug):
    start_time = time.clock()
    yield
    end_time = time.clock()
    if time_record is None:
        time_record = []
    printer("Task {} ran in {:06.4f}s"
            .format(task_name, end_time - start_time))
    time_record.append(end_time - start_time)


def zhu_2013_normalise(dataset, start_column=1):
    """
    Normalise a dataset according to the formula in
    Zhu, 2013.
    """
    max_features = dataset.max(0)
    min_features = dataset.min(0)

    def x_normal(x, column):
        min_x = min_features[column]
        max_x = max_features[column]
        top = x - min_x
        assert top != float('inf')
        bottom = max_x - min_x
        if bottom == 0:
            #log.warning("Column %d has only value %d!",
            #            column, bottom)
            return x
        return 2 * (top/bottom - 1)

    for index, x in np.ndenumerate(dataset[:,start_column:]):
        row_idx, column_idx = index
        column_idx += start_column
        normalised = x_normal(x, column_idx)
        dataset[row_idx][column_idx] = normalised

    return dataset


def zhu_2013_normalise_fast(dataset, start_column=1):

    column_idx = start_column - 1
    set_to_normalise = dataset[:, start_column:]
    labels_column = dataset[:, column_idx]

    def normalise_column(col):
        x_min = col.min()
        x_max = col.max()
        x_diff = x_max - x_min

        if x_diff == 0:
            return col

        col = col - x_min
        col = col * (2/x_diff)
        col -= 1
        return col

    return np.insert(np.apply_along_axis(normalise_column,
                                         0, set_to_normalise),
                     0,
                     labels_column,
                     axis=1)


def read_csv_w_labels(filename):
    """
    Returns an array of values and a list of labels
    """
    data = pandas.read_csv(filename,
                           delimiter=";",
                           quotechar="|",
                           header="infer")

    data = filter_unchanged_disk_data(data,
                                      keep_columns=["is_broken",
                                                    "disk_type"])

    return data

def filter_unchanged_disk_data(data, keep_columns):
    for x in data:
        if "delta" in x or x in keep_columns or not "%s_delta" % x in data:
            continue
        if set(data["%s_delta" % x]) == set([0]):
            log.warning("Dropping column %s due to no deltas", x)
            del data[x]
            del data["%s_delta" % x]
    return data

def disk_training_set():
    data = read_csv_w_labels("../Data/training_data.csv")
    matrix_data = data.as_matrix()
    matrix_data = matrix_data[matrix_data[:,0].argsort()]
    return matrix_data


def disk_training_set_feature_labels():
    data = read_csv_w_labels("../Data/training_data.csv")
    return list(data.keys())


def bad_block_training_set_feature_labels():
    data = read_csv_w_labels("../Data/training_data_bad_blocks.csv")
    return list(data.keys())

def bad_block_training_set():
    data = read_csv_w_labels("../Data/training_data_bad_blocks.csv")
    matrix_data = data.as_matrix()
    matrix_data = matrix_data[matrix_data[:,0].argsort()]
    return matrix_data

def split_disk_data(disk_data):
    """
    Presumes that disk_data is sorted with non-broken to broken disks
    and not normalised.
    """

    last_nonbroken_row = np.where(disk_data[:, 0] == 1.)[0][0]
    log.info("Splitting matrix of size %dx%d at %d",
             *disk_data.shape, last_nonbroken_row)

    nonbroken = disk_data[:last_nonbroken_row]
    broken = disk_data[last_nonbroken_row:]

    log.info("Split sizes are: %d, %d respectively",
             nonbroken.shape[0],
             broken.shape[0])
    return nonbroken, broken


def remove_labels(disk_data):
    return disk_data[:,1:]


def sample_matrix(m, keep_portion=0.5):
    height, _w = m.shape
    closest_sublen = math.ceil(height * keep_portion)
    selected = m[:closest_sublen]
    deselected = m[closest_sublen:]
    return selected, deselected


def read_zhu_2013_data(delete_serials=True):
    data = pandas.read_csv("../Data/Disk_SMART_dataset.csv",
                           delimiter=",",
                           header=None,
                           names=ZHU_FEATURE_LABELS)

    # Remember, -1 = broken, 1 is ok, so broken disks comes first
    data.sort_values(['Disk ID', 'Is broken'],
                     ascending=[True, True],
                     inplace=True)

    # Make sure we only have one of every disk (by ID), preferring
    # broken disks
    data.drop_duplicates(subset='Disk ID', keep='first', inplace=True)

    if delete_serials:
        del data['Disk ID']

    return data


def zhu_2013_training_set():
    data = read_zhu_2013_data(delete_serials=True)
    matrix_data = data.as_matrix()
    # Zhu, 2013 uses -1 to demarcate broken, 1 for OK. We need to change
    # that to 0 for OK, 1 for broken (our convention)
    # abs() is there because otherwise we get -0, a very scary
    # proposition.
    matrix_data[:, 0] = abs((matrix_data[:, 0] - 1)/-2)

    matrix_data = matrix_data[matrix_data[:, 0].argsort()]
    return matrix_data


def random_training_set(count, features):
    random_array = np.random.random(size=(count, features))
    broken_or_ok_column = np.random.randint(0, 2, size=(count, 1))
    training_data = np.append(broken_or_ok_column, random_array, 1)
    training_data = training_data[training_data[:,0].argsort()]
    return training_data


def unique_values_in_dataset(data):
    for x in data:
        print("{}: {}".format(x, ", ".join([str(x)
                                            for x in sorted(set(data[x]))])))


def run_subcommand(args, *rest_args, **kwargs):
    if hasattr(args, "func"):
        args.func(args=args, *rest_args, **kwargs)
    else:
        print("You need to supply a subcommand!")


def calculate_tpr_far(true_positives, true_negatives,
                      false_positives, false_negatives):
    if true_positives + false_negatives == 0:
        tpr = 0
    else:
        tpr = true_positives/(true_positives + false_negatives)
    if true_negatives + false_positives == 0:
        far = 1
    else:
        far = false_positives / (true_negatives + false_positives)

    return tpr, far


def random_cycle_list(lst):
    split_at = random.randint(0, len(lst)-2)
    fst, snd = lst[:split_at], lst[split_at:]
    return [*snd, *fst]


def render_pyplot_bar_chart(data_pairs, x_label, y_label, file_name,
                            label_rotation=0, font_size='small',
                            show_every_nth_label=1):
    ticks, xs = zip(*data_pairs)

    x = np.arange(len(ticks))

    def autolabel(rects, ax):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.text(rect.get_x() + rect.get_width()/2., 1.005*height,
                        '%d' % int(height),
                        ha='center', va='bottom',
                        fontsize='xx-small')

    def hide_tick_labels(ax):
        for i, label in enumerate(ax.xaxis.get_ticklabels()):
            if i % show_every_nth_label == 0:
                continue
            else:
                label.set_visible(False)

    colours = random_cycle_list(palettable.wesanderson.Moonrise1_5.mpl_colors)

    with PdfPages(file_name) as pp:
        fig, ax = plt.subplots()
        rects = plt.bar(x,
                        xs,
                        color=colours)

        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        ax.set_xticks(x)
        ax.set_xticklabels(ticks, rotation=label_rotation,
                           fontsize=font_size)

        autolabel(rects, ax)
        hide_tick_labels(ax)

        plt.tight_layout()

        pp.savefig()


def render_pyplot_scatter_plot(xs, ys, data_labels, file_name,
                               x_label="", y_label=""):
    colours = random_cycle_list(palettable.wesanderson.Moonrise1_5.mpl_colors)

    with PdfPages(file_name) as pp:
        fig, ax = plt.subplots()
        plt.scatter(x=xs, y=ys, c=colours)

        for i, txt in enumerate(data_labels):
            ax.annotate(txt, (xs[i], ys[i]), fontsize='xx-small')

        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        plt.tight_layout()
        pp.savefig()


def tree_as_pdf(t, target_file, feature_names, class_names):
    import pydotplus
    from sklearn import tree

    dot_data = tree.export_graphviz(t, out_file=None,
                                    feature_names=feature_names,
                                    class_names=class_names,
                                    rounded=True,
                                    filled=True,
                                    special_characters=True,
                                    proportion=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf(target_file)


def make_roc_measurements(broken,
                          ok,
                          predict,
                          start_percentage=1,
                          stop_percentage=75,
                          step_size=5,
                          broken_percent=75):
    xs_and_ys = []

    for n in range(start_percentage, stop_percentage + 1, step_size):
        log.debug("Calculating ROC values for %d%%", n)
        xs_and_ys.append(calculate_roc_point(n, broken, ok,
                                             broken_percent=broken_percent,
                                             predict=predict))

    xs_and_ys.append(calculate_roc_point(0, broken, ok,
                                         broken_percent=0,
                                         predict=predict))


    ys, xs, vals = zip(*xs_and_ys)
    return xs, ys, vals


def calculate_roc_point(n, broken, ok, predict, broken_percent):
    percentage = 0.01 * n
    tpr, far, _tree = predict(broken, ok, keep_broken=broken_percent/100,
                              keep_nonbroken=percentage)
    return tpr, far, n


def verify_training(clf, verification_set, expected_labels):
    false_positives = 0
    false_negatives = 0
    true_positives = 0
    true_negatives = 0

    with timed(task_name="verification"):
        for prediction, expected in zip(clf.predict(verification_set),
                                        expected_labels):
            if prediction == expected:
                if expected == PREDICT_FAIL:
                    true_positives += 1
                else:
                    true_negatives += 1
            else:
                if expected == PREDICT_FAIL:
                    false_negatives += 1
                else:
                    false_positives += 1

    tpr, far = calculate_tpr_far(true_positives, true_negatives,
                                 false_positives, false_negatives)
    log.debug("False positives: %d, false negatives: %d, true positives: %d, true negatives: %d. TPR: %f, FAR: %f",
              false_positives, false_negatives,
              true_positives, true_negatives,
              tpr, far)
    return tpr, far, clf


def fmt_cell(cell, as_latex=False, tpr_thresh=None, far_thresh=None):
    if cell is None:
        if as_latex:
            return "\\infty"
        else:
            return "Infinity"
    if isinstance(cell, int):
        return "{}".format(cell)

    tpr_thresh = tpr_thresh if tpr_thresh else float("inf")
    far_thresh = far_thresh if far_thresh else float("-inf")

    def mark_str(num_fmt_str):
        if as_latex:
            return "\\bm{{" + num_fmt_str + "}}"
        else:
            return "*{}*".format(num_fmt_str)

    def format_number(f, three_decimals=False):
        if three_decimals:
            s = "{:.3f}".format(f)
        else:
            s = "{:.2f}".format(f)

        if s[0] == '0':
            return s[1:]
        else:
            return s

    tpr, far = cell
    formatted_tpr = format_number(tpr)
    formatted_far = format_number(far, three_decimals=True)
    if tpr >= tpr_thresh:
        tpr_str = mark_str(formatted_tpr)
    else:
        tpr_str = formatted_tpr

    if far <= far_thresh:
        far_str = mark_str(formatted_far)
    else:
        far_str = formatted_far

    return "{}/{}".format(tpr_str, far_str)


def human_readable_experiment_table(double_dict, mark_far_below, mark_tpr_above):
    """
    Turn a #features => max depth => (tpr, far) dict to a human-readable
    table!
    """

    first_row = [str(k) for k in list(double_dict.values())[0].keys()]
    data_values = [[k, *list(v.values())] for k, v in double_dict.items()]

    strings = ["\t" + "\t\t".join(first_row)]

    for row in data_values:
        strings.append("\t".join([fmt_cell(x, as_latex=False,
                                           tpr_thresh=mark_tpr_above,
                                           far_thresh=mark_far_below)
                                  for x in row]))
    return "\n".join(strings)


def latex_experiment_table(double_dict, mark_far_below, mark_tpr_above, caption, label):
    table_template = """
    \\begin{{table}}
    \\caption{{{}}} \\label{{{}}}
    $\\begin{{array}} {{ *{{{}}}{{c}} }}
    \\toprule
    \t & {} \\\\
    \t \\midrule
    \t {} \\\\
    \\bottomrule
    \\end{{array}}$
    \\end{{table}}
    """

    def make_table_row(values):
        return " & ".join([fmt_cell(x, as_latex=True,
                                    tpr_thresh=mark_tpr_above,
                                    far_thresh=mark_far_below)
                           for x in values])

    first_row = [k for k in list(double_dict.values())[0].keys()]
    data_values = [[k, *list(v.values())] for k, v in double_dict.items()]

    rows = [make_table_row(row) for row in [first_row, *data_values]]
    return table_template.format(caption,
                                 label,
                                 len(first_row) + 1,
                                 rows[0],
                                 " \\\\ \n\t".join(rows[1:]))
