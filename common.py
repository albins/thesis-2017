import argparse
from contextlib import contextmanager
import time
import logging
import pandas
import math

import numpy as np
from elasticsearch import Elasticsearch

log = logging.getLogger(__name__)

ELASTIC_ADDRESS = "localhost:9200"
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
    parser.add_argument('--es-sniff', '-i',
                        help=("Enable ElasticSearch sniffing "
                              " before connecting and on connection failures"),
                        action='store_true',
                        dest="do_es_sniff")


def es_conn_from_args(args):
    es = Elasticsearch([args.es_nodes], timeout=args.timeout,
                       sniff_on_start=args.do_es_sniff,
                       sniff_on_connection_fail=args.do_es_sniff,
                       sniffer_timeout=args.timeout,
                       retry_on_timeout=True,
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
                                             help="valid operations")
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
def timed(task_name, time_record=[], printer=log.debug):
    start_time = time.clock()
    yield
    end_time = time.clock()
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
        assert bottom != 0
        return 2 * (top/bottom - 1)

    for index, x in np.ndenumerate(dataset[:,start_column:]):
        row_idx, column_idx = index
        column_idx += start_column
        normalised = x_normal(x, column_idx)
        dataset[row_idx][column_idx] = normalised

    return dataset


def read_csv_w_labels(filename):
    """
    Returns an array of values and a list of labels
    """
    data = pandas.read_csv(filename,
                           delimiter=";",
                           quotechar="|",
                           header="infer")
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

    last_nonbroken_row = np.where(disk_data[:, 0]== 1.)[0][0]
    nonbroken = disk_data[:last_nonbroken_row]
    broken = disk_data[last_nonbroken_row:]
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
    if delete_serials:
        del data['Disk ID']

    return data


def zhu_2013_training_set():
    data = read_zhu_2013_data(delete_serials=True)
    matrix_data = data.as_matrix()
    matrix_data[matrix_data[:,0].argsort()]
    return matrix_data


def random_training_set(count, features):
    random_array = np.random.random(size=(count, features))
    broken_or_ok_column = np.random.randint(0, 2, size=(count, 1))
    training_data = np.append(broken_or_ok_column, random_array, 1)
    training_data[training_data[:,0].argsort()]
    return training_data


def unique_values_in_dataset(data):
    for x in data:
        print("{}: {}".format(x, ", ".join([str(x)
                                            for x in sorted(set(data[x]))])))
