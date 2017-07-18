#!/usr/bin/env python3

import csv
import re
import statistics
from csv import DictReader, DictWriter
from parse_emails import format_counter
from collections import Counter, defaultdict, OrderedDict
from analysis import SMART_MYSTERY_FIELDS_KEEP, SENSE_FIELDS_KEEP

KEEP_RGEXES = [re.compile(".*_smart_mystery_%d" % f) for
               f in SMART_MYSTERY_FIELDS_KEEP]

KEEP_SENSE_RGEXES = [re.compile(".*_sense_%d" % f) for f in SENSE_FIELDS_KEEP]

def summarise_dicts(ds):
    """
    Summarise the number of distinct values of a dict.
    """
    uniques = defaultdict(set)
    for d in ds:
        for key, value in d.items():
            uniques[key].add(value)
    return uniques

def read_data():
    with open("../Data/cleaned_training_data.csv") as f:
        reader = DictReader(f,
                            delimiter=';',
                            quotechar='|',
                            quoting=csv.QUOTE_MINIMAL)
        rows = []
        for row in reader:
            new_row = OrderedDict()
            for key, v in row.items():
                if "smart_mystery" in key and \
                   not any([rx.match(key) for rx in KEEP_RGEXES]):
                    print("Removing key {}".format(key))
                elif "_sense" in key and \
                     not any([rx.match(key) for rx in KEEP_SENSE_RGEXES]):
                    print("Removing key {}".format(key))
                else:
                    new_row[key] = float(v)

            rows.append(new_row)
    return rows

# with open("../Data/cleaned_training_data.csv", 'w') as f:
#     data = read_data()
#     writer = DictWriter(f,
#                         fieldnames=data[0].keys(),
#                         delimiter=';',
#                         quotechar='|',
#                         quoting=csv.QUOTE_MINIMAL)
#     writer.writeheader()
#     for row in data:
#         writer.writerow(row)


#s = summarise_dicts(read_data())
#for k, v in s.items():
#    if len(v) < 4:
#        print("{}: {}".format(k, ", ".join([str(n) for n in v])))
#    else:
#        print("{}: c: {}, min: {}, max: {}, stdev. {}"
#              .format(k, len(v), min(v), max(v), statistics.stdev(v)))
