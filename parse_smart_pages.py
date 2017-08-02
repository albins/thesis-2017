#!/usr/bin/env python3
import common
from common import format_counter, timed

import sys
import regex
from collections import defaultdict, Counter
import gzip as gz
import logging
import os
import datetime
import csv
import statistics
import time
from datetime import timedelta

import dateparser
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan, parallel_bulk, streaming_bulk
import pytz
import daiquiri

log = daiquiri.getLogger()
elastic_logger = logging.getLogger('elasticsearch')
elastic_logger.setLevel(logging.WARNING)

disk_re = regex.compile("^Disk\s+.*?\.(?P<disk>\d+.\d+):?")
row_re = regex.compile(r".*:\s+(0x([0-9a-f]{2})+\s*)+")
node_re = regex.compile(r"^Node: db(?P<node_name>.+\d+)\s*")
line_re = regex.compile("^\s+(.*?)h\s+(.*?)h\s+(\d+)\s+(\d+)\s+(.*?)h\s+.*")
smart_pages_re = regex.compile(r".*Disk SMART Pages.*")
underline_re = regex.compile('^-+')
sense_error_heading_re = regex.compile('Disk LOG Sense Error.*')
smart_pages_heading_re = regex.compile(r".*Disk SMART Pages.*")
file_re = regex.compile("db(?<node_name>.*?)"
                        "-cluster-mgmt\.(?<timestamp>.*?)\.data\.gz$")

BEGINNING_OF_TIME = datetime.datetime.fromtimestamp(0, tz=pytz.utc)
FILE_CODEC = 'ascii'
DATA_FILE_FIELDS = ["timestamp", "cluster", "disk", "serial",
                    "state", "average_io", "max_io", "retry_count",
                    "timeout_count",
                    *["sense_data{n}".format(n=n) for n in range(1, 6)],
                    "sense_data9",
                    "sense_dataB",
                    "smart_data",
                    "smart_mystery"]
CSV_DELIMITER = ";"

# db-51167.cern.ch:9200
ELASTIC_ADDRESS = "db-51167.cern.ch:9200"
ES_INDEX_BASE = 'netapp-lowlevel'
ES_INDEX = 'netapp-lowlevel-*'
LOW_LEVEL_DATA_Q = '*'


def es_count_disks_by_cluster(es, index, q):
    """
    Return a dictionary of cluster => count for a given query
    """

    body = {
        "size": 0,
        "aggs": {
            "group_by_cluster": {
                "terms": {"size": 0,
                          "field": "cluster_name"
                },
                "aggs": {
                    "count_distinct_disks": {
                        "cardinality": {
                            "field": "disk_location",
                            "precision_threshold": 4000,
                        }
                    }
                }
            }
        }
    }
    res = es.search(index=index, body=body, q=q)
    cluster_counts = dict()
    for bucket in get_buckets(res, aggregation_name="group_by_cluster"):
        bucket_data = bucket['count_distinct_disks']
        count = bucket_data['value']
        cluster = bucket['key']
        cluster_counts[cluster] = count
    return cluster_counts


def count_disks(es):
    return sum(es_count_disks_by_cluster(es, ES_INDEX, q="*").values())


def get_buckets(res, aggregation_name):
    """
    Unpack a list of buckets from an elasticsearch result.
    """

    try:
        return res['aggregations'][aggregation_name]['buckets']
    except KeyError:
        return []


def end_pad(lst, target_length, pad_value):
    diff_len = target_length - len(lst)
    assert diff_len >= 0
    return [*lst, *([pad_value] * diff_len)]


def read_table_row(row):
    match = row_re.match(row)
    if not match:
        return None
    else:
        numbers = [int(x, 16) for x in match.captures(2)]
        if len(numbers) < 10:
            pad_len = 10 - len(numbers)
            pad_list = pad_len * [0]
            numbers += pad_list
        return numbers


def seek_to(lines, re, offset_hint=0, max_search=None):
    offset = offset_hint
    for offset, line in enumerate(lines[offset:], start=offset_hint):
        match = re.match(line)
        if match:
            log.debug("Found a match for {} at offset {}".format(re, offset))
            return offset, match
        elif max_search and (offset - offset_hint) >= max_search:
            log.debug("Gave up after {} lines".format(max_search))
            break

    log.debug("Could not find a match for {} at offset {}".format(re, offset))
    raise IndexError("Could not find any match for {} after line {}"
                     .format(re, offset_hint))


def read_smart_pages(lines, offset_hint=0):
    disks = {}
    current_table = []

    # Seek to "Disk SMART Pages"
    # skip --- line
    offset, _match = seek_to(lines, re=smart_pages_re, offset_hint=offset_hint, max_search=1)
    offset, disk_match = seek_to(lines, re=disk_re, offset_hint=offset, max_search=5)

    disk = disk_match.group('disk')
    offset += 1

    for i, line in enumerate(lines[offset:], start=offset):
        if not line.strip():
            if not lines[i+1].strip():
                # Two successive blank lines: end of data
                break

        maybe_disk = disk_re.match(line)

        if maybe_disk:
            disks[disk] = current_table  # flush table
            current_table = []

            # Start new accumulation
            disk = maybe_disk.group('disk')
            continue

        row = read_table_row(line)
        if row:
            current_table.append(row)
    disks[disk] = current_table
    return offset, disks


def read_table(lines, start_re, re_offset, reducer, accumulator={},
               offset_hint=0, is_at_end=None):
    """
    - reducer: takes accumulator and the columns, returns the new value
      of the accumulator. May raise a ValueError to indicate invalid
      input (which will be ignored).
    - start_re: regular expression to seek to
    - re_offset: how many lines to skip before parsing
    - is_at_end(i, lines): determines if we should stop here, at
      lines[i]. Default is two consecutive empty lines.

    Returns a tuple of last location, accumulator.
    """
    if not is_at_end:
        is_at_end = lambda i, lines: not lines[i].strip()\
                    and not lines[i+1].strip()

    offset, _match = seek_to(lines, re=start_re, offset_hint=offset_hint)
    offset += re_offset
    for i, line in enumerate(lines[offset:], start=offset):
        if is_at_end(i, lines):
            break
        # Filter out anything unreasonable
        columns = [c.strip() for c in regex.split("\s+", line) if c.strip()]
        try:
            accumulator = reducer(accumulator, columns)
        except ValueError as e:
            log.warning("Error reading table columns on line {} ({}): {}"
                        .format(i, " ".join(columns), e))
            continue
    return i, accumulator


def extract_id_column(columns):
    id_s, rest = columns[0].split(".")
    if rest.strip():
        # Apparently, the two first columns have merged
        columns = [id_s, rest, *columns[1:]]

    id = int(id_s)
    return (id, columns)



def read_io_completions_per_disk(lines, offset_hint=0):
    """
    Return offset, {disk => (index, <data>)}
    """

    def acc_disks(disks, columns):
        id, columns = extract_id_column(columns)
        if not len(columns) == 7:
            raise ValueError("Wrong number of columns for I/O completions: {}"
                             .format(len(columns)))

        disk = disk_to_location(columns[1])
        io_values = [int(x) for x in columns[2:]]
        disks[disk] = tuple([id, *io_values])
        return disks

    heading_re = regex.compile("^IO completions per disk.*")
    return read_table(lines, start_re=heading_re,
                      re_offset=3, reducer=acc_disks,
                      offset_hint=offset_hint)

def read_io_ompletion_times_per_index(lines, offset_hint=0):
    """
    Return offset, index => (histogram)
    """
    def acc_indices(indices, columns):
        # Fixme: make sure columns have the correct length
        id, columns = extract_id_column(columns)
        if not len(columns) == 17:
            raise ValueError("Wrong number of columns for I/O completions: {}"
                             .format(len(columns)))

        values = tuple([int(x) for x in columns[1:]])
        indices[id] = values
        return indices

    heading_re = regex.compile("^I/O Completion Time Table.*")
    return read_table(lines, start_re=heading_re, re_offset=5,
                      reducer=acc_indices,
                      offset_hint=offset_hint)


def read_disk_overview(lines):
    """
    Disk overview is one row per disk, featuring:
                            Serial                 Disk   Average   Max    Retry  Timeout  Sense Data
    Disk                    Number                 State   I/O      I/O    count  count    1       2      3     4     5     9   B
    """
    def acc_disks(disks, columns):
        if not len(columns) == 14:
            raise ValueError("Invalid disk overview line!")
        disk_cells = columns[:3] + [int(x) for x in columns[3:]]
        disks.append(disk_cells)
        return disks

    return read_table(lines, start_re=underline_re, re_offset=1,
                      reducer=acc_disks, accumulator=[],
                      is_at_end=lambda i, lines: not lines[i].strip())


def identify_headings(lines):
    headings = defaultdict(list)

    for i, line in enumerate(lines):
        if regex.match(r'^(-+\s*)+$', line) and \
           (i+1 >= len(lines) or not lines[i + 1].strip()):
            headings[lines[i-1].strip()].append(i-1)
        elif line.strip() and not regex.match(r'.*\d.*', line) and \
             not regex.match(r'^-+\s*$', line):
            headings[line.strip()].append(i)

    return headings


def read_sense_error(lines, offset_hint=0):
    start_offset = offset_hint
    data = list()

    start_offset, _match = seek_to(lines,
                                   re=sense_error_heading_re,
                                   offset_hint=offset_hint)
    start_offset += 2

    # Accumulate data
    for i, line in enumerate(lines[start_offset:], start=start_offset):
        if not line.strip():
            # Empty line -- terminator
            break
        else:
            data.append(line)

    return i, data


def read_smart_data(lines, offset_hint=0):
    """
    Returns disk_label => {disk_label => {SMART_id => (status, value, worst value, raw value)}}
    """
    data = defaultdict(dict)
    offset, _match = seek_to(lines, re=smart_pages_heading_re,
                             offset_hint=offset_hint,
                             max_search=2)
    offset, disk_match = seek_to(lines, re=disk_re, offset_hint=offset, max_search=3)
    disk = disk_match.group('disk')
    offset += 1

    for i, line in enumerate(lines[offset:], start=offset):
        if not line.strip():
            if not lines[i+1].strip():
                # Two blank lines -- new data segment
                break
            else:
                continue

        maybe_disk = disk_re.match(line)

        if maybe_disk:
            disk = maybe_disk.group('disk')
            log.debug("Found new disk: {}!".format(disk))
            continue
        elif regex.match(r".*Attribute ID.*", line):
            continue
        elif regex.match(r"^---+.*", line):
            continue
        else:
            line_data = line_re.match(line)
            if not line_data:
                log.debug("Line '{}' does not match!".format(line.strip()))
                raise IndexError
            attr_idH, statusH, value, wrst_value, raw_valueH = line_data.groups()

            data[disk][int(attr_idH, 16)] = (int(statusH, 16),
                                             int(value), int(wrst_value),
                                             int(raw_valueH, 16))
    return i, data


def extract_node_data(lines):
    node_data = dict()
    offset, node_data["disk_overview"] = read_disk_overview(lines)
    node_data['headings'] = identify_headings(lines)

    if node_data['headings']['Disk SMART Pages']:
       assert len(node_data['headings']['Disk SMART Pages'])
       smart_pages_location = node_data['headings']['Disk SMART Pages'][0]
    else:
        smart_pages_location = None

    node_data['smart_mystery'] = {}
    node_data['smart_data'] = {}
    # try:
    #     _, node_data["sense_error"] = read_sense_error(lines,
    #                                                    offset_hint=offset)

    # except IndexError:
    #     log.warning("No sense data found!")

    if smart_pages_location is not None:
        try:
            _, node_data["smart_mystery"] = read_smart_pages(
                lines,
                offset_hint=smart_pages_location)
        except IndexError:
            log.debug("No SMART pages found!")

        try:
            _, node_data["smart_data"] = read_smart_data(
                lines,
                offset_hint=smart_pages_location)
        except IndexError:
            log.debug("No non-obfuscated SMART data")

    offset, node_data['io_completions'] = read_io_completions_per_disk(
        lines,
        offset_hint=offset)

    _, node_data['io_completion_times'] = read_io_ompletion_times_per_index(
        lines, offset_hint=offset)

    return node_data


def read_cluster_data_snapshot(filename):
    """
    Open a gzipped text data file and parse it.

    Returns a dictionary on the format:
    node_name =>
      - disk_overview => [disk label, serial number,
         disk state, average I/O, max I/O, retry count, timeout count,
         sense data 1 ... sense data 9, sense data B]
      - headings => { heading => [line number]}
      - sense_error => {}
      - smart_mystery => [matrix]
      - smart_data => {disk_label => {SMART_id => (status, value, worst value, raw value)}}
      - io_completions => disk => (index, CPIO blocks read, blocks read, blocks written, verifies, MaxQ)
      - io_completions_times => index => (4ms, 8ms, 16ms, 30ms, 50ms,
        100ms, 200ms, 400ms, 800ms, 2000ms, 4000ms, 16000ms, 30 000ms,
        45 000ms, 60 000 ms, 100 000ms)
    """
    cluster_data = defaultdict(list)
    current_node = None

    with gz.open(filename, 'r') as f:
        for b_line in f:
            line = b_line.decode(FILE_CODEC)
            maybe_match = node_re.match(line)
            if maybe_match:
                log.debug("Found a new node {}".
                          format(maybe_match.group('node_name')))
                current_node = maybe_match.group('node_name')
            elif current_node:
                cluster_data[current_node].append(line)
            else:
                pass
#                print("W: skipping line!")

        if not current_node:
            raise ValueError("Did not find a single node declaration!")

    for k, value in cluster_data.items():
        log.debug("Processing data for node {}".format(k))
        cluster_data[k] = extract_node_data(value)

    return cluster_data


def index_files(path):
    """
    Returns a dictionary on the format node name -> [(capture date, file
    name w/full path)], ordered by capture date.

    Dates are timezone-aware, timestamps presumed to be UTC.
    """

    filenames = defaultdict(list)

    for subdir, _dirs, files in os.walk(path):
        for file in files:
            match = file_re.match(file)
            if not match:
                continue
            else:
                node = match.group('node_name')
                timestamp = datetime.datetime.fromtimestamp(
                    float(match.group('timestamp')),
                    pytz.utc)
                full_path = os.path.join(subdir, file)
                filenames[node].append((timestamp, full_path))
    return {k: sorted(v) for k, v in filenames.items()}


def disk_to_location(disk_str):
    """
    Translate a string on the format connector.shelf.bay to shelf.bay,
    with normalised numbers.
    """
    try:
        shelf, bay = [int(x) for x in disk_str.split(".")[1:]]
    except ValueError:
        raise ValueError('Invalid disk string "{}"'.format(disk_str)) from None
    return "{:d}.{:d}".format(shelf, bay)


def disk_types_and_serials_from_path(path):
    """
    Returns a mapping of cluster => disk_location => [(ts, type, firmware revision)])

    All timestamps are timezone-aware, timestamps presumed to be
    UTC. Sorted by timestamp.
    """


    serials_re = regex.compile("db(?<cluster_name>.*?)"
                               "-cluster-mgmt-disk-types\.(?<timestamp>.*?)\.csv\.gz$")

    disk_types_by_cluster = defaultdict(lambda: defaultdict(list))
    for subdir, _dirs, files in os.walk(path):
        for file in files:
            match = serials_re.match(file)
            if not match:
                continue
            else:
                cluster = match.group('cluster_name')
                timestamp = datetime.datetime.fromtimestamp(
                    float(match.group('timestamp')),
                    pytz.utc)
                full_path = os.path.join(subdir, file)

                # Parse the file as CSV
                with gz.open(full_path, 'rt') as data_fp:
                    reader = csv.DictReader(data_fp, delimiter="+")
                    # For some reason, we have two header rows, so skip
                    # the second one too:
                    next(reader)

                    for row in reader:
                        try:
                            disk = disk_to_location(row['disk'])
                            firmware = row['revision']
                            disk_type = row['type']
                            if not firmware or not disk_type:
                                raise ValueError
                        except ValueError as e:
                            log.warning("Invalid disk type data row '%s'",
                                        row)
                            continue

                        disk_types_by_cluster[cluster][disk].append(
                            (timestamp, firmware, disk_type))

    return disk_types_by_cluster



def get_closest_disk_data(serials_index, cluster, disk_location, ts):
    """
    Takes a timezone-aware timestamp and finds the closest associated
    firmware and type data for that disk.

    Returns None if the data wasn't available, otherwise F/W version, disk type
    """
    disk_data_points = serials_index.get(cluster, {}).get(disk_location, [])

    if not disk_data_points:
        log.warning("Found NO type, F/W data for {}:{}"
                    .format(cluster, disk_location))
        return (None, None)

    best_match = (disk_data_points[0][1], disk_data_points[0][2])
    match_distance = abs(ts - disk_data_points[0][0])
    for data_ts, fw_version, disk_type in disk_data_points[1:]:
        distance = abs(ts - data_ts)
        if distance < match_distance:
            best_match = (fw_version, disk_type)
            match_distance = distance

        if distance > match_distance:
            # If distance is growing, we are moving away from the
            # solution
            break
    return best_match


def process_data_file(ts, file_name):
    log.debug("Processing {}".format(file_name))
    return ts, read_cluster_data_snapshot(file_name)


def parse_files(directory_name):
    """
    Load a set of files named db<cluster name>-cluster-mgmt.<timestamp>.data.gz.

    Yields:
    cluster_name, [(capture date, parsed data)]
    """

    files = index_files(directory_name)

    for cluster_name, ts_and_filenames in files.items():
        log.info("Parsing %d files for cluster %s", len(ts_and_filenames),
                 cluster_name)
        if ts_and_filenames:
            yield cluster_name, (process_data_file(*ts_f)
                                 for ts_f in ts_and_filenames)
        else:
            yield cluster_name, []



def analyse_data(cluster, ts_and_data):
    """
    Expects:
    cluster name, (timestamp, node_data)

    Returns:
    - disk_count: number of disks
    - node_count: number of nodes
    - headings: set of all distinct headings seen
    - overview_values[disk] => [(timestamp, row_value_snapshot)]
    - smart_data[disk] => [(timestamp, [attribute] => (status, value, worst value, raw value))]
    - smart_mystery[disk]   => [(timestamp, matrix_snapshot)]
    - disk_io_stats[disk] => [(timestamp, (CPIO blocks read, blocks read, written, verifies, MaxQ))]
    - disk_errors[disk] => [(timestamp, [(cyl, head, sector)])]
    - smart_mystery_diff[disk] => {(row, column) => {value => last timestamp seen}
    - smart_data_diff[disk] => {attribute => {value => count}}
    """
    log.info("Analysing timeseries data for %s", cluster)

    disk_count = None
    node_count = None
    headings = set()
    overview_values = [] #  fixme!
    smart_mystery = defaultdict(list)
    smart_data = defaultdict(list)
    smart_mystery_diff = defaultdict(lambda: defaultdict(dict))
    smart_data_diff = defaultdict(lambda: defaultdict(Counter))


    for ts, cluster_data in ts_and_data:
        data_point_disk_count = sum([len(node['disk_overview'])
                                     for node in cluster_data.values()])
        data_point_node_count = len(cluster_data.keys())

        if disk_count != data_point_disk_count:
            if disk_count:
                log.warning("Updating disk count from {} to {}"
                            .format(disk_count, data_point_disk_count))
            disk_count = data_point_disk_count

        if node_count != data_point_node_count:
            if node_count:
                log.warning("Updating node count from {} to {}"
                            .format(node_count, data_point_node_count))
            node_count = data_point_node_count
        for node_data in cluster_data.values():
            for disk, smart_values in node_data['smart_data'].items():
                assert isinstance(smart_values, dict)
                smart_data[disk].append((ts, smart_values))

            for disk, matrix in node_data['smart_mystery'].items():
                if matrix:
                    smart_mystery[disk].append((ts, matrix))

            headings = headings.union(set(node_data['headings'].keys()))

    for disk, tseries_data in smart_mystery.items():
        for (ts, matrix) in tseries_data:
            for row_no, row in enumerate(matrix):
                for col_no, cell in enumerate(row):
                    current_value = smart_mystery_diff[disk][(row_no, col_no)].get(cell,
                                                                                   BEGINNING_OF_TIME)
                    if ts > current_value:
                        smart_mystery_diff[disk][(row_no, col_no)][cell] = ts

    for disk, tss_and_data in smart_data.items():
        for _ts, data in tss_and_data:
            for smart_attr, value_set  in data.items():
                _status, value, _worst_value, _raw_value = value_set
                smart_data_diff[disk][smart_attr][value] += 1

    return {
        'disk_count': disk_count,
        'node_count': node_count,
        'headings': headings,
        'overview_values': overview_values,
        'smart_mystery': smart_mystery,
        'smart_mystery_diff': smart_mystery_diff,
        'smart_data_diff': smart_data_diff,
        'smart_data': smart_data,
    }


def show_smart_diff(cluster_data):
    for disk, disk_data in cluster_data['smart_mystery_diff'].items():
        print("Disk: {}".format(disk))
        for ((row, col), count) in disk_data.items():
            if len(count) > 1:
                profile = format_counter(count)
            else:
                value = list(count.keys())[0]
                if value != 0:
                    #profile = "Const {}".format(value)
                    profile = None
                else:
                    profile = None

            if profile:
                print("\t{},{}: {}".format(row, col, profile))


    for disk, disk_data in cluster_data['smart_data_diff'].items():
        for attribute, count in disk_data.items():
            if len(count) > 1:
                profile = format_counter(count)
            else:
                value = list(count.keys())[0]
                if value != 0:
                    #profile = "Const {}".format(value)
                    profile = None
                else:
                    profile = None

            if profile:
                print("SMART for {} {}: {}".format(disk, attribute, profile))


def deserialise_csv_row(row):
    # fixme: re-parse the SMART data
    data = row
    data['timestamp'] = datetime.datetime.fromtimestamp(
        float(row['timestamp']))
    smart_data = dict([eval(row['smart_data'])]) if row['smart_data'] else None
    data['smart_data'] = smart_data
    print(data)
    return data


def read_data_file(data_file_path):
    """
    Read a canned data file, yielding its rows as a generator.

    Each row is a dictionary with the following keys:
    - timestamp
    - cluster
    - disk -- shelf.bay
    - smart_data -- a dict of SMART values, or None if not present at this data point
    - smart_mystery -- a tuple of values, or None if not present at this data point
    """
    log.debug("Reading data file {}".format(data_file_path))
    with gz.open(data_file_path, 'rt') as data_fp:
        reader = csv.DictReader(data_fp,
                                fieldnames=DATA_FILE_FIELDS,
                                restval=None,
                                delimiter=CSV_DELIMITER)
        for row in reader:
            yield deserialise_csv_row(row)


def take_duration(row_stream, hours=0, minutes=0, seconds=0):
    """
    Take the first hours, minutes or seconds of data from a data stream.
    """
    first_seen_timestamp = None

    for row in row_stream:
        if not first_seen_timestamp:
            first_seen_timestamp = row['timestamp']
        time_since_first_entry = first_seen_timestamp - row['timestamp']
        target_delta = datetime.timedelta(hours=hours, minutes=minutes,
                                          seconds=seconds)

        if target_delta and abs(time_since_first_entry) > target_delta:
            break
        else:
            yield row


def smart_counts_per_cluster(es):
    """
    Return a dictionary of the number of SMART entries in the data file, by cluster:

    cluster => number of disks with smart data, number of disks with mystery data
    """
    # As the SMART data is a dictionary,
    smart_counts = es_count_disks_by_cluster(es, ES_INDEX, q="smart.1: *")
    mystery_counts = es_count_disks_by_cluster(es, ES_INDEX, q="smart_mystery: *")

    unified_counts = {}
    for cluster in set(smart_counts.keys()).union(set(mystery_counts.keys())):
        smart_count = smart_counts.get(cluster, 0)
        mystery_count = mystery_counts.get(cluster, 0)
        unified_counts[cluster] = (smart_count, mystery_count)

    return unified_counts



def patch_list_with_offset(previous_data, replacement_values, offset):
    """
    Replace the entries of a list, with a given offset.
    """
    if replacement_values:
        head_of_previous = previous_data[:offset]
        return [*head_of_previous, *replacement_values]
    else:
        return previous_data


def data_to_list(data_by_node):
    """
    Take data ordered by node, as returned by
    read_cluster_data_snapshot(), and transform it to a flat list, as
    returned by generate_csv_data().
    """
    # Fixme: pad the entries that doesn't have SMART data

    overview_data_length = 16

    # label -> [data_fields]
    disk_data = dict()

    for _node, node_data in data_by_node.items():
        for data_row in node_data['disk_overview']:
            raw_label, overview_data = data_row[0], data_row[1:]
            # Remove everything before the first . (the connector)
            proper_label = disk_to_location(raw_label)
            disk_data[proper_label] = overview_data

        for disk_label, smart_dict in node_data['smart_data'].items():
            # smart_data_by_field_no = sum([list(values) for key, values
            #                               in sorted(smart_dict.items())], [])
            disk_data[disk_label] = patch_list_with_offset(
                disk_data.get(disk_label, []),
                replacement_values=[sorted(smart_dict.items())],
                offset=overview_data_length)

    data_by_disk = []
    for disk_label, disk_values in disk_data.items():
        data_by_disk.append([disk_label, *disk_values])

    return data_by_disk


def generate_csv_data(data_by_cluster):
    """
    Generate a list of data on the format:
    [timestamp, cluster, disk label, serial number, state, average I/O,
    max I/O, retry count, timeout count, sense data 1...9, sense data B, SMART data]

    Ordered by timestamp.
    """
    data = []

    for cluster, ts_and_data in data_by_cluster.items():
        for ts, data_by_node in ts_and_data:
            for disk_data_point in data_to_list(data_by_node):
                data.append([ts.timestamp(), cluster, *disk_data_point])

    return sorted(data)


def parse_incrementally(file_index, data_file):
    """
    Incrementally parse out the data from a sorted file index, as given
    by file_index().
    """

    timestamp_file = "{}.timestamp".format(data_file)
    try:
        with open(timestamp_file, 'r') as f:
            first_line = f.readline()
            last_seen = datetime.datetime.fromtimestamp(float(first_line))
    except FileNotFoundError:
        last_seen = BEGINNING_OF_TIME

    data = defaultdict(list)
    number_of_files = sum([len(x) for x in file_index.values()])
    current_count = 0
    last_seen_by_cluster = {}

    for cluster_name in file_index.keys():
        last_seen_by_cluster[cluster_name] = last_seen

    for cluster_name, ts_plus_filenames in file_index.items():
        for ts, filename in ts_plus_filenames:
            current_count += 1
            # Should be ordered ascending by timestamp, remember:
            last_seen_by_cluster[cluster_name] = ts
            if ts < last_seen:
                continue
            log.info("Parsed %s/%s files...", current_count, number_of_files)
            data[cluster_name].append(process_data_file(ts, filename))
            if current_count >= 14000:
                break


    csv_data = generate_csv_data(data)
    if csv_data:
        last_seen = min(last_seen_by_cluster.values())

    # Overwrite the first line with the new timestamp
    with open(timestamp_file, 'w') as f:
        f.write(str(last_seen.timestamp()))

    # Append the new data
    with gz.open(data_file, 'at') as f:
        filewriter = csv.writer(f, delimiter=CSV_DELIMITER)
        for row in csv_data:
            filewriter.writerow(row)


def prepare_es_data(cluster, parsed_data, type_fw_index):
    """
    Generate prepared ES data from pre-parsed data, as returned by
    process_data_file().

    To be fed into es_import()
    """
    timestamp, cluster_data = parsed_data

    # Ignore node names
    for node_data in cluster_data.values():
        for disk_data in node_data['disk_overview']:
            disk_location = disk_to_location(disk_data[0])
            smart_data = node_data['smart_data'].get(disk_location, None)
            smart_mystery = node_data['smart_mystery'].get(disk_location, None)
            completions = node_data['io_completions'].get(disk_location)
            assert completions
            fw_version, disk_type = get_closest_disk_data(type_fw_index,
                                                          cluster,
                                                          disk_location,
                                                          timestamp)
            completion_times = node_data['io_completion_times']\
                               .get(completions[0])
            yield {
                '_index': as_es_index(ES_INDEX_BASE, timestamp),
                '_type': "document",
                '_source': {
                    'cluster_name': cluster,
                    'disk_location': disk_location,
                    '@timestamp': timestamp,
                    'smart': smart_data,
                    'smart_mystery': smart_mystery,
                    'serial': disk_data[1],
                    'state': disk_data[2],
                    'avg_io': disk_data[3],
                    'max_io': disk_data[4],
                    'retry_count': disk_data[5],
                    'timeout_count': disk_data[6],
                    'sense_data1': disk_data[7],
                    'sense_data2': disk_data[8],
                    'sense_data3': disk_data[9],
                    'sense_data4': disk_data[10],
                    'sense_data5': disk_data[11],
                    'sense_data9': disk_data[12],
                    'sense_dataB': disk_data[13],
                    'type': disk_type,
                    'fw_version': fw_version,
                    # Remember, the first column of the tuple is the ID:
                    'io_completions': completions[1:],
                    'io_completion_times': completion_times,
                }
            }


def es_get_data(es, index):
    """
    Generate a lightly parsed Elasticsearch data block for the low-level
    data.

    N.B. it is not ordered.
    """
    res = scan(es, index=index, q=LOW_LEVEL_DATA_Q)
    for x in res:
        data = x['_source']
        data['@timestamp'] = dateparser.parse(data['@timestamp'])
        yield data


def es_get_disks(es, index):
    """
    Return a dictionary of cluster => [disks] as stored in the current
    system. Disks are integer tuples of shelf number, bay number.
    """
    q = {"size": 1,
         "aggs": {
             "group_by_cluster": {
                 "terms": {
                     "size": 0,
                     "field": "cluster_name"
                 },
                 "aggs": {
                     "disks": {
                         "terms": {
                             "size": 0,
                             "field": "disk_location"
                         }}}}}}

    cluster_disks = {}
    res = es.search(index=index, body=q)
    buckets = get_buckets(res, 'group_by_cluster')
    for bucket in buckets:
        cluster = bucket['key']
        disk_buckets = bucket['disks']['buckets']
        disks = sorted([tuple(map(int, b['key'].split(".")))
                        for b in disk_buckets])
        cluster_disks[cluster] = disks

    return cluster_disks


def as_es_index(prefix, datetime):
    """
    Generate a canonical ES index name from a datetime and a prefix.
    """
    return "{}-{:%Y-%m-%d-%h}".format(prefix, datetime)


def es_get_high_water_mark(es, index):
    """
    Get the high-water-mark timestamp per-cluster from ES.

    Will return BEGINNING_OF_TIME if there was nothing.

    cluster_name => last seen timestamp
    """
    per_cluster = defaultdict(lambda: BEGINNING_OF_TIME)
    q = {"size": 0,
         "aggs": {
             "group_by_cluster": {
                 "terms": {
                     "size": 0,
                     "field": "cluster_name",
                     "order": {
                         "max_time": "desc"
                     }
                 },
                 "aggs": {
                     "max_time": {
                         "max": {
                             "field": "@timestamp"
                         }}}}}}

    res = es.search(index=index, body=q)
    try:
        buckets = res['aggregations']['group_by_cluster']['buckets']
    except KeyError:
        # Database was empty
        buckets = []

    for bucket in buckets:
        cluster = bucket['key']
        timestamp = dateparser.parse(bucket['max_time']['value_as_string'])
        per_cluster[cluster] = timestamp
    return per_cluster


def es_import(es, documents):
    """
    Take a generator of documents and index them.
    """
    for result in parallel_bulk(es, documents, raise_on_error=True):
        succeeded, description = result
        if not succeeded:
            log.error(description)
        else:
            log.debug(description)


def file_index_to_triplets(file_index):
    """
    Return tuples of cluster name, timestamp, filename, from an index.
    """
    for cluster_name, ts_plus_filenames in file_index.items():
        for ts, filename in ts_plus_filenames:
            yield (cluster_name, ts, filename)


def is_actual(cn_ts_fn, last_seen):
    """
    Takes a dictionary of last seen values for clusters and a stream of
    cluster name, timestamp, file name, returns True if ts is strictly after
    last_seen for that cluster, False otherwise.
    """
    cluster, ts, fn = cn_ts_fn
    if ts <= last_seen[cluster]:
        log.debug("Skipped {}".format(fn))
        return False
    else:
        log.debug("Kept {} for {} because {} > {}"
                  .format(fn, cluster, str(ts), str(last_seen[cluster])))
        return True


def estimate_rate(current_rate, min_rate, max_rate, avg_rate):
    avg_weight = 5
    min_weight = 3
    present_weight = 4
    max_weight = 1
    sum_weights = min_weight + max_weight + present_weight + avg_weight

    estimate = sum([present_weight*current_rate,
                    min_weight*min_rate,
                    max_weight*max_rate,
                    avg_weight*avg_rate])/sum_weights

    return estimate


def file_index_to_es_data(file_index, last_seen, type_fw_index, throttle=1):
    """
    Generate entries to insert into es_import from a file index and
    high-water-mark dictionary.
    """

    interesting_triplets = [triplet for
                            triplet in
                            file_index_to_triplets(file_index)
                            if is_actual(triplet, last_seen)]

    cluster_ts_filenames = [t for i, t in enumerate(interesting_triplets)
                            if i % throttle == 0]

    throttled_count = len(interesting_triplets) - len(cluster_ts_filenames)

    total_num_files = sum([len(x) for x in file_index.values()])
    num_files_used = len(cluster_ts_filenames)
    skipped_files = total_num_files - num_files_used
    interval_start_time = time.time()
    interval_length = 20
    min_rate = 999999999
    max_rate = 0
    average_rate = 0

    log.info("Starting upload of %d files, skipping %d files (%d throttled)",
             num_files_used, skipped_files, throttled_count)

    for i, (cluster, ts, filename) in enumerate(cluster_ts_filenames):
        if i % interval_length == 0 and i != 0:
            current_time = time.time()
            interval_seconds = current_time - interval_start_time
            interval_rate = interval_length/interval_seconds
            average_rate = ((average_rate * (i-1)) + interval_rate)/i
            min_rate = min(interval_rate, min_rate)
            max_rate = max(interval_rate, max_rate)
            files_remaining = num_files_used - i
            estimated_rate = estimate_rate(interval_rate, min_rate, max_rate,
                                           average_rate)
            seconds_remaining = files_remaining/estimated_rate
            time_remaining = timedelta(seconds=seconds_remaining)
            log.info(("Synced %d/%d files (%d skipped)."
                      " Meaning %d [%d--%d] files/s,"
                      " %s remaining at this rate."),
                     i,
                     num_files_used,
                     skipped_files,
                     interval_rate,
                     min_rate,
                     max_rate,
                     str(time_remaining))
            interval_start_time = time.time()
            if interval_seconds < 3 or interval_seconds > 7:
                interval_length = int(estimated_rate * 5)
                log.debug("Interval ran in %d s, adjusting interval length to %d",
                         interval_seconds,
                         interval_length)

        for data_point in prepare_es_data(cluster,
                                          process_data_file(ts, filename),
                                          type_fw_index):
            yield data_point


def parse_into_es(es, file_index, type_fw_index, throttle=1):
    """
    Intelligently take a file index dictionary and sync it with an ES
    instance.

    If throttle is set to an integer, use only every nth record.
    """
    last_seen = es_get_high_water_mark(es, index=ES_INDEX)
    log.info("Using high-water mark: {}".format(
        ", ".join(["{}: {}".format(c, str(v)) for c, v in last_seen.items()])))

    es_import(es, file_index_to_es_data(file_index,
                                        last_seen,
                                        type_fw_index,
                                        throttle=throttle))


def runtime_statistics(runtimes):
    return {'median': statistics.median(runtimes),
            'mean': statistics.mean(runtimes),
            'max': max(runtimes),
            'min': min(runtimes),
            'std.dev.': statistics.pstdev(runtimes),
    }


def profile_parser(data_directory):
    NUM_FILES = 20
    cluster_ts_filenames = list(file_index_to_triplets(
        index_files(data_directory)))[:NUM_FILES]

    runtimes = []
    logging.getLogger('parse_emails').setLevel(logging.WARNING)
    for _ in range(0, 10):
        with timed("Parse {} items of raw data"
                   .format(len(cluster_ts_filenames)),
                   time_record=runtimes):
            for _, ts, filename in cluster_ts_filenames:
                # Force realisation of generator:
                list(process_data_file(ts, filename))
    print(runtime_statistics(runtimes))


def make_incremental_es(es, args):
    type_fw_index = disk_types_and_serials_from_path(args.data_directory)
    parse_into_es(es, index_files(args.data_directory), type_fw_index,
                  throttle=args.throttle)


def print_disks(es, args):
    print("Disk count per cluster:")
    print("\t Clstr \t First \t Last \t Total")
    print("\t--------------------------------")
    sum_count = 0
    for cluster, disks in sorted(es_get_disks(es, index=ES_INDEX).items(),
                                 key=lambda x: len(x[1]), reverse=True):
        print("\t {} \t {:2d}.{:2d} \t {:2d}.{:2d} \t {:4d}"
              .format(cluster, *disks[0], *disks[-1], len(disks)))
        sum_count += len(disks)
    print("\t--------------------------------")
    print("\t Sum: \t \t \t {:4d}".format(sum_count))


def print_smart_report(es, args):
    smart_total, mystery_total = 0, 0
    for cluster_counts in smart_counts_per_cluster(es).values():
        smart, mystery = cluster_counts
        smart_total += smart
        mystery_total += mystery
    print("{}/{} disks had mystery SMART data, {} had normal SMART data"
          .format(mystery_total, count_disks(es), smart_total))


def profile_parser_report(es, args):
    profile_parser(args.data_directory)


if __name__ == '__main__':
    parser = common.make_es_base_parser()
    common.add_subcommands(parent=parser,
                           descriptions=[
                               ('incremental_es',
                                "Sync snapshot data to ES",
                                [
                                    (['--throttle', '-t'],
                                     {'type': int,
                                      'default': 1,
                                      'help': "upload every Nth record",
                                      'dest': 'throttle'}),
                                    (['data_directory'],
                                     {'type': str}),
                                ],
                                make_incremental_es),
                               ('disks',
                                "List disks and their values",
                                [],
                                print_disks),
                               ('smart_stats',
                                "Show statistics about SMART data",
                                [],
                                print_smart_report),
                               ('profile_parse',
                                "Print profiling data for the file parser",
                                [],
                                profile_parser_report),
                           ])

    args = parser.parse_args()
    daiquiri.setup()
    common.set_log_level_from_args(args, log)
    es = common.es_conn_from_args(args)
    common.run_subcommand(args, es=es)
