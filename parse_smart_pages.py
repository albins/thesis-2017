#!/usr/bin/env python3
from parse_emails import format_counter, timed

import sys
import regex
from collections import defaultdict, Counter
import gzip as gz
import logging
import os
import datetime
import csv
import statistics

import dateparser
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan, parallel_bulk
import pytz

log = logging.getLogger()
ch = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter('%(name)s %(levelname)s: %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)
log.setLevel(logging.INFO)

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
ELASTIC_ADDRESS = "localhost"
ES_INDEX_BASE = 'netapp-lowlevel'
ES_INDEX = 'netapp-lowlevel-*'
LOW_LEVEL_DATA_Q = '*'


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


def read_disk_overview(lines):
    """
    Disk overview is one row per disk, featuring:
                            Serial                 Disk   Average   Max    Retry  Timeout  Sense Data
    Disk                    Number                 State   I/O      I/O    count  count    1       2      3     4     5     9   B
    """
    disk_overview = list()
    line_index, _match = seek_to(lines, re=underline_re)
    line_index += 1

    for line_index, line in enumerate(lines[line_index:], start=line_index):
        if not line.strip():
            # Empty line -- terminator
            break
        else:
            # The final "column" is junk:
            disk_strs = regex.split("\s+", line)[:-1]
            try:
                disk_cells = disk_strs[:3] + [int(x) for x in disk_strs[3:]]
            except ValueError as e:
                log.error("Error processing line %d '%s': %s",
                          line_index, line.strip(), str(e))
                continue
            if len(disk_cells) == 14:
                disk_overview.append(disk_cells)
            else:
                log.error("Invalid line %s generated from line %d: '%s'",
                          str(disk_cells), line_index, line.strip())

    return line_index, disk_overview


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


def smart_counts_per_cluster(data_file_path):
    """
    Return a dictionary of the number of SMART entries in the data file, by cluster:

    cluster => number of disks with smart data, number of disks with mystery data
    """

    smart_disks = defaultdict(set)
    smart_mystery_disks = defaultdict(set)

    for row in take_duration(read_data_file(data_file_path), hours=1):
        if row['smart_data']:
            smart_disks[row['cluster']].add(row['disk'])
        if row['smart_mystery']:
            smart_mystery_disks[row['cluster']].add(row['disk'])

    all_cluster_names = set(smart_disks.keys()).union(set(smart_mystery_disks.keys()))

    return {cluster: (len(smart_disks[cluster]), len(smart_mystery_disks[cluster]))
                      for cluster in all_cluster_names}


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
            proper_label = ".".join(raw_label.split(".")[1:])
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


def count_disks(data_file_path):
    disks = set()
    log.debug("Reading disks from {}".format(data_file_path))
    for row in take_duration(read_data_file(data_file_path), minutes=30):
        disks.add((row['cluster'], row['disk']))

    return len(disks)


def prepare_es_data(cluster, parsed_data):
    """
    Generate prepared ES data from pre-parsed data, as returned by
    process_data_file().

    To be fed into es_import()
    """
    timestamp, cluster_data = parsed_data

    # Ignore node names
    for node_data in cluster_data.values():
        for disk_data in node_data['disk_overview']:
            disk_location = ".".join(disk_data[0].split(".")[1:])
            yield {
                '_index': as_es_index(ES_INDEX_BASE, timestamp),
                '_type': "document",
                '_source': {
                    'cluster_name': cluster,
                    'disk_location': disk_location,
                    '@timestamp': timestamp,
                    'smart': node_data['smart_data'].get(disk_location, None),
                    'smart_mystery': node_data['smart_mystery'].get(
                        disk_location, None),
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


def as_es_index(prefix, datetime):
    """
    Generate a canonical ES index name from a datetime and a prefix.
    """
    return "{}-{:%Y-%m-%d}".format(prefix, datetime)


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
    for result in parallel_bulk(es, documents):
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


def file_index_to_es_data(file_index, last_seen):
    """
    Generate entries to insert into es_import from a file index and
    high-water-mark dictionary.
    """
    cluster_ts_filenames = [triplet for
                            triplet in file_index_to_triplets(file_index)
                            if is_actual(triplet, last_seen)]
    total_num_files = sum([len(x) for x in file_index.values()])
    num_files_used = len(cluster_ts_filenames)
    skipped_files = total_num_files - num_files_used

    for i, (cluster, ts, filename) in enumerate(cluster_ts_filenames):
        if i % 20 == 0:
            log.info("Synced %d/%d files (%d skipped)...", i, num_files_used,
                     skipped_files)
        for data_point in prepare_es_data(cluster,
                                          process_data_file(ts, filename)):
            yield data_point


def parse_into_es(es, file_index):
    """
    Intelligently take a file index dictionary and sync it with an ES
    instance.
    """
    last_seen = es_get_high_water_mark(es, index=ES_INDEX)
    log.info("Using high-water mark: {}".format(
        ", ".join(["{}: {}".format(c, str(v)) for c, v in last_seen.items()])))

    es_import(es, file_index_to_es_data(file_index, last_seen))


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


if __name__ == '__main__':
    tasks = sys.argv[1:]
    data_directory = sys.argv[1]
    es = Elasticsearch([ELASTIC_ADDRESS])

    if "incremental_es" in tasks:
        parse_into_es(es, index_files(data_directory))

    if "disks" in tasks:
        print("Saw {} disks".format(count_disks(es)))
    if "smart_stats" in tasks:
        smart_total, mystery_total = 0, 0
        for cluster_counts in smart_counts_per_cluster(es).values():
            smart, mystery = cluster_counts
            smart_total += smart
            mystery_total += mystery

        print("{}/{} disks had mystery SMART data, {} had normal SMART data"
              .format(mystery_total, count_disks(es), smart_total))
    if "smart_changes" in tasks:
        pass
    if "mystery_changes" in tasks:
        pass

    if "profile_parse" in tasks:
        profile_parser(data_directory)
