#!/usr/bin/env python3
from parse_emails import format_counter, timed

import sys
import regex
from collections import defaultdict, Counter
import gzip as gz
import logging
import os
import datetime
from itertools import zip_longest
import multiprocessing

log = logging.getLogger()
ch = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter('%(name)s %(levelname)s: %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)
log.setLevel(logging.INFO)

disk_re = regex.compile("^Disk\s+.*?\.(?P<disk>\d+.\d+):?")
row_re = regex.compile(r".*:\s+(0x([0-9a-f]{2})+\s*)+")
node_re = regex.compile(r"^Node: db(?P<node_name>.+\d+)\s*")

FILE_CODEC = 'ascii'


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
            log.warning("Gave up after {} lines".format(max_search))
            break

    log.debug("Could not find a match for {} at offset {}".format(re, offset))
    raise IndexError("Could not find any match for {} after line {}"
                     .format(re, offset_hint))


def read_smart_pages(lines, offset_hint=0):
    disks = {}
    current_table = []

    # Seek to "Disk SMART Pages"
    # skip --- line
    offset, _match = seek_to(lines, re=regex.compile(r".*Disk SMART Pages.*",
                                                     offset_hint=offset_hint))
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
    line_index, _match = seek_to(lines, re=regex.compile('^-+'))
    line_index += 1

    for line_index, line in enumerate(lines[line_index:], start=line_index):
        if not line.strip():
            # Empty line -- terminator
            break
        else:
            # The final "column" is junk:
            disk_strs = regex.split("\s+", line)[:-1]
            disk_cells = disk_strs[:3] + [int(x) for x in disk_strs[3:]]
            disk_overview.append(disk_cells)

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
                                   re=regex.compile('Disk LOG Sense Error.*'),
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
    offset, _match = seek_to(lines, re=regex.compile(r".*Disk SMART Pages.*",
                                                     offset_hint=offset_hint))
    offset, disk_match = seek_to(lines, re=disk_re, offset_hint=offset, max_search=3)
    disk = disk_match.group('disk')
    offset += 1
    line_re = regex.compile("^\s+(.*?)h\s+(.*?)h\s+(\d+)\s+(\d+)\s+(.*?)h\s+.*")

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
                log.error("Line '{}' does not match!".format(line.strip()))
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
    node_data['smart_mystery'] = {}
    node_data['smart_data'] = {}
    # try:
    #     _, node_data["sense_error"] = read_sense_error(lines,
    #                                                    offset_hint=offset)

    # except IndexError:
    #     log.warning("No sense data found!")

    try:
        _, node_data["smart_mystery"] = read_smart_pages(lines,
                                                         offset_hint=offset)
    except IndexError:
        log.warning("No SMART pages found!")

    try:
        _, node_data["smart_data"] = read_smart_data(lines, offset_hint=offset)
    except IndexError:
        log.warning("No non-obfuscated SMART data")

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
    name w/full path)]
    """

    filenames = defaultdict(list)
    file_re = regex.compile("db(?<node_name>.*?)"
                            "-cluster-mgmt\.(?<timestamp>.*?)\.data\.gz$")

    for subdir, _dirs, files in os.walk(path):
        for file in files:
            match = file_re.match(file)
            if not match:
                continue
            else:
                node = match.group('node_name')
                timestamp = datetime.datetime.fromtimestamp(
                    float(match.group('timestamp')))
                full_path = os.path.join(subdir, file)
                filenames[node].append((timestamp, full_path))
    return {k: sorted(v) for k, v in filenames.items()}


def process_data_file(ts, file_name):
    log.debug("Processing {}".format(file_name))
    return ts, read_cluster_data_snapshot(file_name)


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def ungroup(iterable):
    return sum(iterable, [])


def load_data_chunk(chunk):
    """
    Process a piece of data on the format
    [(date captured, file_name)], returning
    [(date caputured, parsed data)].
    """
    with timed(task_name="Load data chunk"):
        return [process_data_file(*ts_f) for ts_f in chunk if ts_f]


def parse_files(directory_name):
    """
    Load a set of files named db<cluster name>-cluster-mgmt.<timestamp>.data.gz.

    Returns:
    all_data[cluster_name] => [(capture date, parsed data)]
    """

    files = index_files(directory_name)
    p = multiprocessing.Pool()

    all_data = {}
    work_chunk_size = 400

    for cluster_name, ts_and_filenames in files.items():
        work_chunks = grouper(ts_and_filenames[:200], work_chunk_size)
        all_data[cluster_name] = ungroup(map(load_data_chunk, work_chunks))

    return all_data


def analyse_data(cluster, ts_and_data):
    """
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
    log.info("Analysing timeseries data for {}".format(cluster))

    disk_count = None
    node_count = None
    headings = set()
    overview_values = []
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
                                                                                   datetime.datetime.fromtimestamp(0))
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


if __name__ == '__main__':
    with timed(task_name="Parse everything"):
        data = parse_files(sys.argv[1])
        #print(list(data.values())[0])

    with timed(task_name="Data analysis"):
        analysed_data = {c: analyse_data(c, ts_data)
                         for c, ts_data in data.items()}

    disk_count = sum([d['disk_count'] for d in analysed_data.values()])
    log.info("Saw {} disks".format(disk_count))

    for cluster, cluster_data in analysed_data.items():
        if cluster_data['smart_data']:
            log.info("Cluster {} had smart data for {}/{} disks"
                  .format(cluster, len(cluster_data['smart_data'].keys()),
                          cluster_data['disk_count']))
        if cluster_data['smart_mystery']:
            log.info("Cluster {} had mystery smart data for {}/{} disks!"
                     .format(cluster,
                             len(cluster_data['smart_mystery'].keys()),
                             cluster_data['disk_count']))

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
