#!/usr/bin/env python3
from parse_emails import format_counter

import sys
import regex
from collections import defaultdict, Counter
import gzip as gz

disk_re = regex.compile("^Disk\s+0a.(?P<disk>.*):")
row_re = regex.compile(r".*:\s+(0x([0-9a-f]{2})+\s*)+")
node_re = regex.compile(r"^Node: db(?P<node_name>.+\d+)\s*")

FILE_CODEC = 'ascii'


    # diff_matrix = defaultdict(Counter)

    # with gz.open(filename, 'r') as f:
    #     results = read_page_blocks(f)
    #     for disk, matrix in results.items():
    #         for row_no, row in enumerate(matrix):
    #             for col_no, cell in enumerate(row):
    #                 diff_matrix[(row_no, col_no)][cell] += 1

    # for (row, col), count in diff_matrix.items():
    #     if len(count) > 1:
    #         profile = format_counter(count)
    #     else:
    #         value = list(count.keys())[0]
    #         if value != 0:
    #             profile = "Const {}".format(value)
    #         else:
    #             profile = None

    #     if profile:
    #         print("{},{}: {}".format(row, col, profile))


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


def read_page_blocks(block):
    disks = {}
    disk = None
    current_table = []

    for b_line in f:
        line = b_line.decode("ascii")
        if not line.strip():
            continue
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
    return disks


def read_disk_overview(lines):
    disk_overview = list()
    count = 0
    for line_index, line in enumerate(lines):
        if regex.match('^-+', line):
            # We've found the table!
            break
        else:
            count += 1

    for line_index, line in enumerate(lines[line_index:], start=line_index):
        if not line.strip():
            # Empty line -- terminator
            break
        else:
            # The final "column" is junk:
            disk_cells = regex.split("\s+", line)[:-1]
            disk_overview.append(disk_cells)

    return line_index, disk_overview


def extract_node_data(lines):
    node_data = dict()
    offset_hint, node_data["disk_overview"] = read_disk_overview(lines)
    return node_data


def read_cluster_data_snapshot(filename):
    cluster_data = defaultdict(list)
    current_node = None

    with gz.open(filename, 'r') as f:
        for b_line in f:
            line = b_line.decode(FILE_CODEC)
            maybe_match = node_re.match(line)
            if maybe_match:
                current_node = maybe_match.group('node_name')
            elif current_node:
                cluster_data[current_node].append(line)
            else:
                pass
#                print("W: skipping line!")

        if not current_node:
            raise ValueError("Did not find a single node declaration!")

    for k, value in cluster_data.items():
        cluster_data[k] = extract_node_data(value)

    return cluster_data


if __name__ == '__main__':
    filenames = sys.argv[1:]

    all_data = {}
    skip_count = 0
    disk_count = Counter()
    node_count = Counter()

    for filename in filenames:
        cluster_name = regex.match(".*?db(.*)-cluster.*", filename).group(1)
        if cluster_name not in all_data:
            all_data[cluster_name] = read_cluster_data_snapshot(filename)
        else:
            skip_count += 1

    for cluster, data in all_data.items():
        disk_count[cluster] += sum([len(node['disk_overview'])
                                    for node in data.values()])
        node_count[cluster] += len(data)

    print("Disk drives: {} disk drives ({} in total)"
          .format(format_counter(disk_count),
                  sum(disk_count.values())))
    print("Nodes: {} ({} in total)."
          .format(format_counter(node_count),
                  sum(node_count.values())))
    print("Analysed {} clusters".format(len(all_data.keys())))
