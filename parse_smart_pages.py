#!/usr/bin/env python3
from parse_emails import format_counter

import sys
import regex
from collections import defaultdict, Counter

disk_re = regex.compile("^Disk\s+0a.(?P<disk>.*):")
row_re = regex.compile(r".*:\s+(0x([0-9a-f]{2})+\s*)+")

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

    for line in f:
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


if __name__ == '__main__':
    filename = sys.argv[1]
    diff_matrix = defaultdict(Counter)

    with open(filename, 'r') as f:
        results = read_page_blocks(f)
        for disk, matrix in results.items():
            for row_no, row in enumerate(matrix):
                for col_no, cell in enumerate(row):
                    diff_matrix[(row_no, col_no)][cell] += 1

    for (row, col), count in diff_matrix.items():
        if len(count) > 1:
            profile = format_counter(count)
        else:
            value = list(count.keys())[0]
            if value != 0:
                profile = "Const {}".format(value)
            else:
                profile = None

        if profile:
            print("{},{}: {}".format(row, col, profile))
