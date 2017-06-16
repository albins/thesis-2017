#!/usr/bin/env python3
from parse_emails import format_counter

from collections import Counter, defaultdict
from datetime import datetime, timedelta

import pytz
import regex
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
import dateparser

ELASTIC_ADDRESS = "db-51167.cern.ch:9200"
ES_INDEX = "syslog-*"
READERR_REPAIR_Q = "event_type: raid.rg.readerr.repair.*",
FAIL_REASONS = ["raid.config.filesystem.disk.failed",
                "raid.disk.online.fail",
                "disk.failmsg",
                "disk.partner.diskFail",
                "raid.fdr.fail.disk.sick",
                "raid.fdr.reminder"]

DISK_FAIL_Q = " OR ".join(["event_type: {}".format(e) for e in FAIL_REASONS])
SCRUB_TIME_Q = "tags: Disk_Scrub_Complete AND scrub_seconds: *"


def get_broken_block(message_body):
    if regex.match("^\s*Fixing bad parity.*", message_body):
        block_re = regex.compile(".*block #(?P<disk_block>\d+)\s+$")
    else:
        block_re = regex.compile(
            "^Fixing.*, disk block \(DBN\) (?P<disk_block>\d+),.*")

    match = block_re.match(message_body)
    disk_block = None
    if match:
        try:
            disk_block = int(match.group('disk_block'))
        except (ValueError, IndexError):
            print(message_body)
    else:
        print(message_body)
    return disk_block


def get_disk_location(data):
    """
    Try getting the disk location the reasonable way, falling back to
    regex search.
    """
    disk_re = regex.compile(r".*\b\d{1,2}[a-d]\.(?P<disk_location>\d{1,2}\.\d{1,2})(\b|P\d).*")

    try:
        return data['disk_location']
    except KeyError:
        match = disk_re.match(data['body'])
        if match:
            return match.group('disk_location')
        else:
            print("Error getting disk from {}".format(data['body']))
            return None


def get_bad_blocks(es):
    """
    Returns:
    Tuple of (timestamp, cluster, disk_location, broken_block)
    """

    res = scan(es, index=ES_INDEX, q=READERR_REPAIR_Q)

    for x in res:
        data = x['_source']
        timestamp = dateparser.parse(data['@timestamp'])
        body = data["body"]
        disk_location = get_disk_location(data)
        cluster = data['cluster_name']
        broken_block = get_broken_block(body)

        yield (timestamp, cluster, disk_location, broken_block)


def count_bad_blocks(results):
    """
    Returns: cluster => disk => bad block count
    """

    faults = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for ts, cluster, disk, block in results:
        faults[cluster][disk][block].append(ts)

    counter = defaultdict(Counter)
    for cluster, disk_data in faults.items():
        for disk_name, bad_blocks in disk_data.items():
            counter[cluster][disk_name] += len(bad_blocks.keys())
    return counter


def get_broken_disks(es):
    """
    Return a tuple of (timestamp, cluster, disk, failure reason)

    Note that the same disk _will_ be reported many, many times!
    """
    res = scan(es, index=ES_INDEX, q=DISK_FAIL_Q)

    for x in res:
        data = x['_source']
        timestamp = dateparser.parse(data['@timestamp'])
        disk_location = get_disk_location(data)
        cluster = data['cluster_name']

        yield (timestamp, cluster, disk_location, data['event_type'])


def cluster_broken_disks(results):
    """
    Return:
    cluster => disk => first sign of failure
    """

    failure_dates = defaultdict(lambda: defaultdict(lambda:
                                                    datetime.now(pytz.utc)))

    for ts, cluster, disk, _fail_reason in results:
        try:
            previous = failure_dates[cluster][disk]
            failure_dates[cluster][disk] = min(previous, ts)
        except TypeError:
            print(previous, ts)
    return failure_dates


def print_bad_blocks_report(es):
    total_bad_block_count = 0
    disks_with_bad_blocks_count = 0
    clusters_with_bad_blocks_count = 0
    for cluster, disk_counter in count_bad_blocks(get_bad_blocks(es)).items():
        print("Cluster {}: {}".format(cluster, format_counter(disk_counter)))
        total_bad_block_count += sum(disk_counter.values())
        disks_with_bad_blocks_count += len(disk_counter.keys())
        clusters_with_bad_blocks_count += 1

    print("Total: {} bad blocks on {} disks in {} clusters"
          .format(total_bad_block_count,
                  disks_with_bad_blocks_count,
                  clusters_with_bad_blocks_count))


def print_broken_disks_report(es):
    clustered_data = cluster_broken_disks(get_broken_disks(es))
    for cluster, broken_disks in clustered_data.items():
        for disk_name, first_broke in broken_disks.items():
            print("Cluster {}, disk {} first broke at {}"
                  .format(cluster, disk_name, first_broke))


def print_correlation_report(es):
    broken_disks = set()
    disks_with_bad_blocks = set()

    broken_disk_data = cluster_broken_disks(get_broken_disks(es))
    for cluster, cluster_disks in broken_disk_data.items():
        for disk_name in cluster_disks.keys():
            broken_disks.add((cluster, disk_name))

    for cluster, bad_blocks in count_bad_blocks(get_bad_blocks(es)).items():
        for disk_name in bad_blocks.keys():
            disks_with_bad_blocks.add((cluster, disk_name))

    print("Broken disks with bad blocks: ({})"
          .format(broken_disks & disks_with_bad_blocks,
                  len(broken_disks & disks_with_bad_blocks)))
    print("Broken disks without bad blocks: {} ({})"
          .format(broken_disks - disks_with_bad_blocks,
                  len(broken_disks - disks_with_bad_blocks)))
    print("Non-broken disks with bad blocks: {} ({})"
          .format(disks_with_bad_blocks - broken_disks,
                  len(disks_with_bad_blocks - broken_disks)))


def get_scrubbing_durations(es):
    """
    Return scrubbing durations, as a tuple:
    (timestamp, cluster, disk, timedelta of scrubbing duration)
    """

    res = scan(es, index=ES_INDEX, q=SCRUB_TIME_Q,
               _source=["body", "disk_location", "scrub_seconds", "@timestamp",
                        "cluster_name"])

    for r in res:
        data = r['_source']
        timestamp = dateparser.parse(data['@timestamp'])
        disk = get_disk_location(data)
        cluster = data['cluster_name']
        try:
            scrubbing_time = data['scrub_seconds']
        except KeyError:
            continue

        yield (timestamp, cluster, disk, timedelta(seconds=scrubbing_time))


def get_minmax_scrub_durations(scrub_durations):
    min_so_far = timedelta(hours=100*24*365)  #  100 years
    max_so_far = timedelta(seconds=0)

    max_result = None
    min_result = None

    for timestamp, cluster, disk, delta in scrub_durations:
        if delta > max_so_far:
            max_result = (timestamp, cluster, disk, delta)
            max_so_far = delta
        if delta < min_so_far:
            min_result = (timestamp, cluster, disk, delta)
            min_so_far = delta

    return min_result, max_result


def print_scrubbing_report(es):
    (_, _, _, min_delta), (_, _, _, max_delta) = get_minmax_scrub_durations(
        get_scrubbing_durations(es))

    print("Scrubbing durations were in the range of {} to {}."
          .format(str(min_delta), str(max_delta)))


if __name__ == '__main__':
    es = Elasticsearch([ELASTIC_ADDRESS])
    #print_bad_blocks_report(es)
    #print_broken_disks_report(es)
    #print_correlation_report(es)
    print_scrubbing_report(es)
