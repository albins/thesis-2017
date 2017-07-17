#!/usr/bin/env python3
from parse_emails import format_counter
from parse_smart_pages import count_disks

from collections import Counter, defaultdict
from datetime import datetime, timedelta
import time
import sys
import logging
import argparse
import statistics
import csv

import pytz
import regex
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Q, A
from elasticsearch.helpers import scan
import dateparser

log = logging.getLogger()

#ELASTIC_ADDRESS = "db-51167.cern.ch:9200"
ELASTIC_ADDRESS = "localhost:9200"
ES_SYSLOG_INDEX = "syslog-*"
ES_LOWLEVEL_BASE = 'netapp-lowlevel'
ES_LOWLEVEL_INDEX = '{}-*'.format(ES_LOWLEVEL_BASE)
READERR_REPAIR_Q = "event_type: raid.rg.readerr.repair.*",
FAIL_REASONS = ["raid.config.filesystem.disk.failed",
                "raid.disk.online.fail",
                "disk.failmsg",
                "disk.partner.diskFail",
                "raid.fdr.fail.disk.sick",
                "raid.fdr.reminder"]
TROUBLE_REASONS =  [
    "disk.write.failure",
    "scsi.cmd.aborted",
    "scsi.cmd.notReadyCondition",
    "disk.senseError",
    "scsi.cmd.aborted",
    "dfu.readTestFailed",
    "dfu.readTestFailed.fatal",
    "disk.ioMediumError",
    "diskown.errorDuringIO",
    "diskown.errorReadingOwnership",
    "raid.read.media.err",
    "raid.rg.diskcopy.aborted",
    "raid.rg.diskcopy.read.err",
    "raig.rg.diskcopy.cant.start",
    "raid.rg.diskcopy.done",
    "raid.disk.timeout.recovery.read.err",
    "raid.rg.scrub.media.recommend.reassign.err",
    "raid.rg.scrub.media.err",
    "raid.disk.timeout.ios.flush.end",
    "raid.disk.online.dirty.ranges",
    "raid.rg.readerr.repair.data",
    "disk.ioRecoveredError.reassign",
    "disk.ioRecoveredError.retry",
    "shm.threshold.ioLatency",
    "fmmb.disk.check.ready.failed",
    "disk.readReservationFailed",
    "disk.ioReassignSuccess",
    "coredump.findcore.read.failed",
    "raid.tetris.media.err",
    "scsi.cmd.notReadyConditionEMSOnly",
    "scsi.cmd.checkCondition",
    "disk.powercycle",
    ]

TROUBLE_FUZZY_KEYWORDS = ["error", "bad", "fault", "fail", "debounce", "powercycle"]
HARD_FAILURE_INDICATORS = ["raid.fdr.reminder",
                           "raid.disk.missing",
                           "raid.disk.unload.done"]

ONE_WEEK = timedelta(hours=7*24)
UTC_NOW =  datetime.fromtimestamp(time.time(), tz=pytz.utc)
TIME_BEFORE_FAILURE = timedelta(hours=10)

Q_TROUBLE_EVENTS = Q("bool", should=[*[Q('match', event_type=t)
                                       for t in TROUBLE_REASONS],
                                     *[Q('fuzzy', body=kw)
                                       for kw in TROUBLE_FUZZY_KEYWORDS]],
                     minimum_should_match=1)

Q_FAIL_EVENTS = Q("bool", should=[Q('match', event_type=t)
                                     for t in FAIL_REASONS],
                     minimum_should_match=1)
Q_JUNK_EVENTS = Q('term', event_type="cf.disk.skipped")

DISK_FAIL_Q = " OR ".join(["event_type: {}".format(e) for e in FAIL_REASONS])
SCRUB_TIME_Q = "tags: Disk_Scrub_Complete AND scrub_seconds: *"
TYPE_TO_NUMBER = {'ssd': 1, 'fsas': 2, 'bsas': 3}
NUMBER_TO_TYPE = {v: k for k, v in TYPE_TO_NUMBER.items()}
SMART_FIELDS = [ # from zhu, 2013
    1,
    3,
    5,
    7,
    9,
    187,
    189,
    194,
    195,
    197,
]
SMART_LENGTH = len(SMART_FIELDS)
SMART_MYSTERY_LENGTH = 220
SMART_RAW_IDX = 3

def flatten(lst):
    return sum(lst, [])

def windowed_query(s, start=None, end=UTC_NOW):
    time_range = {'gte': start} if start else {}
    time_range['lte'] = end

    s = s.sort({"@timestamp": {"order": "asc"}})\
        .filter("range", **{'@timestamp': time_range})
    return s


def windowed_syslog(es, center, width):
    """
    Return a Search for all log events in a window of a given timedelta
    width from the syslog, as centered in center.
    """
    radius = width/2
    start = center - radius
    end = center + radius

    return windowed_query(Search(using=es, index=ES_SYSLOG_INDEX),
                          start=start, end=end)


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
            #print("Error getting disk from {}".format(data['body']))
            return None


def get_bad_blocks(es):
    """
    Returns:
    Tuple of (timestamp, cluster, disk_location, broken_block)
    """

    res = scan(es, index=ES_SYSLOG_INDEX, q=READERR_REPAIR_Q)

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
    Return documents for broken disks

    Note that the same disk _will_ be reported many, many times!

    Sterams are ordered by time, first to last.
    """
    q = Q('bool', must=[Q_FAIL_EVENTS])

    s = Search(using=es, index=ES_SYSLOG_INDEX)\
        .query(q)\
        .sort({"@timestamp": {"order": "asc"}})

    for data_point in s.scan():
        yield deserialise_log_entry(data_point)


def cluster_broken_disks(results):
    """
    Return:
    cluster => disk => first sign of failure
    """

    failure_dates = defaultdict(dict)

    for result in results:
        ts = result["@timestamp"]
        cluster = result['cluster_name']
        disk = result['disk_location']

        if disk in failure_dates[cluster]:
            continue
        else:
            failure_dates[cluster][disk] = ts
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
    disk_count = count_disks(es)

    broken_disk_data = cluster_broken_disks(get_broken_disks(es))
    for cluster, cluster_disks in broken_disk_data.items():
        for disk_name in cluster_disks.keys():
            broken_disks.add((cluster, disk_name))

    for cluster, bad_blocks in count_bad_blocks(get_bad_blocks(es)).items():
        for disk_name in bad_blocks.keys():
            disks_with_bad_blocks.add((cluster, disk_name))

    print("Broken disks with bad blocks: {}"
          .format(len(broken_disks & disks_with_bad_blocks)))
    print("Broken disks without bad blocks: {}"
          .format(len(broken_disks - disks_with_bad_blocks)))
    print("Non-broken disks with bad blocks: {}"
          .format(len(disks_with_bad_blocks - broken_disks)))

    print("P(broken disk) = {:3.2f}%"
          .format(len(broken_disks)*100/disk_count))
    print("P(bad blocks) = {:3.2f}%"
          .format(len(disks_with_bad_blocks)*100/disk_count))
    print("P(bad block| broken disk) = {:3.2f}%"
          .format(len(broken_disks & disks_with_bad_blocks)*100/len(broken_disks)))
    print("P(broken disk| bad blocks) = {:3.2f}%"
          .format(len(broken_disks & disks_with_bad_blocks)*100/len(disks_with_bad_blocks)))


def get_scrubbing_durations(es):
    """
    Return scrubbing durations, as a tuple:
    (timestamp, cluster, disk, timedelta of scrubbing duration)
    """

    res = scan(es, index=ES_SYSLOG_INDEX, q=SCRUB_TIME_Q,
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


def get_overview_data(es, cluster_name, disk_name):
    failure_message = None
    false_positives = []
    for failure_doc in failed(es, cluster_name, disk_name):
        is_fp = was_false_positive(es, failure_doc)
        if not is_fp:
            failure_message = failure_doc
            break
        else:
            false_positives.append(failure_doc)

    if failure_message:
        time_of_death = failure_message['@timestamp']
        failure_status = ("The disk failed at {} with fault {}"
                          .format(str(time_of_death),
                                  failure_message['event_type']))
        first_pre_failure_message = next(troubles(es, cluster_name, disk_name,
                                                  before=time_of_death), None)

        troubles_started = first_pre_failure_message['@timestamp'] \
                           if first_pre_failure_message else time_of_death

        last_non_trouble_log= next(non_troubles(es, cluster_name, disk_name,
                                                before=troubles_started), None)
        if last_non_trouble_log:
            last_non_trouble = "{}: {}".format(str(last_non_trouble_log['@timestamp']),
                                               last_non_trouble_log['event_type'])
        else:
            last_non_trouble = None

        bad_blocks = get_disk_bad_blocks(es, cluster_name, disk_name)
        if bad_blocks:
            bad_blocks_status = ", ".join([str(x) for x in sorted(bad_blocks)])
        else:
            bad_blocks_status = None

        pred_msg = was_predicted(es, failure_message)

        prediction_status = "not predicted" if not pred_msg else \
                            "successful {} at {}".format(
                                pred_msg['event_type'],
                                str(pred_msg['@timestamp']))

    else:
        first_pre_failure_message = next(troubles(es, cluster_name, disk_name), None)
        failure_status = "The disk has not failed (yet)"
        last_non_trouble = None
        prediction_status = None if not false_positives else ", ".join(
            ["false positive {} at {}".format(fp['event_type'], fp['@timestamp'])
             for fp in false_positives])
        bad_blocks_status = None

    if first_pre_failure_message:
        pre_failure_sign_status = ("{}: {}"
                                   .format(str(first_pre_failure_message['@timestamp']),
                                           first_pre_failure_message['event_type']))

    else:
        pre_failure_sign_status = "No pre-failure signs"


    return {
        'disk_location': disk_name,
        'cluster_name': cluster_name,
        'failure_status': failure_status,
        'pre_failure_sign_status': pre_failure_sign_status,
        'last_non_trouble_log': last_non_trouble,
        'bad_blocks_status': bad_blocks_status,
        'prediction_status': prediction_status,
        'smart_history': get_smart_history(cluster_name, disk_name),
        'disk_type': get_disk_type(es, cluster_name, disk_name),
    }


def get_smart_history(cluster, disk):
    """
    Return a compressed list of the SMART history for a given disk, or
    None if the disk had no SMART history.

    Return:
    A dictionary on the form:
    [field] => [(timestamp, value)]
    """
    pass


def get_disk_type(es, cluster, disk):
    s = Search(using=es, index=ES_LOWLEVEL_INDEX)\
        .sort({"@timestamp": {"order": "desc"}})

    s = filter_by_cluster_disk(s, cluster, disk)[0]
    result = next(s.scan())
    return result['type']


def get_disk_bad_blocks(es, cluster_name, disk_name, start=None, end=UTC_NOW):
    """
    Return a set of (bad_block_no) for a given disk.
    """
    s = Search(using=es, index=ES_SYSLOG_INDEX)\
        .query(Q('wildcard', event_type="raid.rg.readerr.repair.*"))

    s = filter_by_cluster_disk(windowed_query(s, start=start, end=end),
                               cluster_name=cluster_name,
                               disk_location=disk_name)

    bad_blocks = set()
    for msg in s.scan():
        log.debug("Got broken block for %s %s",
                  cluster_name, disk_name)
        body = msg.to_dict()["body"]
        broken_block = get_broken_block(body)
        bad_blocks.add(broken_block)

    return bad_blocks


def print_disk_report(data):
    report_str = """Report for disk at {disk_location} on cluster {cluster_name}

    Disk type: {disk_type}
    Failure status: {failure_status}
    Last non-trouble log entry: {last_non_trouble_log}
    First pre-failure sign: {pre_failure_sign_status}
    Bad blocks: {bad_blocks_status}
    NetApp prediction status: {prediction_status}
    """

    print(report_str.format(**data))


def filter_by_cluster_disk(s, cluster_name, disk_location):
    """
    Take a query and return the same query, only filtering on cluster
    name and disk location.
    """
    return s.filter("term", cluster_name=cluster_name)\
            .filter("query_string", query=disk_location)


def filter_junk_events(s):
    return s.filter(Q('bool', must_not=[Q_JUNK_EVENTS]))


def failed(es, cluster_name, disk_name):
    """
    Return a list of failure messages, first to last, for a given disk
    location.
    """

    q = Q('bool', must=[Q_FAIL_EVENTS])

    s = Search(using=es, index=ES_SYSLOG_INDEX)\
        .query(q)\
        .sort({"@timestamp": {"order": "asc"}})

    s = filter_by_cluster_disk(s, cluster_name, disk_name)

    for data_point in s:
        yield deserialise_log_entry(data_point)


def was_resurrection(doc):
    """
    Return True if the document in question represents a resurrected
    disk.
    """
    return doc['event_type'] in ["dbm.pitstop.complete",
                                 "disk.dhm.scan.success"]


def was_false_positive(es, failure_document):
    """
    Return False if the given document was a true failure, otherwise
    return True, or a document proving that it was a false positive.

    False positives are indicated by the disk being resurrected as a
    spare after entering maintenance mode.

    Look for raid.disk.maint.done indicating that the disk passed
    maintenance mode.

    raid.fdr.reminder, raid.disk.missing, raid.rg.recons.done,
    raid.disk.unload.done all signify an actually failed drive, so if
    any of those comes up -- immediately give up and declare failure.
    """
    time_of_death = failure_document['@timestamp']

    s = filter_by_cluster_disk(
        windowed_syslog(es, center=time_of_death, width=ONE_WEEK),
        cluster_name=failure_document['cluster_name'],
        disk_location=failure_document['disk_location'])\
        .filter(Q('bool', must_not=[Q_JUNK_EVENTS]))

    for log_event in (deserialise_log_entry(l) for l in s.scan()):
        event_type = log_event['event_type']

        if event_type in HARD_FAILURE_INDICATORS:
            return False
        elif event_type == 'raid.disk.maint.failed':
            # Apparently the disk failed its maintenance testing
            return False
        elif was_resurrection(log_event):
            return log_event
    return True


def was_predicted(es, failure_document):
    """
    Return the first document proving that the failure was predicted
    by the system, or False if it wasn't.

    A failure is considered predicted if the disk was successfully
    copied before the failure without incidents, and mispredicted
    otherwise, including if the copy process was started but failed.

    Look for raid.disk.predictiveFailure to indicate the prediction,
    raid.disk.maint.start to indicate the disk entering maintenance
    mode.

    See raid.rg.diskcopy.done without raid.rg.diskcopy.aborted or
    read.err within a reasonable timeframe.

    raid.disk.maint.failed is a surefire sign that the disk *was*
    successfully predicted.
    """
    time_of_death = failure_document['@timestamp']
    in_maintenance_mode = False
    copy_started = False
    prefail_msg = None

    s = filter_junk_events(
        filter_by_cluster_disk(
        windowed_syslog(es, center=time_of_death, width=ONE_WEEK),
        cluster_name=failure_document['cluster_name'],
        disk_location=failure_document['disk_location']))

    for log_event in (deserialise_log_entry(l) for l in s.scan()):
        event_type = log_event['event_type']
        log.debug("Saw event type {}".format(event_type))
        ts = log_event['@timestamp']

        if event_type == 'raid.disk.maint.start':
            log.info("Entered maintenance mode!")
            in_maintenance_mode = True
        elif event_type == 'raid.disk.maint.failed':
            # The disk just failed maintenance testing, which should
            # only happen after successful copy.
            log.info("Maintenance mode failed, means we copied successfully.")
            return log_event
        elif event_type == 'raid.disk.maint.done':
            log.info("Maintenance mode done -- misprediction!")
            # It was a prediction -- but a misprediction
            return log_event
        elif event_type == 'raid.disk.predictiveFailure':
            if time_of_death - ts > timedelta(hours=2):
                # Don't count prefails very close in time to the actual
                # failure
                prefail_msg = log_event
        elif event_type == 'raid.rg.diskcopy.done':
            return log_event
        elif event_type == 'raid.rg.diskcopy.read.err':
            return False
        elif event_type == 'raid.rg.diskcopy.aborted':
            # This might be a bad idea -- I don't know if it can be
            # restarted?
            return False


    return prefail_msg if prefail_msg else False


def deserialise_log_entry(entry):
    log_entry = entry.to_dict()
    log_entry['@timestamp'] = dateparser.parse(log_entry['@timestamp'])
    log_entry['disk_location'] = get_disk_location(entry)
    return log_entry


def troubles(es, cluster_name, disk_name, before="now", after=None):
    """
    Return all trouble-related log entries before a given date (or now,
    if there was no such date)
    """
    q_cluster_disk = Q('bool', must=[Q_TROUBLE_EVENTS])

    s = Search(using=es, index=ES_SYSLOG_INDEX)\
        .query(q_cluster_disk)

    s = windowed_query(filter_by_cluster_disk(s, cluster_name, disk_name),
                       start=after, end=before)

    for r in s:
        yield deserialise_log_entry(r)


def prefail_false_positives(es):
    pass


def prefail_false_negatives(es):
    pass


def print_prefail_performance_report(es):
    pass


def non_troubles(es, cluster_name, disk_name, before="now"):
    """
    Ordered newest to oldest!
    """
    q_cluster_disk = Q('bool', must=[Q('query_string', query=disk_name),
                                     Q('bool', must_not=[Q_TROUBLE_EVENTS,
                                                         Q_FAIL_EVENTS])])

    s = Search(using=es, index=ES_SYSLOG_INDEX)\
        .query(q_cluster_disk)\
        .sort({"@timestamp": {"order": "desc"}})\
        .filter("term", cluster_name=cluster_name)\
        .filter("range", **{'@timestamp': {'lt': before}})

    for r in filter_junk_events(s).scan():
        yield deserialise_log_entry(r)


def print_prediction_stats(es):

    # identify failed drives
    # for each such drive, determine if it was failed
    # determine if the drive was predicted

    # identify mispredictions

    # Prediction count: count of predicted drives
    # Failure count: number of failed drives
    # True positive rate: number of correctly predicted drives/number of failed drives
    # False positive rate: count of predicted / count of all predictions
    pass


def bucket_broken_disks(failure_messages, window_width):
    """
    return a mapping of cluster name -> disk name -> [t0, t1, ... tn]
    for times the disk broke.
    """
    # cluster -> disk -> [t0, t1, ... tn]
    broke_at = defaultdict(lambda: defaultdict(list))
    for msg in failure_messages:
        cluster = msg['cluster_name']
        disk = msg['disk_location']
        ts = msg['@timestamp']

        # it's the first occurrence in this location
        if not disk in broke_at[cluster]:
            broke_at[cluster][disk].append(ts)
        else:
            time_since_last_failure = min([abs(ts - recorded_time)
                                           for recorded_time in
                                           broke_at[cluster][disk]])
            if time_since_last_failure > ONE_WEEK:
                # If it's been one WEEK since we last observed the disk
                # breaking, presume it's a new fault.
                broke_at[cluster][disk].append(ts)
                log.info("It's been a week -- presume a new failure for %s %s",
                         cluster, disk)
            else:
                log.debug("Discarding recent failure for %s %s: %s",
                          cluster, disk, time_since_last_failure)
    return broke_at


def get_disks(es):
    broke_at = bucket_broken_disks(get_broken_disks(es),
                                   window_width=ONE_WEEK)

    ok_index = "{}-2017-06-21".format(ES_LOWLEVEL_BASE)
    search_start = datetime(2017, 6, 21, tzinfo=pytz.utc)
    search_end = search_start + timedelta(minutes=20)


    s = Search(using=es, index=ok_index)\
        .filter("range", **{'@timestamp':
                            {'gte': search_start,
                             'lte': search_end}})

    cluster_disks = defaultdict(dict)
    found_disks = 0
    disk_count = count_disks(es)

    for res in s.scan():
        if found_disks >= disk_count:
            log.info("Found data for all disks -- bailing out!")
            break
        data = res.to_dict()
        cluster = data['cluster_name']
        disk = data['disk_location']
        c_broken_count = len(broke_at.get(cluster, {}))

        if not disk in cluster_disks[cluster]:
            log.debug("Found new data for %s %s. Got %d/%d entries so far",
                      cluster, disk, found_disks, disk_count)
            broke_data = broke_at[cluster][disk] if disk in broke_at[cluster]\
                         else None
            disk_type = data['type']
            cluster_disks[cluster][disk] = {'cluster_name': cluster,
                                            'disk_location': disk,
                                            'broke_at': broke_data,
                                            'type': normalise_type(disk_type),
                                            'c_broken_count': c_broken_count}
            found_disks += 1
        else:
            log.debug("Ignoring data for %s %s. Got %d entries so far",
                      cluster, disk, found_disks)

    for cluster_data in cluster_disks.values():
        for disk_data in cluster_data.values():
            yield disk_data


def get_read_error_count(es, cluster, disk, at):
    """
    Count the number of reported read errors in the logs for a given
    disk before at.
    """
    s = Search(using=es, index=ES_SYSLOG_INDEX)\
        .query(Q('term', event_type="raid.read.media.err"))

    s = filter_by_cluster_disk(windowed_query(s, end=at),
                               cluster_name=cluster,
                               disk_location=disk)

    s.params(search_type="count")

    return s.execute().to_dict()['hits']['total']


def get_ll_data(es, cluster, disk, at):
    """
    Return the closest match for low-level data for a given disk near a
    date at.
    """
    log.debug("Getting low-level data for %s %s close to %s",
             cluster, disk, str(at))
    s = Search(using=es, index=ES_LOWLEVEL_INDEX)
    s = filter_by_cluster_disk(s, cluster_name=cluster,
                               disk_location=disk)
    s.filter("range", **{"@timestamp": {'lte': at}})\
     .sort({"@timestamp": {"order": "desc"}})

    # Only get the first result
    s = s[0]

    for r in s.scan():
        return r.to_dict()
    raise ValueError


def window_disk_data(es, cluster, disk, start=None, end=UTC_NOW):
    # FIXME: do this async!
    bad_blocks = get_disk_bad_blocks(es, cluster, disk, start, end)
    try:
        bad_block_stdev = statistics.stdev(bad_blocks)
    except statistics.StatisticsError:
        bad_block_stdev = -1

    trouble_count = len(list(troubles(es, cluster, disk,
                                      before=end, after=start)))

    read_errors = get_read_error_count(es, cluster, disk, end)

    at = start if start else end
    disk_data = get_ll_data(es, cluster, disk, at=at)
    if disk_data['smart']:
        log.info("Disk had smart values for %s",
                 ", ".join(sorted(disk_data['smart'].keys())))
        smart_values = [disk_data['smart'].get(
            str(field), [-1]*SMART_RAW_IDX)[SMART_RAW_IDX] for
                        field in SMART_FIELDS]
    else:
        smart_values = [-1] * SMART_LENGTH
    smart_mystery = flatten(disk_data['smart_mystery']) if \
                    disk_data['smart_mystery'] \
                    else [-1] * SMART_MYSTERY_LENGTH
    io_completions = disk_data['io_completions']
    io_completion_times = disk_data['io_completion_times']
    sense_1_5 = [disk_data['sense_data{}'.format(x)] for x in range(1,6)]



    # Sanity-checks:
    assert len(io_completions) == 5, \
        "io_completions had length {}, should be 5".format(len(io_completions))
    #assert len(io_completion_times) == 16 # Apparently broken. Investigate.
    assert len(smart_mystery) == SMART_MYSTERY_LENGTH, \
        "smart mystery had length {}, vals {}, should be {}".format(
            len(smart_mystery),
            ", ".join([str(x) for x in smart_mystery]),
            SMART_MYSTERY_LENGTH)
    assert len(smart_values) == SMART_LENGTH, \
        "SMART values had length {}, expected {}".format(
            len(smart_values), SMART_LENGTH)
    assert isinstance(trouble_count, int), \
        "trouble_count should be int, was {}".format(type(trouble_count))
    assert isinstance(read_errors, int), \
        "read_errors should be int, was {}".format(type(read_errors))

    #   - min, max, delta for:
    #     - retry_count (lldata)
    #     - timeout_count (lldata)
    return [len(bad_blocks),
            bad_block_stdev,
            trouble_count,
            read_errors,
            *smart_values,
            *smart_mystery,
            *io_completions,
            #*io_completion_times,
            *sense_1_5,
            disk_data['sense_data9'],
            disk_data['sense_dataB'],
            disk_data['avg_io'],
            disk_data['max_io'],
            disk_data['retry_count'],
            disk_data['timeout_count'],
    ]


def normalise_type(disk_type_str):
    """
    Take a disk type string, as reported by the database, and transform
    it to a number 1-3. -1 for no match.
    """
    try:
        cleaned_type = (str(disk_type_str)).strip().lower()
        return TYPE_TO_NUMBER[cleaned_type]
    except IndexError:
        log.error("No such disk type '%s'!", disk_type_str)
        return -1


def prepare_training_data(es):
    # for each cluster:
    # - count of broken disks in cluster
    #
    # for each disk:
    # - broken 1/0
    # - type of disk (normalise to int)
    # - for each window (12h, 48h, week, all time):
    #   - read error count for drive
    #   - number of bad blocks for drive
    #   - standard deviation of bad block numbers
    #   - smart (values to columns)
    #   - smart_mystery (values to columns)
    #   - count of pre-fail messages, not ready counts etc
    #   - min, max, delta for:
    #     - number of IO completions (len 5)
    #     - io_completion_times (len 16)
    #     - retry_count
    #     - avg_io
    #     - max_io
    #     - timeout_count
    #     - sense_data1-9+B
    disks = get_disks(es)

    for disk in disks:
        is_broken = 0 if disk['broke_at'] is None else 1
        disk_label = disk['disk_location']
        cluster = disk['cluster_name']

        # Fixme: handle disks failing twice!
        if disk['broke_at']:
            first_breakage = min(disk['broke_at'])
            window_end = first_breakage - TIME_BEFORE_FAILURE
        else:
            window_end = UTC_NOW

        window_start_dates = [window_end - dur for dur in
                              [timedelta(hours=12),
                               timedelta(hours=48), ONE_WEEK]]
        log.debug("Window start dates are: %s",
                  ", ".join([str(x) for x in window_start_dates]))


        try:
            twelve_h, fourtyeight_h, week = map(
                lambda t: window_disk_data(es, cluster, disk_label, start=t,
                                           end=window_end), window_start_dates)

            all_time = window_disk_data(es, cluster, disk_label)
        except Exception as e:
            log.error("Error %s rendering time windows for disk %s %s. Skipping it!",
                      str(e), cluster, disk_label)
            continue

        yield [is_broken, disk['type'],
               disk['c_broken_count'],
               *twelve_h, *fourtyeight_h, *week, *all_time]


def make_report(es, args):
    includes = set(args.include)
    if "scrub" in includes:
        print_scrubbing_report(es)
    if "bad_blocks" in includes:
        print_bad_blocks_report(es)
    if "broken_disks" in includes:
        print_broken_disks_report(es)
    if  "correlation" in includes:
        print_correlation_report(es)
    if "prediction_stats" in includes:
        print_prediction_stats(es)


def make_training_data(es, args):
    window_sizes = ["12h", "48h", "1w", "forever"]
    values_in_window = ["bad_block_count",
                        "bad_block_stdev",
                        "trouble_log_count",
                        "read_error_count",
                        *["smart_raw_%d" % f for f in SMART_FIELDS],
                        *["smart_mystery_%d" % i for i in
                          range(1,SMART_MYSTERY_LENGTH+1)],
                        "cpio_blocks_read",
                        "blocks_read",
                        "blocks_written",
                        "verifies",
                        "max_q",
                        *["sense_%d" % i for i in range (1,6)],
                        "sense_9",
                        "sense_b",
                        "avg_io",
                        "max_io",
                        "retry_count",
                        "timeout_count",
    ]
    window_field_names = flatten([["{}_{}".format(w, heading) for heading in
                                   values_in_window] for w in window_sizes])
    fieldnames = ['is_broken', 'cluster_broken_count',
                  'disk_type', *window_field_names]
    log.info("Using field names %s",
             ", ".join(fieldnames))
    with args.writefile as csvfile:
        writer = csv.writer(csvfile,
                                delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(fieldnames)
        for row in prepare_training_data(es):
            writer.writerow(row)


def make_disk_report(es, args):
    cluster = args.cluster.lower()
    disk_name = args.disk_location
    print_disk_report(get_overview_data(es, cluster, disk_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--verbose', '-v', action='count', dest='verbose_count',
                        default=0)
    parser.add_argument('--timeout_s', '-t', nargs='?', dest='timeout',
                        default=10,
                        type=int, help="Database operation timeout in seconds")
    parser.add_argument('--es_node', '-s', nargs='*', type=str,
                        dest='es_nodes',
                        help="Elasticsearch node to connect to",
                        default=[ELASTIC_ADDRESS])

    subparsers = parser.add_subparsers(title='operations',
                                       help="valid operations")
    parser_report = subparsers.add_parser('report',
                                          help="Generate reports")

    parser_report.add_argument('include', type=str, nargs='+',
                               choices=["scrub", "bad_blocks",
                                        "broken_disks", "correlation",
                                        "prediction_stats"])

    parser_report.set_defaults(func=make_report)

    parser_training = subparsers.add_parser('training',
                                            help="Generate training data")
    parser_training.add_argument('-w','--writefile', type=argparse.FileType('w'), default='-')

    parser_training.set_defaults(func=make_training_data)

    parser_disk_report = subparsers.add_parser('disk',
                                               help="Show history for a given disk")
    parser_disk_report.add_argument('cluster', type=str)
    parser_disk_report.add_argument('disk_location', type=str)
    parser_disk_report.set_defaults(func=make_disk_report)

    args = parser.parse_args()
    log_level = (max(3 - args.verbose_count, 0) * 10)
    log.setLevel(log_level)
    es_conn = Elasticsearch(args.es_nodes, timeout=args.timeout)
    args.func(es=es_conn, args=args)
