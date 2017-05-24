#!/usr/bin/env python3

import mailbox
import tempfile
import os.path
import subprocess
import multiprocessing
import time
from contextlib import contextmanager
import logging
import sys
import gzip
from functools import partial
from collections import Counter, defaultdict
import re
import datetime

import lxml.etree
import dateparser
import dateutil

log = logging.getLogger(__name__)
ch = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(name)s %(levelname)s: %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)
log.setLevel(logging.NOTSET)

ASUP_NS = "http://asup_search.netapp.com/ns/ASUP/1.1"
LOG_NS = "http://asup_search.netapp.com/ns/T_EMS_LOCAL_LOG/1.0"

NAMESPACES = {'l': LOG_NS,
              'a': ASUP_NS}

cluster_re = re.compile(".*?db(?P<cluster>.*?)\d+\s+\(.*")
disk_re = re.compile("(?P<connector>[0-9][a-d])\.((?P<shelf>\d{1,2})\.)?(?P<bay>\d{1,2})")
syslog_re = re.compile('(?P<some_code>([0-9]|[a-f])+\.([0-9]|[a-f])+\s([0-9]|[a-f])+)\s(?P<date>.*?)\s+\[(?P<facility>.*):(?P<log_level>.*)\] (?P<body>.*)')


def read_xml_string(element, *element_hierarchy):
    target = "/".join(element_hierarchy)
    return element.findtext(target, namespaces=NAMESPACES)


@contextmanager
def timed(task_name, time_record=[]):
    start_time = time.clock()
    yield
    end_time = time.clock()
    log.info("Task {} ran in {:06.4f}s"
             .format(task_name, end_time - start_time))
    time_record.append(end_time - start_time)


def read_gzipped_log(gzip_fn):
    log = []
    with gzip.open(gzip_fn, 'r') as log_f:
        for line in log_f:
            log.append(line.decode('utf-8'))
    return log


def read_context(xml_fn):
    parameters = {}

    with open(xml_fn, 'r') as f:
        tree = lxml.etree.parse(f)

    data_rows = tree.xpath('/l:T_EMS_LOCAL_LOG/a:ROW', namespaces=NAMESPACES)
    for row in data_rows:
        parameters['time'] = dateparser.parse(read_xml_string(row, 'l:time'))
        parameters['severity'] = read_xml_string(row, 'l:severity')
        parameters['event_type'] = read_xml_string(row, 'l:messagename')

        for parameter_item in row.xpath('l:parameters/a:list/a:li/text()',
                                        namespaces=NAMESPACES):
            k, v = parameter_item.split(":")
            parameters[k] = v

    return parameters


def parse_issue_statement(statement):
    index_re = re.compile("^(?P<index>\d+)\)")
    kv_re = re.compile("^(?P<key>.*)=(?P<value>.*)")

    index = int(index_re.match(statement).group('index'))

    tail = re.split(index_re, statement)[-1]

    data = {'index': index}

    for key_value in re.split("\s*,\s*", tail):
        kv_cleaned = key_value.strip()
        match = kv_re.match(kv_cleaned)
        value = match.group('value')
        key = match.group('key')

        if key == 'timefailed' or key == 'timelastseen':
            timestamp = re.match(r"(\d+)\s", value).group(1)
            value = dateparser.parse(timestamp)

        data[key] = value

    return data


def read_registry(txt_fn):
    with open(txt_fn, 'r') as f:
        completed_statements = []
        current_statement = []
        token_statement_start = re.compile("^\d+\)")

        for line in f:
            # NetApp has fucked up the newlines, because of course they have.
            if not token_statement_start.match(line):
                # We are not at a new statement, just keep reading
                current_statement.append(line.rstrip('\n'))
                continue
            else:
                # We are at a new line with "1) ..."
                if current_statement:
                    # We have a complete statement
                    statement_string = ", ".join(current_statement)
                    completed_statements.append(
                        parse_issue_statement(statement_string))
                current_statement = []
                current_statement.append(line.rstrip('\n'))

        # We reached the end of the file, parse the last line if any
        if current_statement:
            statement_string = ", ".join(current_statement)
            completed_statements.append(
                parse_issue_statement(statement_string))

        return completed_statements

def parse_syslog_msg(msg):
    match = syslog_re.match(msg)
    date_parsed = dateutil.parser.parse(match.group('date'))
    result = {'original': msg,
              'facility': match.group('facility'),
              'log_level': match.group('log_level'),
              'date': date_parsed,
    }
    return result


def read_messages_log(gzip_fn):
    log = read_gzipped_log(gzip_fn)
    return [parse_syslog_msg(msg) for msg in log]


def read_ems_log_file(gzip_fn):
    return read_gzipped_log(gzip_fn)


def sevenz_extract(z_file, cwd):
    p = subprocess.Popen(["7z", "e", z_file],
                         stdout=subprocess.PIPE, cwd=cwd,
                         stderr=subprocess.PIPE)
    outs, errs = p.communicate()
    if p.returncode != 0:
        print(errs.decode("utf-8"))
        raise Exception
    log.debug("Exctracted {} into {}".format(z_file, cwd))


def parse_mail_part(part, mail):
    with tempfile.TemporaryDirectory() as temp_dir:
        wd = partial(os.path.join, temp_dir)

        with open(wd(part.get_filename()), 'wb') as f:
            payload = part.get_payload(decode=True)
            f.write(payload)

        sevenz_extract(wd(part.get_filename()), temp_dir)

        @contextmanager
        def maybe_missing():
            try:
                yield
            except FileNotFoundError as e:
                log.info('Mail "{}" lacks file {}'
                          .format(mail.get('Subject'),
                                  os.path.split(e.filename)[-1]))
                log.debug("Files are: {}."
                          .format(", ".join(os.listdir(wd()))))

        registry = None
        ems_log = None
        syslog = None
        context = None

        with maybe_missing():
            registry = read_registry(wd('FAILED-DISK-REGISTRY.txt'))
        with maybe_missing():
            ems_log = read_ems_log_file(wd('EMS-LOG-FILE.gz'))
        with maybe_missing():
            syslog = read_messages_log(wd('messages.log.gz'))
        with maybe_missing():
            context = read_context(wd('disk-fault-context.xml'))

    if all([x is None for x in [registry, ems_log, syslog, context]]):
        # No data was extracted
        return None

    return {'registry': registry,
            'ems_log': ems_log,
            'syslog': syslog,
            'context': context}


def parse_mail(mail):
    part_data = None
    for part in mail.walk():
        if part.get_content_maintype() == 'multipart':
            continue
        if part.get('Content-Disposition') is None:
            continue
        if is_signature(part):
            continue

        attached_data = parse_mail_part(part, mail)
        if attached_data:
            part_data = attached_data
            break

    if not part_data:
        log.warning('Mail "{}" does not have any attachments'
                    .format(mail['Subject']))
    cluster = cluster_re.match(mail['Subject']).group('cluster').strip()
    if not cluster:
        raise ValueError(cluster_re.match(mail['Subject']))

    return {'subject': mail['Subject'],
            'cluster': cluster,
            'date': dateparser.parse(mail['Date']),
            'parts_data': part_data}


def is_signature(part):
    content_type = part.get('Content-Type').split(";")[0]
    return content_type == 'application/pkcs7-signature'


def format_counter(counter):
    return ", ".join(["{} ({})".format(k, c) for k, c in
                      sorted(counter.items(),
                             key=lambda tpl: tpl[1],
                             reverse=True)])

def analyse_data(results):
    total = len(results)
    partial_data = 0
    no_data = 0
    missing_data = Counter()
    disk_status = Counter()
    cluster_status = Counter()
    dates = []
    syslog_windows = []
    failure_reasons = Counter()
    fault_times = defaultdict(list)

    for result in results:
        dates.append(result['date'])
        if not result['parts_data']:
            no_data += 1
            continue

        part_data = result['parts_data']
        found_missing = False
        for k, v in part_data.items():
            if v is None:
                found_missing = True
                missing_data[k] += 1
        if found_missing:
            partial_data += 1

            if not part_data['registry']:
                # Without this, we cannot do analysis
                continue
        if part_data['context']:
            failure_reasons[part_data['context']['failure_reason']] += 1

        if part_data['syslog']:
            first_log_entry = part_data['syslog'][0]['date']
            last_log_entry = part_data['syslog'][-1]['date']
            delta = last_log_entry - first_log_entry
            syslog_windows.append(delta)

        num_failed_disks = 0
        registry_failed_dates = []
        for disk in part_data['registry']:
            if disk['device'] == "NotPresent":
                continue

            num_failed_disks += 1
            registry_failed_dates.append(disk['timefailed'])

            shelf = disk_re.match(disk['device']).group('shelf')
            bay = disk_re.match(disk['device']).group('bay')

            if not shelf:
                device_name = "{}:{}".format(result['cluster'], bay)
            else:
                device_name = "{}:{}.{}".format(result['cluster'], shelf, bay)
            disk_status[device_name] += 1
            cluster_status[result['cluster']] += 1

        if part_data['context'] and len(registry_failed_dates) <= 1:
            context_fault_date = part_data['context']['time']
            fault_times[result['cluster']].append(context_fault_date)
        elif registry_failed_dates:
            fault_times[result['cluster']] += registry_failed_dates
        else:
            fault_times[result['cluster']].append(result['date'])

    return {'count': total,
            'partial_data': partial_data,
            'no_data': no_data,
            'missing_data': missing_data,
            'disk_status': disk_status,
            'cluster_status': cluster_status,
            'dates': dates,
            'syslog_windows': syslog_windows,
            'failure_reasons': failure_reasons,
            'fault_times': fault_times,
    }

if __name__ == '__main__':
    mbox_file = sys.argv[1]

    print("Parsing {}".format(mbox_file))

    time_records = []
    p = multiprocessing.Pool()

    with timed("Parallel extract", time_records):
        results = list(map(parse_mail, mailbox.mbox(mbox_file)))


    analysis = analyse_data(results)


    print(("Read {} emails, of which {} had a complete data set, {}"
           " contained no data, and {} had partial data. Execution took"
           " {:06.4f}s.")
          .format(analysis['total'], analysis['total'] - analysis['partial_data'] - analysis['no_data'],
                  analysis['no_data'], analysis['partial_data'],
                  time_records[0]))

    print("The following data was missing: {}."
          .format(format_counter(analysis['missing_data'])))
    print("Saw {} disk failures:".format(sum(analysis['disk_status'].values())))
    print("Disk statuses were: {}".format(format_counter(analysis['disk_status'])))
    print("Failures per cluster were: {}".format(format_counter(analysis['cluster_status'])))
    print("Date range was {}--{}".format(min(analysis['dates']), max(analysis['dates'])))
    print("Syslog windows were in the range {}--{}".format(
        min([x for x in analysis['syslog_windows'] if x > datetime.timedelta(0)]),
        max(analysis['syslog_windows'])))
    print("Observed failure reasons (from context) were: {}"
          .format(format_counter(analysis['failure_reasons'])))
