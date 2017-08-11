#!/usr/bin/env python3
import json
import time
import argparse
from datetime import timedelta

import common

import elasticsearch
from elasticsearch.helpers import streaming_bulk
import daiquiri

SYSLOG_INDEX_BASE = "itdb_netapp-syslog"

log = daiquiri.getLogger()


def get_es_dump(f):
    for line in f:
        if not line.strip():
            continue

        data = json.loads(line)
        if not data:
            continue

        del data['_score']
        date_part = "-".join(data['_index'].split("-")[1:])
        new_index = "{}-{}".format(SYSLOG_INDEX_BASE,
                                   date_part)
        data['_index'] = new_index
        print(data)
        yield data


if __name__ == '__main__':
    parser = common.make_es_base_parser()
    parser.add_argument('readfile',
                        type=argparse.FileType('r'))

    args = parser.parse_args()
    daiquiri.setup()
    common.set_log_level_from_args(args, log)
    es = common.es_conn_from_args(args)


    with args.readfile as json_file:
        line_count = sum(1 for line in json_file)
        json_file.seek(0)

        documents = get_es_dump(json_file)
        start_time = time.time()

        for i, result in enumerate(streaming_bulk(es, documents,
                                                  raise_on_error=False)):
            if i > 1:
                seconds_past = time.time() - start_time
                avg_rate = seconds_past / (i-1)
            else:
                avg_rate = 0
            lines_left = line_count - i
            est_seconds_left = lines_left * avg_rate

            succeeded, description = result
            log.info("Processed %d/%d documents (%d%% done). Estimated time left: %s",
                     i, line_count, i/line_count * 100,
                     timedelta(seconds=est_seconds_left))

            if not succeeded:
                log.error(description)
            else:
                log.debug(description)
