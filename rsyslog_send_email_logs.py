#!/usr/bin/env python3

import sys
import socket
import mailbox
from parse_emails import parse_mail

RSYSLOG_SERVER = "db-51167"
RSYSLOG_PORT = 514

if __name__ == '__main__':
    mbox_file = sys.argv[1]
    emails = map(parse_mail, mailbox.mbox(mbox_file))
    for email in emails:
        if not email['parts_data']:
            continue

        parts = email['parts_data']

        if not parts['syslog']:
            continue

        syslog = parts['syslog']

        with socket.socket(socket.AF_INET,
                           socket.SOCK_STREAM) as s:

            s.connect((RSYSLOG_SERVER, RSYSLOG_PORT))
            for entry in syslog:
                message = ""
                try:
                    s.send(message.encode("utf-8"))
                except Exception as e:
                    print(e)
                    break


# Target: <5>May 23 03:25:02 [dbnash5132:raid.rg.media_scrub.suspended:notice]: /aggr1_rac5132/plex0/rg4: media scrub suspended at stripe 777407872 after 2001:00:21.0

# Source: 0000005b.00c29168 115471b0 Thu Jan 12 2017 00:21:22 +01:00 [callhome.management.log:info] Call home for MANAGEMENT_LOG
