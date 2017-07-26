import argparse
from contextlib import contextmanager
import time
import logging

log = logging.getLogger(__name__)

ELASTIC_ADDRESS = "localhost:9200"

def add_elasticsearch_options(parser, default_address):
    parser.add_argument('--timeout_s', '-t', nargs='?', dest='timeout',
                        default=10,
                        type=int, help="Database operation timeout in seconds")
    parser.add_argument('--es_node', '-s', nargs='*', type=str,
                        dest='es_nodes',
                        help="Elasticsearch node to connect to",
                        default=[default_address])


def set_log_level_from_args(args, logger):
    log_level = (max(3 - args.verbose_count, 0) * 10)
    logger.setLevel(log_level)


def setup_verbosity_flags(parser):
    parser.add_argument('--verbose', '-v', action='count', dest='verbose_count',
                        default=0)

def make_es_base_parser():
    parser = argparse.ArgumentParser(description="")
    setup_verbosity_flags(parser)
    add_elasticsearch_options(parser, ELASTIC_ADDRESS)
    return parser


def add_subcommands(parent, descriptions):
    subparser_master = parent.add_subparsers(title='operations',
                                             help="valid operations")
    added_parsers = [subparser_master]
    for description in descriptions:
        name, hlp, args, func = description
        parser = subparser_master.add_parser(name, help=hlp)
        parser.set_defaults(func=func)
        for arg_set in args:
            fun_args, kwd_args = arg_set
            parser.add_argument(*fun_args, **kwd_args)
        added_parsers.append(parser)
    return added_parsers



def format_counter(counter):
    return ", ".join(["{} ({})".format(k, c) for k, c in
                      sorted(counter.items(),
                             key=lambda tpl: tpl[1],
                             reverse=True)])


@contextmanager
def timed(task_name, time_record=[], printer=log.info):
    start_time = time.clock()
    yield
    end_time = time.clock()
    printer("Task {} ran in {:06.4f}s"
             .format(task_name, end_time - start_time))
    time_record.append(end_time - start_time)
