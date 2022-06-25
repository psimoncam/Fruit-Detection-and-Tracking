# py-motmetrics - Metrics for multiple object tracker (MOT) benchmarking.
# https://github.com/cheind/py-motmetrics/
#
# MIT License
# Copyright (c) 2017-2020 Christoph Heindl, Jack Valmadre and others.
# See LICENSE file for terms.

"""Compute metrics for trackers using MOTChallenge ground-truth data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from collections import OrderedDict
import glob
import logging
import os
from pathlib import Path

import motmetrics as mm


def parse_args():
    """Defines and parses command-line arguments."""
    parser = argparse.ArgumentParser(description="""
Compute metrics for trackers using MOTChallenge ground-truth data.

Files
-----
All file content, ground truth and test files, have to comply with the
format described in

Milan, Anton, et al.
"Mot16: A benchmark for multi-object tracking."
arXiv preprint arXiv:1603.00831 (2016).
https://motchallenge.net/

Structure
---------

Layout for ground truth data
    <GT_ROOT>/<SEQUENCE_1>/gt/gt.txt
    <GT_ROOT>/<SEQUENCE_2>/gt/gt.txt
    ...

Layout for test data
    <TEST_ROOT>/<SEQUENCE_1>.txt
    <TEST_ROOT>/<SEQUENCE_2>.txt
    ...

Sequences of ground truth and test will be matched according to the `<SEQUENCE_X>`
string.""", formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--groundtruths', type=str, help='Directory containing ground truth files.')
    parser.add_argument('--tests', type=str, help='Directory containing tracker result files')
    parser.add_argument('--loglevel', type=str, help='Log level', default='info')
    parser.add_argument('--fmt', type=str, help='Data format', default='mot15-2D')
    parser.add_argument('--solver', type=str, help='LAP solver to use for matching between frames.')
    parser.add_argument('--id_solver', type=str, help='LAP solver to use for ID metrics. Defaults to --solver.')
    parser.add_argument('--exclude_id', dest='exclude_id', default=False, action='store_true',
                        help='Disable ID metrics')
    return parser.parse_args()


def compare_dataframes(gts, ts):
    """Builds accumulator for each sequence."""
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:
            logging.info('Comparing %s...', k)
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
        else:
            logging.warning('No ground truth for %s, skipping.', k)

    return accs, names


def main():
    # pylint: disable=missing-function-docstring
    args = parse_args()

    loglevel = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(loglevel, int):
        raise ValueError('Invalid log level: {} '.format(args.loglevel))
    logging.basicConfig(level=loglevel, format='%(asctime)s %(levelname)s - %(message)s', datefmt='%I:%M:%S')

    if args.solver:
        mm.lap.default_solver = args.solver

    gtfiles = glob.glob(os.path.join(args.groundtruths, '*/gt/gt.txt'))
    tsfiles = [f for f in glob.glob(os.path.join(args.tests, '*.txt')) if not os.path.basename(f).startswith('eval')]

    logging.info('Found %d groundtruths and %d test files.', len(gtfiles), len(tsfiles))
    logging.info('Available LAP solvers %s', str(mm.lap.available_solvers))
    logging.info('Default LAP solver \'%s\'', mm.lap.default_solver)
    logging.info('Loading files.')

    gt = OrderedDict([(Path(f).parts[-3], mm.io.loadtxt(f, fmt=args.fmt, min_confidence=1)) for f in gtfiles])
    ts = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt=args.fmt)) for f in tsfiles])

    mh = mm.metrics.create()
    accs, names = compare_dataframes(gt, ts)
    aux = {}
    for acc, name in zip(accs, names):
        aux[name] = {"switches": acc.switches_dict, "transfer": acc.transfer_dict, "migrate": acc.migrate_dict, "ascend": acc.ascend_dict,"ids_per_track": acc.ids_per_track, "tracks_per_id": acc.tracks_per_id}

    # metrics = list(mm.metrics.motchallenge_metrics)
    metrics = ["idf1", "idp", "idr", "recall", "precision", "num_unique_objects", "mostly_tracked", "partially_tracked", "mostly_lost", "num_false_positives", "num_misses", "num_switches", "num_fragmentations", "mota", "motp", "num_transfer", "num_ascend", "num_migrate"]
    # metrics = ["idf1", "idp", "idr", "recall", "precision", "num_unique_objects", "mostly_tracked", "partially_tracked", "mostly_lost", "num_false_positives", "num_misses", "num_switches", "num_fragmentations", "mota", "motp"]
    if args.exclude_id:
        metrics = [x for x in metrics if not x.startswith('id')]

    logging.info('Running metrics')

    if args.id_solver:
        mm.lap.default_solver = args.id_solver
    summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
    for name, dict_seq in aux.items():
        print("\nInformation of sequence {}".format(name))
        ids_per_track = {}
        tracks_per_id = {}
        switches = {}
        ascend = {}
        transfer = {}
        migrate = {}
        
        for obj_id in dict_seq["switches"].keys():
            if dict_seq["switches"][obj_id] not in switches.keys():
                switches[dict_seq["switches"][obj_id]] = 0
            switches[dict_seq["switches"][obj_id]] += 1
            
        for obj_id in dict_seq["ascend"].keys():
            if dict_seq["ascend"][obj_id] not in ascend.keys():
                ascend[dict_seq["ascend"][obj_id]] = 0
            ascend[dict_seq["ascend"][obj_id]] += 1
            
        for obj_id in dict_seq["transfer"].keys():
            if dict_seq["transfer"][obj_id] not in transfer.keys():
                transfer[dict_seq["transfer"][obj_id]] = 0
            transfer[dict_seq["transfer"][obj_id]] += 1
            
        for obj_id in dict_seq["migrate"].keys():
            if dict_seq["migrate"][obj_id] not in migrate.keys():
                migrate[dict_seq["migrate"][obj_id]] = 0
            migrate[dict_seq["migrate"][obj_id]] += 1
            
        for track_id in dict_seq["ids_per_track"].keys():
            list_ids = dict_seq["ids_per_track"][track_id]
            if len(list_ids) not in ids_per_track.keys():
                ids_per_track[len(list_ids)] = 0
            ids_per_track[len(list_ids)] += 1
            
        for obj_id in dict_seq["tracks_per_id"].keys():
            list_tracks = dict_seq["tracks_per_id"][obj_id]
            if len(list_tracks) not in tracks_per_id.keys():
                tracks_per_id[len(list_tracks)] = 0
            tracks_per_id[len(list_tracks)] += 1
            
        print("\nID switches [a match between a object and hypothesis was found but differs from previous assignment (hypothesisid != previous)]")
        for num_switches, quantity_oids in switches.items():
            print("{} objects with {} ID switches".format(quantity_oids, num_switches))
            
        print("\nID ascends [a match between a object and hypothesis was found but differs from previous assignment  (hypothesisid is new)]")
        for num_ascends, quantity_oids in ascend.items():
            print("{} objects with {} ID ascends".format(quantity_oids, num_ascends))
            
        print("\nID transfers [a match between a object and hypothesis was found but differs from previous assignment (objectid != previous)]")
        for num_transfers, quantity_oids in transfer.items():
            print("{} objects with {} ID transfers".format(quantity_oids, num_transfers))
            
        print("\nID migrations [a match between a object and hypothesis was found but differs from previous assignment  (objectid is new)]")
        for num_migrations, quantity_oids in migrate.items():
            print("{} objects with {} ID migrations".format(quantity_oids, num_migrations))
        
        print("\nIDs per track")
        for num_ids, quantity_tracks in ids_per_track.items():
            print("{} tracks with {} different object ids".format(quantity_tracks, num_ids))
            
        print("\nTracks per object ID")
        for num_tracks, quantity_oids in tracks_per_id.items():
            print("{} objects with {} different tracks".format(quantity_oids, num_tracks))
            
    logging.info('Completed')


if __name__ == '__main__':
    main()