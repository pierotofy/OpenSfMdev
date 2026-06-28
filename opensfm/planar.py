# pyre-strict
"""Homography helpers for the planar reconstruction algorithm.

These functions are deliberately kept in a *leaf* module that imports only
numpy, OpenCV, networkx, ``random`` and ``opensfm.log``. ``_compute_planar_homography``
is dispatched to ``loky`` worker processes by ``reconstruction.planar_reconstruction``;
a worker unpickling that task imports this module by name. Keeping it free of the
heavy ``opensfm.reconstruction`` import graph avoids the
``reconstruction_helpers`` <-> ``rig`` circular import that a fresh import of
``opensfm.reconstruction`` triggers in a worker interpreter.
"""

import logging
import random
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import networkx as nx
import numpy as np
from numpy.typing import NDArray
from opensfm import log

logger: logging.Logger = logging.getLogger(__name__)


def find_planar_homography(
    common_tracks_index: Dict[Tuple[str, str], Tuple[int, int, int]],
    common_tracks_data: NDArray,
    pair: Tuple[str, str],
    graph: nx.DiGraph,
    error_threshold: float,
    ransac_error_threshold: float,
) -> Tuple[Optional[NDArray], Optional[Set[float]]]:
    """Estimate a plane-inducing homography for an image pair.

    Based on TRASAC: https://people.eecs.berkeley.edu/~yima/psfile/Planar-CVPR12.pdf

    Samples affine homographies for ``pair`` and scores them by inliers across
    the pair and its (up to two) most-connected adjacent pairs, returning the
    best homography together with the set of inlier track ids for ``pair``.
    """
    log.setup()
    num_trials = 15
    max_iters = 1000
    trials: List[Dict[str, Any]] = []

    num_tracks, rec_start, rec_end = common_tracks_index[pair]
    ct_frame = common_tracks_data[rec_start:rec_end].reshape((num_tracks, 8))
    pair_tracks, p1, p2, hp1, hp2 = (
        ct_frame[:, :1].T[0],
        ct_frame[:, 1:3],
        ct_frame[:, 4:6],
        ct_frame[:, 1:4],
        ct_frame[:, 4:7],
    )

    adjacent_pairs: List[Tuple[Tuple[str, str], int]] = []
    for p in pair:
        for n in graph[p]:
            for ap in ((n, p), (p, n)):
                if ap in common_tracks_index and ap != pair:
                    adjacent_pairs.append((ap, common_tracks_index[ap][0]))

    adjacent_pairs = sorted(adjacent_pairs, key=lambda e: e[1], reverse=True)
    adjacent_pairs = adjacent_pairs[:2]

    i = 0
    while len(trials) < num_trials and i < max_iters:
        i += 1

        r: Dict[str, Any] = {
            "H": None,
            "inliers": set(),
            "pair_inliers": set(),
        }

        # Sample 4
        sample_ids = random.sample(range(0, num_tracks), 4)

        # Compute homography
        track_points1 = p1.take(sample_ids, axis=0)
        track_points2 = p2.take(sample_ids, axis=0)

        r["H"], _ = cv2.estimateAffinePartial2D(track_points1, track_points2)
        r["H"] = np.vstack([r["H"], [0, 0, 1]])

        # Classify each track
        for j in range(num_tracks):
            err = np.linalg.norm(hp2[j] - r["H"].dot(hp1[j]))
            if err < error_threshold:
                r["pair_inliers"].add(pair_tracks[j])

        if len(r["pair_inliers"]) <= 4:
            continue

        r["inliers"] = set(r["pair_inliers"])

        for adj_pair, _ in adjacent_pairs:
            adj_num_tracks, adj_rec_start, adj_rec_end = common_tracks_index[adj_pair]
            adj_ct_frame = common_tracks_data[adj_rec_start:adj_rec_end].reshape(
                (adj_num_tracks, 8)
            )
            adj_pair_tracks, new_p1, new_p2, new_hp1, new_hp2 = (
                adj_ct_frame[:, :1].T[0],
                adj_ct_frame[:, 1:3],
                adj_ct_frame[:, 4:6],
                adj_ct_frame[:, 1:4],
                adj_ct_frame[:, 4:7],
            )

            inliers_common_tracks = set(adj_pair_tracks).intersection(
                r["pair_inliers"]
            )
            if len(inliers_common_tracks) <= 4:
                continue

            inliers_p1: List[NDArray] = []
            inliers_p2: List[NDArray] = []
            for x in range(len(adj_pair_tracks)):
                if adj_pair_tracks[x] in inliers_common_tracks:
                    inliers_p1.append(new_p1[x])
                    inliers_p2.append(new_p2[x])
            inliers_p1_arr = np.reshape(inliers_p1, (len(inliers_p1), 2))
            inliers_p2_arr = np.reshape(inliers_p2, (len(inliers_p2), 2))

            H, _ = cv2.estimateAffinePartial2D(inliers_p1_arr, inliers_p2_arr)

            if H is None:
                continue

            H = np.vstack([H, [0, 0, 1]])

            for j in range(adj_num_tracks):
                err = np.linalg.norm(new_hp2[j] - H.dot(new_hp1[j]))
                if err < error_threshold:
                    r["inliers"].add(adj_pair_tracks[j])

        trials.append(r)

    max_inliers = -1
    best_t: Optional[Dict[str, Any]] = None
    for trial in trials:
        num_inliers = len(trial["inliers"])
        if num_inliers > max_inliers:
            best_t = trial
            max_inliers = num_inliers

    if best_t is None:
        return None, None

    return best_t["H"], best_t["pair_inliers"]


def Rt_from_H(H: NDArray, K: NDArray, K1: NDArray) -> Tuple[NDArray, NDArray]:
    """Recover a (R, t) plane-induced motion from an homography.

    ``K`` is the camera matrix and ``K1`` its inverse.
    """
    H = H.copy()

    _, s, _ = np.linalg.svd(K1.dot(H).dot(K))
    H /= s[1]

    R = H.copy()
    R[0][2] = R[1][2] = 0
    R[2][2] = 1.0

    t = H[:, 2]
    t[0] /= K[0][0]
    t[1] /= K[1][1]
    t[2] = 0

    return R, t


def _compute_planar_homography(
    args: Tuple[
        Tuple[str, str],
        Dict[Tuple[str, str], Tuple[int, int, int]],
        NDArray,
        nx.DiGraph,
    ],
) -> Tuple[Tuple[str, str], Optional[NDArray], Optional[NDArray]]:
    error_threshold = 0.002
    ransac_error_threshold = 0.004

    pair, common_tracks_index, common_tracks_data, graph = args
    num_tracks, _, _ = common_tracks_index[pair]

    H, plane_inliers = find_planar_homography(
        common_tracks_index,
        common_tracks_data,
        pair,
        graph,
        error_threshold,
        ransac_error_threshold,
    )
    if H is None or plane_inliers is None:
        logger.warning("Could not compute homography for %s" % str(pair))
        return (pair, None, None)

    num_outliers = num_tracks - len(plane_inliers)
    logger.info(
        "%s <=> %s inliers: %s outliers: %s"
        % (pair[0], pair[1], len(plane_inliers), num_outliers)
    )

    return (pair, H, np.linalg.inv(H))
