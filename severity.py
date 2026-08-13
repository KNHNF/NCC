"""Void severity scoring for the NCC composites defect challenge.

NCC gave two scoring scripts that disagree with each other, and we were told
both run against submissions:

  GitHub evaluation.py:
    length      = straight-line distance between the two farthest pixels
    severity    = length + 0.5 * sqrt(area)
    threshold   = 25

  Drive score_submission.py:
    length      = geodesic, longest path staying inside the void's own pixels
    severity    = length * 0.5 * sqrt(area)
    threshold   = 60 microns

Both formulas are implemented here, verified against the two official scripts
in test_severity.py before either is trusted. Do not add a third "improved"
formula, the point is matching NCC's scoring exactly, not inventing a better one.
"""

import numpy as np
from scipy.spatial.distance import cdist
from skimage.measure import label

VOID = 2
MERGE_DISTANCE_UM = 40.0

STRAIGHT_LINE_THRESHOLD = 25
GEODESIC_THRESHOLD_UM = 60.0


def _straight_line_length_um(coords, um_per_px):
    """evaluation.py's definition: farthest pair of pixels, straight line."""
    if len(coords) < 2:
        return 0.0
    d = cdist(coords, coords)
    return float(d.max()) * um_per_px


def _geodesic_length_um(mask_bool, um_per_px):
    """score_submission.py's definition: longest in-shape path, via double-BFS.

    Double-BFS tree-diameter trick: BFS from any pixel to find the farthest
    pixel A, then BFS from A to find the farthest pixel from A. That distance
    is the geodesic diameter. Diagonal steps count as sqrt(2) so the result
    is a physical distance, not a hop count.
    """
    import heapq

    pts = np.argwhere(mask_bool)
    if len(pts) <= 1:
        return 0.0

    index = {(int(r), int(c)): i for i, (r, c) in enumerate(pts)}
    neigh = [(-1, -1, 2 ** 0.5), (-1, 0, 1.0), (-1, 1, 2 ** 0.5), (0, -1, 1.0),
             (0, 1, 1.0), (1, -1, 2 ** 0.5), (1, 0, 1.0), (1, 1, 2 ** 0.5)]

    def farthest(start_idx):
        dist = np.full(len(pts), np.inf)
        dist[start_idx] = 0.0
        heap = [(0.0, start_idx)]
        while heap:
            d, i = heapq.heappop(heap)
            if d > dist[i]:
                continue
            r, c = pts[i]
            for dr, dc, w in neigh:
                j = index.get((int(r) + dr, int(c) + dc))
                if j is not None and d + w < dist[j]:
                    dist[j] = d + w
                    heapq.heappush(heap, (dist[j], j))
        finite = np.where(np.isfinite(dist))[0]
        best = finite[np.argmax(dist[finite])]
        return best, dist[best]

    a, _ = farthest(0)
    _, diameter = farthest(a)
    return float(diameter) * um_per_px


def severity_straight_line(mask, um_per_px):
    """evaluation.py-style: severity = length + 0.5 * sqrt(area), threshold 25.

    Merged voids (within 40um): lengths and areas summed, no gap distance added.
    Returns (max_severity, per_group_severities).
    """
    labelled = label(mask == VOID, connectivity=2)
    n = labelled.max()
    if n == 0:
        return 0.0, []

    coords = [np.argwhere(labelled == i) for i in range(1, n + 1)]
    lengths = [_straight_line_length_um(c, um_per_px) for c in coords]
    areas = [len(c) * um_per_px ** 2 for c in coords]

    groups = _merge_groups(coords, um_per_px)
    severities = []
    for grp in groups:
        L = sum(lengths[i] for i in grp)
        A = sum(areas[i] for i in grp)
        severities.append(L + 0.5 * np.sqrt(A))

    return (max(severities) if severities else 0.0), severities


def severity_geodesic(mask, um_per_px):
    """score_submission.py-style: severity = length * 0.5 * sqrt(area), threshold 60.

    Merged voids (within 40um): each void's own geodesic length summed, PLUS
    the gap distances between them (minimum spanning tree of the gaps).
    Returns (max_severity, per_group_severities).
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.spatial import cKDTree

    labelled = label(mask == VOID, connectivity=2)
    n = labelled.max()
    if n == 0:
        return 0.0, []

    comps = [labelled == i for i in range(1, n + 1)]
    coords = [np.argwhere(c) for c in comps]
    lengths = [_geodesic_length_um(c, um_per_px) for c in comps]
    areas = [len(c) * um_per_px ** 2 for c in coords]
    boundaries = [c * um_per_px for c in coords]

    gaps = np.full((n, n), np.inf)
    for i in range(n):
        tree = cKDTree(boundaries[i])
        for j in range(i + 1, n):
            d = tree.query(boundaries[j])[0].min()
            gaps[i, j] = gaps[j, i] = d

    groups, seen = [], set()
    for i in range(n):
        if i in seen:
            continue
        stack, grp = [i], []
        while stack:
            k = stack.pop()
            if k in seen:
                continue
            seen.add(k)
            grp.append(k)
            stack.extend(j for j in range(n)
                         if j not in seen and gaps[k, j] < MERGE_DISTANCE_UM)
        groups.append(sorted(grp))

    severities = []
    for grp in groups:
        if len(grp) == 1:
            L, A = lengths[grp[0]], areas[grp[0]]
        else:
            sub = gaps[np.ix_(grp, grp)].copy()
            sub[~np.isfinite(sub)] = 0.0
            mst = minimum_spanning_tree(csr_matrix(sub))
            L = sum(lengths[i] for i in grp) + float(mst.sum())
            A = sum(areas[i] for i in grp)
        severities.append(L * 0.5 * np.sqrt(A))

    return (max(severities) if severities else 0.0), severities


def _merge_groups(coords, um_per_px):
    """Group void indices whose gap is under MERGE_DISTANCE_UM (union-find)."""
    from scipy.spatial import cKDTree

    n = len(coords)
    boundaries = [c * um_per_px for c in coords]
    gaps = np.full((n, n), np.inf)
    for i in range(n):
        if len(boundaries[i]) == 0:
            continue
        tree = cKDTree(boundaries[i])
        for j in range(i + 1, n):
            if len(boundaries[j]) == 0:
                continue
            d = tree.query(boundaries[j])[0].min()
            gaps[i, j] = gaps[j, i] = d

    groups, seen = [], set()
    for i in range(n):
        if i in seen:
            continue
        stack, grp = [i], []
        while stack:
            k = stack.pop()
            if k in seen:
                continue
            seen.add(k)
            grp.append(k)
            stack.extend(j for j in range(n)
                         if j not in seen and gaps[k, j] < MERGE_DISTANCE_UM)
        groups.append(sorted(grp))
    return groups


def call(mask, um_per_px):
    """Both pass/fail calls for one mask. Returns dict, use both, not one."""
    sev_sl, _ = severity_straight_line(mask, um_per_px)
    sev_geo, _ = severity_geodesic(mask, um_per_px)
    return {
        "severity_straight_line": sev_sl,
        "fail_straight_line": sev_sl >= STRAIGHT_LINE_THRESHOLD,
        "severity_geodesic": sev_geo,
        "fail_geodesic": sev_geo >= GEODESIC_THRESHOLD_UM,
    }
