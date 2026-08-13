"""Cross-check severity.py against NCC's own two scripts on a synthetic case.

Do not trust severity.py's numbers until this passes. The two official scripts
are imported directly (not reimplemented from memory) so there is no room for
a transcription error to hide a real bug.

Run: python test_severity.py
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "ncc-challenge"))
sys.path.insert(0, str(HERE.parent / "ncc-official-drive"))

import evaluation as gh_eval          # GitHub evaluation.py
import score_submission as drive_eval  # Drive score_submission.py
import severity as ours

UM_PER_PX = 2.0  # arbitrary for this synthetic test


def make_synthetic_mask():
    """A 40x40 mask with two voids: an L-shape (concave, so straight-line vs
    geodesic length actually differ) plus a small separate void close enough
    to merge with it (within 40um at 2 um/px, i.e. within 20px).
    """
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[5:20, 20:35] = 1  # fibre background so matrix/fibre/void all present

    # L-shaped void: straight-line farthest pair cuts across the concavity,
    # geodesic has to go around it, the two lengths should differ.
    mask[5:6, 5:20] = 2    # horizontal arm
    mask[5:20, 5:6] = 2    # vertical arm

    # Small separate void, 6px away from the L (well under the 20px merge
    # radius at 2 um/px), so it should merge into the same group.
    mask[25:28, 10:13] = 2

    return mask


def compare(name, ours_val, ref_val, tol=1e-6):
    ok = abs(ours_val - ref_val) < tol
    status = "MATCH" if ok else "MISMATCH"
    print(f"  [{status}] {name}: ours={ours_val:.4f}  reference={ref_val:.4f}")
    return ok


def main():
    mask = make_synthetic_mask()
    all_ok = True

    print("Straight-line formula vs GitHub evaluation.py")
    ours_sl, _ = ours.severity_straight_line(mask, UM_PER_PX)
    ref_sev, ref_n = gh_eval.compute_max_severity(mask, UM_PER_PX)
    all_ok &= compare("max severity", ours_sl, ref_sev)

    print("\nGeodesic formula vs Drive score_submission.py")
    ours_geo, _ = ours.severity_geodesic(mask, UM_PER_PX)
    _, ref_max = drive_eval.severity_for_image(mask, UM_PER_PX)
    all_ok &= compare("max severity", ours_geo, ref_max)

    print("\nCombined call()")
    result = ours.call(mask, UM_PER_PX)
    print(f"  {result}")

    print("\n" + ("ALL CHECKS PASSED" if all_ok else "MISMATCH FOUND, do not trust severity.py yet"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
