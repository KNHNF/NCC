# NCC, AI Hackathon composites defect challenge

Independent analysis for the National Composites Centre (NCCUK) challenge at the UWE Bristol / Tech West England Advocates / Rootcause.ai AI Hackathon, 13-14 Aug 2026. Team Nexus. This repo is my own working copy, built alongside the team's submission, not a replacement for it.

## The task

Segment carbon fibre composite micrographs into matrix, fibre and void, then call each specimen pass or fail on defect severity, matching NCC's real certification criteria.

## Why this repo exists

NCC supplied two scoring scripts, `evaluation.py` on GitHub and `score_submission.py` in their Drive folder, and the two disagree: different severity formula (addition vs multiplication), different pass/fail threshold (25 vs 60 microns), different void length definition (straight-line vs geodesic). We were told both run against submissions. `severity.py` here implements both, and `test_severity.py` checks each implementation directly against NCC's own scripts on a synthetic case, rather than trusting a hand-rolled version of either formula.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install numpy scipy pandas pillow scikit-image
```

Requires the challenge repo and NCC's Drive files cloned as siblings to this folder (`../ncc-challenge` and `../ncc-official-drive`), since `test_severity.py` imports NCC's real scripts directly rather than reimplementing them.

## Structure

- `severity.py` - both severity formulas, verified, this is the part that matters most
- `test_severity.py` - cross-checks `severity.py` against NCC's own `evaluation.py` and `score_submission.py` on a synthetic mask. Run this after touching either formula, before trusting the numbers.

## Status

Severity scoring verified against both official scripts. Model training and submission pipeline not started yet.
