# Repo settings for portfolio use, do this after the hackathon, not before

## Rename the repo

Current name is `NCC`, too generic and unclear out of context. On GitHub: repo page, Settings, rename to something like:

- `cfrp-void-detection` (recommended, describes what it does, not who it's for)
- `composite-defect-segmentation`
- `ncc-composites-defect-detection` (keeps the client context, fine too)

Renaming doesn't break the local clone, `git remote -v` will still resolve via GitHub's redirect, but update your local remote URL afterward to be safe:
```bash
git remote set-url origin git@github.com:KNHNF/<new-name>.git
```

## About section (top right of the repo page)

- **Description**: one sentence, e.g. "U-Net segmentation and defect-severity scoring for carbon-fibre composite micrographs, built for the NCCUK AI Hackathon challenge."
- **Website**: leave blank unless you deploy the Gradio app somewhere permanent.
- **Topics**: add tags so it's discoverable and reads well on your profile, e.g. `computer-vision`, `pytorch`, `semantic-segmentation`, `u-net`, `hackathon`, `medical-imaging` (no, skip that one, not medical), `materials-science`.

## README polish

Already has methodology, results, both confusion matrices, the demo recording, and an honest limitations section, that's the strong part, a portfolio reviewer respects seeing what didn't work as much as what did. Add a one-line note near the top once the hackathon's over confirming the outcome (placed, didn't place, feedback received), context a reviewer will want.

## Pin it to your profile

GitHub profile page, "Customize your pins", select this repo. Given the depth here (real dataset audit, a caught specification conflict, honest reporting of a degenerate metric case), this is a genuinely strong portfolio piece for the quant/ML direction you're aiming at, better than a generic Kaggle-tutorial repo.

## License

Add an MIT license (GitHub's own "Add file" then "Choose a license template" does this in one click) if you want it clearly reusable. Optional, doesn't affect the hackathon in any way.
