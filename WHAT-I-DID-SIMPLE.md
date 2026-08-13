# What I've been doing, in plain terms

## The problem, in one sentence

NCC gives us a photo of a cross-section of carbon fibre material. We need to find the empty gaps (voids) in it, measure how bad they are, and decide if the part is safe to use or not.

## What I built, step by step

**1. A way to check we're grading it right.**

NCC gave two different rulebooks for deciding pass or fail, and they don't agree with each other, one says fail if a defect scores 25, the other says fail if it scores 60, and they measure the defect differently too. I wrote code that follows both rulebooks exactly, then tested it against NCC's own scoring scripts on a fake example to prove my code gives the same answer they would. That's done and verified.

**2. Then found which rulebook is actually real.**

Looked at the evidence, a note left in NCC's own code history literally says "updated to match the judge," and one of NCC's own slides confirms the same numbers. So we now know which rulebook counts. I still built for both, just in case.

**3. Fixed a mistake in how we test ourselves.**

We only have about 1,550 real photos, but NCC gave us 4,000 files, because each photo also has stretched, flipped, or brightened copies. At first, my code was accidentally putting a photo's copy in one pile and the original in the other pile when checking how good the model is, which is like studying with the answers already half-memorised, it makes the model look better than it really is. Fixed that, so our test now only uses fair, unseen photos.

**4. Trained a model to spot the voids.**

Used a well-known type of AI model (a U-Net) that's good at labelling every single pixel in an image as either "solid material," "fibre," or "empty gap." Real result on photos it's never seen before: it gets the gaps right about 78% of the time by overlap, and out of 616 test photos, it only missed 13 real defects while correctly catching 185, which is a strong, honest result.

**5. Built a report card.**

Wrote code that takes the model's guesses, scores them the same way NCC would, and produces a clear picture (a confusion matrix) showing correct passes, correct fails, missed defects, and false alarms, all labelled in plain English, not just numbers.

## What's still to do

- Test the model on photos at a size it's never really seen before, since the real test photos are all one particular zoom level that's rare in our training photos. This tells us if the model will actually work on the real thing or just on photos similar to what it already knows.
- Get Akilesh's version (a simpler, non-AI method) finished so we can show both side by side.
- Wire the whole thing into a demo: upload a photo, watch it get marked up, see the measurement and the pass/fail decision explained in words, not just a picture.
