"""
evaluate.py
-----------
Evaluate predicted masks against ground truth.

Reports three families of metrics:

1. Quality (Dice, IoU, HD95)
   Computed via a matching strategy (semantic or hungarian).
   Averaged over ALL GT entries — including missed ones, which contribute
   dice=0, iou=0, hd95=inf. Captures "given each GT, how good is the best
   matching prediction, including the case of no prediction at all".

2. Coverage (Recall @ IoU threshold)
   Per GT, max IoU vs ANY predicted mask >= threshold.
   Independent of matching. Captures "how much of the GT does the method
   actually find?".

3. Cleanliness (Precision @ IoU threshold)
   Per prediction, max IoU vs ANY GT mask >= threshold.
   Independent of matching. Captures "how many of our predictions actually
   correspond to anatomical structures?".

   F1 is the harmonic mean of precision and recall.

Why three families?

   Quality (Dice) mixes "how good is the match" with "how much do I miss".
   Coverage isolates the recall side. Cleanliness isolates the precision
   side. With all three, the matched-vs-missed-vs-junk story is fully
   visible — Dice alone hides false positives (Hungarian discards extra
   predictions) and dilutes recall gains across many already-matched GT
   entries.

P/R are reported at multiple IoU thresholds (default: 0.5 and 0.7) so the
gains can be checked under both a lenient "anatomical hit" criterion and
a stricter "tight boundary" criterion.

Two matching strategies for the quality metrics:
- semantic:  match by organ name (few-shot / text-guided modes).
- hungarian: match by best IoU (unsupervised mode, where names are obj_N).

Usage:
    python evaluate.py \\
        --gt data/processed/XRayNicoSent/masks/ \\
        --pred results/.../masks/ \\
        --output results/.../

    # Custom IoU thresholds for P/R/F1:
    python evaluate.py ... --iou-thresholds 0.5 0.75

Output:
    <output>/metrics.csv      — per-image per-organ quality metrics
    <output>/summary.json     — aggregated quality, coverage, cleanliness
"""

from project.evaluation.runner import main

if __name__ == "__main__":
    main()
