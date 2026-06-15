"""
evaluate.py
-----------
Evaluate predicted masks against ground truth.

Reports three families of metrics:

1. Quality (Dice, IoU, HD95)
   Computed via a matching strategy (semantic or greedy).
   A matched pair whose IoU < --match-threshold (default 0.5) is demoted to
   missing: the GT contributes dice=0, iou=0, hd95=inf as if no prediction
   existed.  Pairs at or above the threshold are evaluated normally.
   Averaged over ALL GT entries including missed ones.

2. Coverage (Recall @ IoU threshold)
   Per GT, max IoU vs ANY predicted mask >= threshold.
   Independent of matching and match_threshold. Captures "how much of the GT
   does the method actually find?".

3. Cleanliness (Precision @ IoU threshold)
   Per prediction, max IoU vs ANY GT mask >= threshold.
   Independent of matching and match_threshold. Captures "how many of our
   predictions actually correspond to anatomical structures?".

   F1 is the harmonic mean of precision and recall.

Why three families?

   Quality (Dice) mixes "how good is the match" with "how much do I miss".
   Coverage isolates the recall side. Cleanliness isolates the precision
   side. With all three, the matched-vs-missed-vs-junk story is fully
   visible — Dice alone hides false positives (greedy matching discards extra
   predictions) and dilutes recall gains across many already-matched GT
   entries.

P/R are reported at multiple IoU thresholds (default: 0.1 to 0.9, step 0.1)
so the gains can be checked under both lenient and strict overlap criteria.
1.0 is excluded because at threshold=1.0 only pixel-perfect predictions are
true positives, which is degenerate for real-world masks.

Two matching strategies for the quality metrics:
- semantic:  match by organ name (few-shot / text-guided modes).
- greedy:    match by best IoU (unsupervised mode, where names are obj_N).

match_threshold (--match-threshold, default 0.5):
   IoU gate applied AFTER matching. A pair whose IoU is below this value
   is treated as if the GT was not detected: pred_name is set to None and
   quality metrics are computed against an empty mask (Dice=0, IoU=0,
   hd95=worst case). The match_threshold value is saved in summary.json
   for traceability. Coverage and mAP metrics are NOT affected by this gate
   (they scan their own threshold grids independently).

Usage:
    python evaluate.py \\
        --gt data/processed/XRayNicoSent/masks/ \\
        --pred results/.../masks/ \\
        --output results/.../

    # Custom IoU gate for quality matching:
    python evaluate.py ... --match-threshold 0.75

    # Custom IoU thresholds for P/R/F1 coverage metrics:
    python evaluate.py ... --iou-thresholds 0.5 0.75

Output:
    <output>/metrics.csv      — per-image per-organ quality metrics
    <output>/summary.json     — aggregated quality, coverage, cleanliness,
                                mAP, PR curves at 0.50 and 0.75 IoU
"""

from project.evaluation.runner import main

if __name__ == "__main__":
    main()
