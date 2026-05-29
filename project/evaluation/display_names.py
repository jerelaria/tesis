"""Display names and ordering for experiments and metrics across plots."""

DISPLAY_NAMES = {
    # --- Unsupervised ---
    "unsup_baseline":         "Grid Prompting",
    "unsup_kmeans":           "KMeans",
    "unsup_hdbscan":          "HDBSCAN",
    "unsup_kmeans_refine":    "KMeans + Refine",
    "unsup_hdbscan_refine":   "HDBSCAN + Refine",
    # --- Few-shot baselines ---
    "fs_indep_baseline_1ref":  "SAM2 Video (1 ref)",
    "fs_indep_baseline_3ref":  "SAM2 Video (3 ref)",
    "fs_indep_baseline_5ref":  "SAM2 Video (5 ref)",
    "fs_indep_baseline_10ref": "SAM2 Video (10 ref)",
    # --- Few-shot independent ---
    "fs_indep_1ref":           "Independent (1 ref)",
    "fs_indep_3ref":           "Independent (3 ref)",
    "fs_indep_5ref":           "Independent (5 ref)",
    "fs_indep_10ref":          "Independent (10 ref)",
    "fs_indep_refine_1ref":    "Indep + Refine (1 ref)",
    "fs_indep_refine_3ref":    "Indep + Refine (3 ref)",
    "fs_indep_refine_5ref":    "Indep + Refine (5 ref)",
    "fs_indep_refine_10ref":   "Indep + Refine (10 ref)",
    # --- Few-shot iterative ---
    "fs_iter_1ref":            "Iterative (1 ref)",
    "fs_iter_3ref":            "Iterative (3 ref)",
    "fs_iter_5ref":            "Iterative (5 ref)",
    "fs_iter_10ref":           "Iterative (10 ref)",
    "fs_iter_refine_1ref":     "Iter + Refine (1 ref)",
    "fs_iter_refine_3ref":     "Iter + Refine (3 ref)",
    "fs_iter_refine_5ref":     "Iter + Refine (5 ref)",
    "fs_iter_refine_10ref":    "Iter + Refine (10 ref)",
}

# Canonical display order: grouped by approach, then by ref count.
# Unknown experiments are appended alphabetically at the end.
EXPERIMENT_ORDER = [
    # Unsupervised
    "unsup_baseline",
    "unsup_kmeans",            "unsup_kmeans_refine",
    "unsup_hdbscan",           "unsup_hdbscan_refine",
    # Few-shot 1-ref (baseline first, then pipeline variants)
    "fs_indep_baseline_1ref",
    "fs_indep_1ref",           "fs_indep_refine_1ref",
    "fs_iter_1ref",            "fs_iter_refine_1ref",
    # Few-shot 3-ref
    "fs_indep_baseline_3ref",
    "fs_indep_3ref",           "fs_indep_refine_3ref",
    "fs_iter_3ref",            "fs_iter_refine_3ref",
    # Few-shot 5-ref
    "fs_indep_baseline_5ref",
    "fs_indep_5ref",           "fs_indep_refine_5ref",
    "fs_iter_5ref",            "fs_iter_refine_5ref",
    # Few-shot 10-ref
    "fs_indep_baseline_10ref",
    "fs_indep_10ref",          "fs_indep_refine_10ref",
    "fs_iter_10ref",           "fs_iter_refine_10ref",
]

MODE_COLORS = {
    "unsup":    "#5B8DB8",
    "fs_indep": "#E8A838",
    "fs_iter":  "#6BBF6B",
}


def get_display_name(name: str) -> str:
    return DISPLAY_NAMES.get(name, name)
