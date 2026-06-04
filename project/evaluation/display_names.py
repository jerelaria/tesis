"""Display names and ordering for experiments and metrics across plots."""

DISPLAY_NAMES = {
    # --- Unsupervised ---
    "unsup_baseline":                  "Grid Prompting (raw)",
    "unsup_hdbscan":                   "HDBSCAN (no prop.)",
    "unsup_hdbscan_propagation":       "HDBSCAN + Prop. Indep.",
    "unsup_hdbscan_propagation_iter":  "HDBSCAN + Prop. Iter.",
    # --- Few-shot baseline ---
    "fs_baseline_1ref":   "SAM2 Video (1 ref)",
    "fs_baseline_4ref":   "SAM2 Video (4 ref)",
    "fs_baseline_7ref":   "SAM2 Video (7 ref)",
    "fs_baseline_10ref":  "SAM2 Video (10 ref)",
    # --- Few-shot independent propagation ---
    "fs_propagation_1ref":   "FS Prop. Indep. (1 ref)",
    "fs_propagation_4ref":   "FS Prop. Indep. (4 ref)",
    "fs_propagation_7ref":   "FS Prop. Indep. (7 ref)",
    "fs_propagation_10ref":  "FS Prop. Indep. (10 ref)",
    # --- Few-shot iterative propagation ---
    "fs_propagation_iter_1ref":   "FS Prop. Iter. (1 ref)",
    "fs_propagation_iter_4ref":   "FS Prop. Iter. (4 ref)",
    "fs_propagation_iter_7ref":   "FS Prop. Iter. (7 ref)",
    "fs_propagation_iter_10ref":  "FS Prop. Iter. (10 ref)",
}

# Canonical display order: unsupervised first, then few-shot grouped by K.
# Unknown experiments are appended alphabetically at the end.
EXPERIMENT_ORDER = [
    # Unsupervised
    "unsup_baseline",
    "unsup_hdbscan",
    "unsup_hdbscan_propagation",
    "unsup_hdbscan_propagation_iter",
    # Few-shot 1-ref
    "fs_baseline_1ref",
    "fs_propagation_1ref",
    "fs_propagation_iter_1ref",
    # Few-shot 4-ref
    "fs_baseline_4ref",
    "fs_propagation_4ref",
    "fs_propagation_iter_4ref",
    # Few-shot 7-ref
    "fs_baseline_7ref",
    "fs_propagation_7ref",
    "fs_propagation_iter_7ref",
    # Few-shot 10-ref
    "fs_baseline_10ref",
    "fs_propagation_10ref",
    "fs_propagation_iter_10ref",
]

MODE_COLORS = {
    "unsup":            "#5B8DB8",
    "fs_baseline":      "#E8A838",
    "fs_propagation":   "#6BBF6B",
    "fs_propagation_iter": "#B87A5B",
}


def get_display_name(name: str) -> str:
    return DISPLAY_NAMES.get(name, name)
