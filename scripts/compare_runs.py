#!/usr/bin/env python3
"""Compare two co-segmentation run directories for mask equivalence.

Usage:
    python scripts/compare_runs.py results/run_a results/run_b

Exit code 0 if all checks pass, non-zero otherwise.
"""

import argparse
import json
import sys
from pathlib import Path


def _check(label: str, passed: bool) -> bool:
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {label}")
    return passed


def compare_runs(dir_a: Path, dir_b: Path) -> bool:
    all_pass = True

    masks_a = dir_a / "masks"
    masks_b = dir_b / "masks"

    if not masks_a.exists() or not masks_b.exists():
        missing = []
        if not masks_a.exists():
            missing.append(str(masks_a))
        if not masks_b.exists():
            missing.append(str(masks_b))
        all_pass &= _check(f"masks/ directories exist", False)
        for m in missing:
            print(f"  Missing: {m}")
        return all_pass

    all_pass &= _check("masks/ directories exist", True)

    # 1. Same image subdirectories
    subdirs_a = {d.name for d in masks_a.iterdir() if d.is_dir()}
    subdirs_b = {d.name for d in masks_b.iterdir() if d.is_dir()}
    subdirs_ok = subdirs_a == subdirs_b
    all_pass &= _check("same image subdirectories", subdirs_ok)
    if not subdirs_ok:
        only_a = subdirs_a - subdirs_b
        only_b = subdirs_b - subdirs_a
        if only_a:
            print(f"  Only in A: {sorted(only_a)}")
        if only_b:
            print(f"  Only in B: {sorted(only_b)}")

    # 2. Same PNGs and byte-identical masks
    common_subdirs = subdirs_a & subdirs_b
    png_sets_ok = True
    masks_identical = True

    for subdir in sorted(common_subdirs):
        pngs_a = {f.name for f in (masks_a / subdir).glob("*.png")}
        pngs_b = {f.name for f in (masks_b / subdir).glob("*.png")}
        if pngs_a != pngs_b:
            png_sets_ok = False
            print(f"  PNG mismatch in {subdir}:")
            only_a = pngs_a - pngs_b
            only_b = pngs_b - pngs_a
            if only_a:
                print(f"    Only in A: {sorted(only_a)}")
            if only_b:
                print(f"    Only in B: {sorted(only_b)}")
            continue

        for png in sorted(pngs_a):
            ba = (masks_a / subdir / png).read_bytes()
            bb = (masks_b / subdir / png).read_bytes()
            if ba != bb:
                masks_identical = False
                print(f"  Byte mismatch: {subdir}/{png}")

    all_pass &= _check("same PNG filenames per image", png_sets_ok)
    all_pass &= _check("all mask PNGs byte-identical", masks_identical)

    # 3. scores.json per image subdirectory (if present)
    scores_checked = 0
    scores_ok = True
    for subdir in sorted(common_subdirs):
        sa_path = masks_a / subdir / "scores.json"
        sb_path = masks_b / subdir / "scores.json"
        if sa_path.exists() and sb_path.exists():
            sa = json.loads(sa_path.read_text())
            sb = json.loads(sb_path.read_text())
            if sa != sb:
                scores_ok = False
                print(f"  scores.json differs: {subdir}")
            scores_checked += 1
        elif sa_path.exists() != sb_path.exists():
            scores_ok = False
            print(f"  scores.json missing in one run: {subdir}")
            scores_checked += 1

    if scores_checked > 0:
        all_pass &= _check(f"scores.json identical ({scores_checked} checked)", scores_ok)

    # 4. summary.json
    sum_a = dir_a / "summary.json"
    sum_b = dir_b / "summary.json"
    if sum_a.exists() and sum_b.exists():
        sa = json.loads(sum_a.read_text())
        sb = json.loads(sum_b.read_text())
        all_pass &= _check("summary.json identical", sa == sb)
        if sa != sb:
            keys_differ = [k for k in set(sa) | set(sb) if sa.get(k) != sb.get(k)]
            print(f"  Differing keys: {keys_differ}")
    elif sum_a.exists() != sum_b.exists():
        print("[WARN] summary.json present in only one run — skipping comparison")

    return all_pass


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compare two co-segmentation run directories."
    )
    p.add_argument("dir_a", help="First run directory")
    p.add_argument("dir_b", help="Second run directory")
    args = p.parse_args()

    dir_a = Path(args.dir_a)
    dir_b = Path(args.dir_b)

    print(f"Comparing:\n  A: {dir_a}\n  B: {dir_b}\n")
    ok = compare_runs(dir_a, dir_b)
    print()
    print("Result:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
