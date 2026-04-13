#!/bin/bash
set -e

VERSION=${1:?"Usage: ./run_suite.sh <version> [--max-images N] [--override KEY=VAL ...]"}
shift

# Forward global options (--max-images, --override, --skip-*)
GLOBAL_OPTS=("$@")

# ── Dataset 1: XRay ──────────────────────────────────────────────────
./run_experiment.sh "$VERSION" XRayNicoSent/images \
    --fs-sizes 1 5 10 \
    --ref-images \
        1256842362861431725328351539259305635_u1qifz \
        10155709300728342918543955138521808206_f7cj92 \
        10287653421930576798556842610982533460_vpbhw6 \
        10383960670432673238945376919735423432_hd3moq \
        10996416492353037588312781035930080694_8rstz0 \
        13353724432735380699905228693882625716_1tbyf9 \
        CHNCXR_0291_0 \
        CHNCXR_0296_0 \
        CHNCXR_0297_0 \
        MCUCXR_0091_0 \
        MCUCXR_0092_0 \
    "${GLOBAL_OPTS[@]}"

# ── Dataset 2: Sunnybrook ────────────────────────────────────────────
./run_experiment.sh "$VERSION" SunnybrookNicoSent/images \
    --fs-sizes 1 5 10 \
    --ref-images \
        SCD0000101_IM_0003_0079 \
        SCD0000101_IM_0003_0199 \
        SCD0000101_IM_0003_0219 \
        SCD0000201_IM_0002_0060 \
        SCD0000301_IM_0003_0087 \
        SCD0000401_IM_0002_0060 \
        SCD0000401_IM_0002_0067 \
    "${GLOBAL_OPTS[@]}"

echo ""
echo "Suite complete. Results in results/${VERSION}/"