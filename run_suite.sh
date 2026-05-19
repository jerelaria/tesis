#!/bin/bash
set -e

VERSION=${1:?"Usage: ./run_suite.sh <version> [--max-images N] [--override KEY=VAL ...]"}
shift

# Forward global options (--max-images, --override, --skip-*)
GLOBAL_OPTS=("$@")

# ── Dataset 1: XRay ──────────────────────────────────────────────────
# ./run_experiment.sh "$VERSION" XRayNicoSent/images \
#     --fs-sizes 1 3 \
#     --ref-images \
#         1256842362861431725328351539259305635_u1qifz \
#         10155709300728342918543955138521808206_f7cj92 \
#         10287653421930576798556842610982533460_vpbhw6 \
#         10383960670432673238945376919735423432_hd3moq \
#         10996416492353037588312781035930080694_8rstz0 \
#         13353724432735380699905228693882625716_1tbyf9 \
#         CHNCXR_0291_0 \
#         CHNCXR_0296_0 \
#         CHNCXR_0297_0 \
#         MCUCXR_0091_0 \
#         MCUCXR_0092_0 \
#     "${GLOBAL_OPTS[@]}"

# # ── Dataset 2: Sunnybrook ────────────────────────────────────────────
# ./run_experiment.sh "$VERSION" SunnybrookNicoSent/images \
#     --fs-sizes 1 3 \
#     --ref-images \
#         SCD0000101_IM_0003_0079 \
#         SCD0000101_IM_0003_0199 \
#         SCD0000101_IM_0003_0219 \
#         SCD0000201_IM_0002_0060 \
#         SCD0000301_IM_0003_0087 \
#         SCD0000401_IM_0002_0060 \
#         SCD0000401_IM_0002_0067 \
#     "${GLOBAL_OPTS[@]}"

# ── Dataset 3: Wrist_AP ──────────────────────────────────────────────
# ./run_experiment.sh "$VERSION" Wrist_AP/images \
#     --fs-sizes 1 3 5 7\
#     --ref-images \
#         3_jpg.rf.6d6b62374823ff3eab9270d28d0c736c \
#         8_jpg.rf.3d18eb5a4637820198c9755d7e4a0c2f \
#         63_jpg.rf.54dd616072a977faa0ca2b3da5678c3d \
#         194_jpg.rf.5ab50347c1f663c9b96e945c17b6a468 \
#         260_jpg.rf.44467fa3fc02357cbae7cefd91980074 \
#         313_jpg.rf.963a42f593c058e54f0b9510742d663a \
#         320_jpg.rf.51c4bab542f31ab4c719e1a1c2828a21 \
#     "${GLOBAL_OPTS[@]}"

# # ── Dataset 3: Sunnybrook basal stratum ──────────────────────────────
# ./run_experiment.sh "$VERSION" SunnybrookNicoSent_basal/images \
#     --max-images 30 \
#     --fs-sizes 1 3 \
#     --skip-textguided \
#     "${GLOBAL_OPTS[@]}"

# # ── Dataset 4: Sunnybrook mid stratum ────────────────────────────────
# ./run_experiment.sh "$VERSION" SunnybrookNicoSent_mid/images \
#     --max-images 30 \
#     --fs-sizes 1 3 \
#     --skip-textguided \
#     "${GLOBAL_OPTS[@]}"

# # ── Dataset 5: Sunnybrook apex stratum ───────────────────────────────
# ./run_experiment.sh "$VERSION" SunnybrookNicoSent_apex/images \
#     --max-images 30 \
#     --fs-sizes 1 3 \
#     --skip-textguided \
#     "${GLOBAL_OPTS[@]}"

# ── Dataset: SyntheticV1_easy (Block 1: sanity check) ─────────────
# ./run_experiment.sh "$VERSION" SyntheticV1_easy/images \
#     --fs-sizes 1 5 10 \
#     "${GLOBAL_OPTS[@]}"

 ./run_experiment.sh "$VERSION" SyntheticV1_medium/images \
    --fs-sizes 1 5 10 \
    "${GLOBAL_OPTS[@]}"
