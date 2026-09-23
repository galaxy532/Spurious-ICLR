#!/usr/bin/env bash
# =====================================================================================
# Session 5 -- the alpha > 1 hunt, moved from the images to the PARTITION.
#
# Read SESSION5.md first. In one paragraph: sessions 1-4 varied the backbone and then
# the images, and every per-group margin ratio came back an exact tie. Both of those
# kept Waterbirds' group variable, g = 1[place != y], which is derived from the
# background -- a nuisance chosen to be uncorrelated with how hard the bird is to
# recognise. Session 5 leaves every pixel and every feature alone and changes only the
# PARTITION, to one built from bird size in frame, which is a real difficulty axis.
#
# It also does something sessions 1-4 never did: it calibrates the instrument, so that
# a tie can be distinguished from an asymmetry too small for the instrument to see.
#
# THERE IS NO GPU WORK IN THIS SESSION AT ALL.
# The features are the ones session 4 already extracted. Everything here is CPU:
# reading mask PNGs, and LinearSVC fits on a 4795 x 768 matrix. Expect ~20-40 minutes
# end to end, well inside one sitting, so there is a single stage rather than A/B.
#
#   bash run_session5.sh
#
# What it writes, all under results/ so a `git pull` brings everything back:
#   results/v5_bird_fraction.{csv,json,md}       bird-pixel fraction per image
#   results/v5_margin_power.{json,md}            THE CALIBRATION -- read this first
#   results/v5_group_margins_*.{json,md}         the re-partitioned margin ratios
#   results/features_v5_*.npz                    the re-partitioned bundles
#   results/logs/session5_<stamp>/SUMMARY.md     every step, command, duration, tail
#   results/logs/session5_<stamp>/*.log          the full log of every step
#
# Options
#   BUNDLE=<path>   the source bundle. Left unset, the undegraded session-4 dinov2
#                   train bundle is located automatically -- feature bundles are
#                   gitignored (`*.npz`), so where they sit is a property of the
#                   machine that extracted them, not of the repo. Session 4 wrote
#                   them to the repo root; `results/` is searched too.
#   RULES="..."     partition rules to try (default: "median tercile")
#   DATA_ROOT=...   passed through as SPURIOUS_DATA_ROOT if you set it
#   NO_DOWNLOAD=1   fail instead of fetching the CUB segmentations
#   PUSH=1          commit and push results/ at the end
# =====================================================================================

set -u
cd "$(dirname "$0")"
PY=${PYTHON:-python}
BUNDLE=${BUNDLE:-}
WANT=features_v4_waterbirds_dinov2_train.npz
RULES=${RULES:-"median tercile"}
NO_DOWNLOAD=${NO_DOWNLOAD:-}
STAMP=$(date +%Y%m%d_%H%M%S)
LOG=results/logs/session5_${STAMP}
mkdir -p "$LOG"
SUMMARY=$LOG/SUMMARY.md
START_ALL=$(date +%s)

if [ -n "${DATA_ROOT:-}" ]; then export SPURIOUS_DATA_ROOT="$DATA_ROOT"; fi

# ---------------------------------------------------------------------------
# Locate the source bundle.
#
# `*.npz` is gitignored, so a bundle's location is a property of the machine that
# extracted it rather than of the repo, and hardcoding one path is how this
# aborted the first time. Session 4 wrote bundles to the repo root
# (`--bundles features_v4_waterbirds_dinov2*_train.npz`, a bare glob); `results/`
# is searched as well. The exact name has no degradation tag, so an exact-name
# search cannot pick up `..._res100g0_train.npz` by accident.
# ---------------------------------------------------------------------------
if [ -z "$BUNDLE" ]; then
  # Explicit candidates in preference order, so the choice is deterministic
  # rather than whatever `find` happens to return first.
  for CAND in "$WANT" "results/$WANT" "../$WANT"; do
    if [ -f "$CAND" ]; then BUNDLE="$CAND"; break; fi
  done
fi
if [ -z "$BUNDLE" ]; then
  BUNDLE=$(find . -maxdepth 3 -name "$WANT" -type f 2>/dev/null | head -1)
fi

if [ -z "$BUNDLE" ] || [ ! -f "$BUNDLE" ]; then
  {
    echo "# Session 5 summary ($STAMP)"
    echo ""
    echo "## ABORTED before any measurement"
    echo ""
    echo "Could not find the source bundle \`$WANT\`."
    echo ""
    echo "Feature bundles are gitignored, so this is about where they live on THIS"
    echo "machine, not about the repo. What is present:"
    echo ""
    echo '```'
    find . -maxdepth 2 -name 'features_v*.npz' -type f 2>/dev/null | head -40
    echo '```'
    echo ""
    echo "Re-run pointing at the right one, e.g."
    echo ""
    echo '    BUNDLE=path/to/bundle.npz nohup setsid bash run_session5.sh > session5_nohup.out 2>&1 &'
  } >> "$SUMMARY"
  echo "ABORTED: could not find $WANT. Bundles present:" >&2
  find . -maxdepth 2 -name 'features_v*.npz' -type f 2>/dev/null | head -40 >&2
  echo "Re-run with BUNDLE=<path> if it is somewhere else." >&2
  exit 2
fi
echo "source bundle: $BUNDLE"

echo "# Session 5 summary ($STAMP)" > "$SUMMARY"
{
  echo ""
  echo "Started $(date '+%Y-%m-%d %H:%M:%S %Z') on $(hostname)."
  echo ""
  echo "- source bundle: \`$BUNDLE\`"
  echo "- partition rules: \`$RULES\`"
  echo "- no GPU work in this session; everything below is CPU."
  echo ""
} >> "$SUMMARY"

{
  echo "== date";   date
  echo "== git";    git rev-parse HEAD 2>&1; git status --short 2>&1 | head -50
  echo "== python"; $PY --version 2>&1
  $PY -c "import numpy, sklearn, PIL, pandas; print('numpy', numpy.__version__, 'sklearn', sklearn.__version__, 'pillow', PIL.__version__, 'pandas', pandas.__version__)" 2>&1
  echo "== data root"; $PY -c "import datasets; print(datasets.DATA_ROOT)" 2>&1
  echo "== bundle";  ls -l "$BUNDLE" 2>&1
} > "$LOG/environment.txt" 2>&1

run_step() {
  local name=$1; shift
  local t0 t1 rc
  t0=$(date +%s)
  echo "[$(date +%H:%M:%S)] START $name: $*"
  ls -1 results 2>/dev/null | sort > "$LOG/$name.files.before"
  ( "$@" ) 2>&1 | tee "$LOG/$name.raw"
  rc=${PIPESTATUS[0]}
  sed 's/.*\r//' "$LOG/$name.raw" > "$LOG/$name.log" && rm -f "$LOG/$name.raw"
  t1=$(date +%s)
  ls -1 results 2>/dev/null | sort > "$LOG/$name.files.after"
  echo "[$(date +%H:%M:%S)] END   $name: exit $rc after $(( (t1 - t0) / 60 )) min"
  {
    echo "## $name"
    echo ""
    echo "- command: \`$*\`"
    echo "- start: $(date -d @$t0 '+%H:%M:%S'), end: $(date -d @$t1 '+%H:%M:%S'), duration: $(( (t1 - t0) / 60 )) min $(( (t1 - t0) % 60 )) s"
    echo "- exit code: **$rc** $( [ "$rc" -eq 0 ] && echo '(ok)' || echo '(FAILED)')"
    echo "- log: \`$LOG/$name.log\`"
    echo "- new files in results/: $(comm -13 "$LOG/$name.files.before" "$LOG/$name.files.after" | tr '\n' ' ')"
    echo ""
    echo '```'
    tail -n 60 "$LOG/$name.log" 2>/dev/null
    echo '```'
    echo ""
  } >> "$SUMMARY"
  rm -f "$LOG/$name.files.before" "$LOG/$name.files.after"
  return $rc
}

# =========================== stage 0: self-checks ====================================
# Same discipline as session 4: if the pieces do not pass their own tests, nothing
# measured afterwards would mean anything, so nothing is measured.
run_step 00_selftest_cub_masks    $PY cub_masks.py --self-test;    A=$?
run_step 01_selftest_regroup      $PY regroup.py --self-test;      B=$?
run_step 02_selftest_margin_power $PY margin_power.py --self-test; C=$?
run_step 03_integration           $PY validate_session5.py;        D=$?

if [ $A -ne 0 ] || [ $B -ne 0 ] || [ $C -ne 0 ] || [ $D -ne 0 ]; then
  {
    echo "## ABORTED"
    echo ""
    echo "A self-check failed. Nothing measured afterwards would mean anything, so"
    echo "no measurement was run. Read the four logs above."
  } >> "$SUMMARY"
  echo "ABORTED: self-check failed. Read $SUMMARY"
  exit 1
fi

# =========================== stage 1: the calibration ================================
# Deliberately FIRST. It answers whether any partition of these features can produce a
# ratio other than 1. If the answer is no, the bird-size result below is already
# explained before it is read, and the whole line is settled for ~10 minutes of CPU.
run_step 10_margin_power $PY margin_power.py --bundle "$BUNDLE" --tag v5_margin_power
POWER_OK=$?

# =========================== stage 2: the masks ======================================
DL_FLAG=""
[ -n "$NO_DOWNLOAD" ] && DL_FLAG="--no-download"
# shellcheck disable=SC2086
run_step 20_cub_masks $PY cub_masks.py $DL_FLAG --tag v5_bird_fraction
MASKS_OK=$?

if [ $MASKS_OK -ne 0 ]; then
  {
    echo "## PARTITION SKIPPED"
    echo ""
    echo "\`cub_masks.py\` did not finish cleanly. The usual cause is the alignment"
    echo "check: if a mask's pixel dimensions differ from its composite's, the CUB"
    echo "segmentation does not describe the Waterbirds image and the bird fraction"
    echo "is meaningless. Read \`results/v5_bird_fraction.md\`."
    echo ""
    echo "The calibration in \`10_margin_power\` above is unaffected and stands on"
    echo "its own -- it needs no masks."
  } >> "$SUMMARY"
  echo "masks failed; skipping the partition. Read $SUMMARY"
else
  # ======================== stage 3: the re-partition ================================
  for RULE in $RULES; do
    run_step "30_regroup_${RULE}" $PY regroup.py \
        --bundle "$BUNDLE" --fractions results/v5_bird_fraction.csv --rule "$RULE"
    if [ $? -eq 0 ]; then
      run_step "31_margins_${RULE}" $PY group_margins.py --no-lp \
          --tag "v5_group_margins_${RULE}" \
          --bundles "results/features_v5_*_bf${RULE}*.npz"
    else
      { echo "### 30_regroup_${RULE} failed; its margins were not measured."; echo ""; } \
        >> "$SUMMARY"
    fi
  done
fi

# =========================== wrap up =================================================
END_ALL=$(date +%s)
{
  echo "## WHAT TO DO NEXT"
  echo ""
  echo "Read in this order. The first one decides how to read the rest."
  echo ""
  echo "1. \`results/v5_margin_power.md\` -- **the calibration**. Its \"Answer\""
  echo "   section says whether ANY partition of these features can give a ratio"
  echo "   other than 1. If it says NO, then every tie below is explained by the"
  echo "   instrument and not by Waterbirds, and that is the finding of the session."
  echo "   Check the \`pinned\` rows first: they must recover their planted ratio, or"
  echo "   nothing else on that page can be read."
  echo ""
  echo "2. \`results/v5_bird_fraction.md\` -- the alignment verdict must be OK, and"
  echo "   the three leakage AUCs should sit near 0.5. An AUC far from 0.5 against"
  echo "   \`y\` means bird size is a proxy for the label and the partition is void."
  echo ""
  echo "3. \`results/v5_group_margins_median.md\` -- the headline re-partition. Every"
  echo "   row is the SAME features as the session-4 control; only \`g\` differs."
  echo ""
  echo "4. \`results/v5_group_margins_tercile.md\` -- read the treatment row against"
  echo "   its \`_control\` row, NOT against session 4. The tercile rule drops the"
  echo "   middle third and re-standardises, so only the matched control is comparable."
  echo ""
  echo "Total wall clock: $(( (END_ALL - START_ALL) / 60 )) min."
  echo ""
} >> "$SUMMARY"

if [ "${PUSH:-}" = "1" ]; then
  run_step 90_push bash -c "git add -A results && git commit -m 'session 5: bird-size re-partition + margin-ratio calibration ($STAMP)' && git push"
fi

echo "done. Read $SUMMARY"
