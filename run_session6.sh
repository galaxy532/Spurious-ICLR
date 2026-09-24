#!/usr/bin/env bash
# =====================================================================================
# Session 6 -- screen pre-existing image properties for a group that avoids the margin.
#
# Read SESSION6.md first. In one paragraph: session 5 showed that the separator w_hat is
# fitted on (phi, y) alone, so when only the group variable changes the per-group margin
# ratio exceeds 1 if and only if one group contains NONE of the points attaining the
# global margin (421 of 4,795 on the dinov2 train bundle). Bird size failed that by 184
# points. This session asks whether ANY pre-registered, pre-existing property of the
# photographs -- CUB's 312 attributes, part visibility, bounding box, bird size tails --
# picks out a set of images that avoids those points more than chance allows.
#
# THERE IS NO GPU WORK. One SVM fit per split, then every candidate is a lookup. The
# only slow step is the one-time download of CUB-200-2011 (1.2 GB; only its text files
# are extracted). Expect ~5 minutes of compute plus the download.
#
#   PUSH=1 nohup setsid bash run_session6.sh > results/logs/session6_nohup.out 2>&1 &
#
# With PUSH=1, results/ is committed and pushed WHENEVER the script exits -- on success,
# on an abort, or on a failed step -- so the logs of a failed run come back on a
# `git pull` too. The nohup file is written into results/logs/ for the same reason.
#
# What it writes, all under results/ so a `git pull` brings it back:
#   results/v6_cub_meta.{md,json}              the CUB join and its checks
#   results/v6_sv_screen.{md,json}             THE SCREEN -- read this first
#   results/v6_sv_screen_full.md               every candidate, eligible or not
#   results/v6_confirm_check.md                screen vs the unmodified group_margins.py
#   results/v6_confirm_margins.{md,json}       group_margins.py on the confirmation bundles
#   results/v6_sv_screen_test.{md,json}        the same screen on the test split
#   results/v6_replication.md                  do the train hits replicate on test?
#   results/logs/session6_<stamp>/SUMMARY.md   every step, command, duration, tail
# Not pulled (gitignored): results/v6_cub_meta.npz, results/features_v6_*.npz.
#
# Options
#   BUNDLE=<path>       train bundle (default: features_v4_waterbirds_dinov2_train.npz,
#                       located automatically, as in session 5)
#   TEST_BUNDLE=<path>  test bundle for the replication (default: the dinov2 test bundle,
#                       v4 then v3, located automatically; skipped if none is found)
#   MIN_FRAC=0.01       smallest eligible group, as a fraction of n
#   DATA_ROOT=...       passed through as SPURIOUS_DATA_ROOT if you set it
#   NO_DOWNLOAD=1       fail instead of fetching CUB-200-2011 / the segmentations
#   PUSH=1              commit and push results/ when the script exits, however it exits
# =====================================================================================

set -u
cd "$(dirname "$0")"
PY=${PYTHON:-python}
BUNDLE=${BUNDLE:-}
TEST_BUNDLE=${TEST_BUNDLE:-}
MIN_FRAC=${MIN_FRAC:-0.01}
NO_DOWNLOAD=${NO_DOWNLOAD:-}
STAMP=$(date +%Y%m%d_%H%M%S)
LOG=results/logs/session6_${STAMP}
mkdir -p "$LOG"
SUMMARY=$LOG/SUMMARY.md
START_ALL=$(date +%s)

if [ -n "${DATA_ROOT:-}" ]; then export SPURIOUS_DATA_ROOT="$DATA_ROOT"; fi

# ---------------------------------------------------------------------------
# Locate bundles. `*.npz` is gitignored, so where a bundle sits is a property of
# the machine that extracted it, not of the repo (session 5, SESSION5.md §8).
# ---------------------------------------------------------------------------
locate() {   # locate <name> ... : first existing among ./, results/, ../, then find
  local name cand
  for name in "$@"; do
    for cand in "$name" "results/$name" "../$name"; do
      if [ -f "$cand" ]; then echo "$cand"; return 0; fi
    done
  done
  for name in "$@"; do
    cand=$(find . -maxdepth 3 -name "$name" -type f 2>/dev/null | head -1)
    if [ -n "$cand" ]; then echo "$cand"; return 0; fi
  done
  return 1
}

[ -z "$BUNDLE" ] && BUNDLE=$(locate features_v4_waterbirds_dinov2_train.npz)
[ -z "$TEST_BUNDLE" ] && TEST_BUNDLE=$(locate features_v4_waterbirds_dinov2_test.npz \
                                               features_v3_waterbirds_dinov2_test.npz)

echo "# Session 6 summary ($STAMP)" > "$SUMMARY"

# ---------------------------------------------------------------------------
# Push on EVERY exit path. Session 5's runner pushed only at the very end, so a
# run that aborted early (a failed self-check, a failed download) brought nothing
# back and the log had to be copied by hand. A trap fires on any exit.
# ---------------------------------------------------------------------------
finish() {
  local rc=$?
  if [ "${PUSH:-}" = "1" ]; then
    { echo "## 90_push"; echo "";
      echo "- committing and pushing results/ (the run exited with status $rc)"; echo ""; } >> "$SUMMARY"
    if { git add -A results && git commit -q -m "session 6 results ($STAMP, exit $rc)" \
         && git push; } > "$LOG/90_push.log" 2>&1; then
      echo "pushed results/ (run exit status $rc)"
    else
      echo "PUSH FAILED -- see $LOG/90_push.log, then push by hand:" >&2
      echo "  git add -A results && git commit -m 'session 6 results' && git push" >&2
    fi
  fi
}
trap finish EXIT
if [ -z "$BUNDLE" ] || [ ! -f "$BUNDLE" ]; then
  {
    echo ""
    echo "## ABORTED before any measurement"
    echo ""
    echo "Could not find the train bundle \`features_v4_waterbirds_dinov2_train.npz\`."
    echo "Bundles present on this machine:"
    echo ""
    echo '```'
    find . .. -maxdepth 2 -name 'features_v*.npz' -type f 2>/dev/null | head -40
    echo '```'
    echo ""
    echo "Re-run with BUNDLE=<path>."
  } >> "$SUMMARY"
  echo "ABORTED: no train bundle. Read $SUMMARY" >&2
  exit 2
fi
echo "train bundle: $BUNDLE"
echo "test bundle:  ${TEST_BUNDLE:-(none found; replication will be skipped)}"

{
  echo ""
  echo "Started $(date '+%Y-%m-%d %H:%M:%S %Z') on $(hostname)."
  echo ""
  echo "- train bundle: \`$BUNDLE\`"
  echo "- test bundle (replication): \`${TEST_BUNDLE:-none found -- replication skipped}\`"
  echo "- minimum group size: ${MIN_FRAC} of n"
  echo "- no GPU work in this session."
  echo ""
} >> "$SUMMARY"

{
  echo "== date";   date
  echo "== git";    git rev-parse HEAD 2>&1; git status --short 2>&1 | head -50
  echo "== python"; $PY --version 2>&1
  $PY -c "import numpy, scipy, sklearn, pandas; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'sklearn', sklearn.__version__, 'pandas', pandas.__version__)" 2>&1
  echo "== data root"; $PY -c "import datasets; print(datasets.DATA_ROOT)" 2>&1
  echo "== bundles"; ls -l "$BUNDLE" ${TEST_BUNDLE:+"$TEST_BUNDLE"} 2>&1
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

note() {   # note <heading> <text...> : heading on its own line, then the text
  local h=$1; shift
  { echo "$h"; echo ""; echo "$@"; echo ""; } >> "$SUMMARY"
}

# =========================== stage 0: self-checks ====================================
run_step 00_selftest_cub_meta   $PY cub_meta.py --self-test;   A=$?
run_step 01_selftest_sv_screen  $PY sv_screen.py --self-test;  B=$?
run_step 02_integration         $PY validate_session6.py;      C=$?
if [ $A -ne 0 ] || [ $B -ne 0 ] || [ $C -ne 0 ]; then
  note "## ABORTED" "A self-check failed, so nothing was measured. Read the three logs above."
  echo "ABORTED: self-check failed. Read $SUMMARY"
  exit 1
fi

DL_FLAG=""
[ -n "$NO_DOWNLOAD" ] && DL_FLAG="--no-download"

# =========================== stage 1: metadata =======================================
# The bird fraction and image sizes come from session 5's csv. It is gitignored, so it
# exists only where session 5 ran; recompute it (1-2 min) if it is not here.
if [ -f results/v5_bird_fraction.csv ]; then
  note "## 10_bird_fraction" "- reused \`results/v5_bird_fraction.csv\` from session 5."
else
  # shellcheck disable=SC2086
  run_step 10_bird_fraction $PY cub_masks.py $DL_FLAG --tag v5_bird_fraction
  if [ $? -ne 0 ]; then
    note "## ABORTED" "cub_masks.py failed; the screen needs its csv. Read its log above."
    exit 1
  fi
fi

# shellcheck disable=SC2086
run_step 20_cub_meta $PY cub_meta.py $DL_FLAG --fractions results/v5_bird_fraction.csv
if [ $? -ne 0 ]; then
  note "## ABORTED" "cub_meta.py failed its structural checks or could not fetch the" \
       "archive. Read \`results/v6_cub_meta.md\` and the log above. Nothing was screened."
  exit 1
fi

# =========================== stage 2: the screen =====================================
run_step 30_sv_screen $PY sv_screen.py --bundle "$BUNDLE" --meta results/v6_cub_meta.npz \
    --min-frac "$MIN_FRAC"
SCREEN_OK=$?

# =========================== stage 3: confirmation ===================================
# The unmodified instrument must reproduce the screen's ratios on the bundles it wrote.
if [ $SCREEN_OK -eq 0 ] && [ -s results/v6_confirm_list.txt ]; then
  # shellcheck disable=SC2046
  run_step 40_confirm_margins $PY group_margins.py --no-lp --tag v6_confirm_margins \
      --bundles $(cat results/v6_confirm_list.txt)
  [ $? -eq 0 ] && run_step 41_confirm_check $PY sv_screen.py \
      --compare results/v6_confirm_margins.json
else
  note "## 40_confirm_margins" "- skipped: the screen did not complete."
fi

# =========================== stage 4: replication ====================================
if [ $SCREEN_OK -eq 0 ] && [ -n "$TEST_BUNDLE" ] && [ -f "$TEST_BUNDLE" ]; then
  run_step 50_sv_screen_test $PY sv_screen.py --bundle "$TEST_BUNDLE" --split test \
      --meta results/v6_cub_meta.npz --min-frac "$MIN_FRAC" \
      --tag v6_sv_screen_test --no-confirm
  [ $? -eq 0 ] && run_step 51_replicate $PY sv_screen.py \
      --replicate results/v6_sv_screen_test.json
else
  note "## 50_sv_screen_test" "- skipped: no dinov2 test bundle found (set TEST_BUNDLE=...)," \
       "or the train screen did not complete. The train screen stands on its own."
fi

# =========================== wrap up =================================================
END_ALL=$(date +%s)
{
  echo "## WHAT TO DO NEXT"
  echo ""
  echo "Read in this order."
  echo ""
  echo "1. \`results/v6_cub_meta.md\` -- the verdict must be OK (every Waterbirds image"
  echo "   found in CUB, every attribute and part present exactly once). If not, nothing"
  echo "   below was run."
  echo "2. \`results/v6_confirm_check.md\` -- must say AGREEMENT OK. If not, the screen"
  echo "   is not measuring what group_margins.py measures and must not be read."
  echo "3. \`results/v6_sv_screen.md\` -- the Answer section, then the depletion table."
  echo "   Note |S| and the detectability floor at the top."
  echo "4. \`results/v6_replication.md\` -- only meaningful if step 3 has a hit. A hit"
  echo "   that does not replicate on the test split is not a finding."
  echo ""
  echo "Total wall clock: $(( (END_ALL - START_ALL) / 60 )) min."
  echo ""
} >> "$SUMMARY"

echo "done. Read $SUMMARY"
