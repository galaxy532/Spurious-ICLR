#!/usr/bin/env bash
# run_session4.sh -- the alpha > 1 hunt, by degrading the core feature in ONE GROUP.
#
#   cd /notebooks/Spurious-ICLR
#   nohup setsid bash run_session4.sh A > session4A_nohup.out 2>&1 &     # cheap, ~1 h
#   ... read results/v4_group_margins.md, choose levels ...
#   LEVELS="1.0 0.35 0.20" nohup setsid bash run_session4.sh B > session4B_nohup.out 2>&1 &
#
# WHY TWO STAGES
#   Stage A is a LADDER: it extracts frozen-DINOv2 features at several degradation
#   levels (train split only, no backbone training at all) and measures the per-group
#   hard margins of each. That is the quantity the branch condition is about, it costs
#   minutes per level, and it tells you BEFORE you spend six GPU-hours whether any
#   level has moved gamma_min/gamma_maj off 1. Session 3 could not settle this: the
#   iterate's own per-group margins were still spread 0.87 to 1.30 across the eps grid
#   at z = 1e6 and closing by only ~11% per half-decade.
#
#   Stage B is the EXPENSIVE part: full splits at the levels you chose, then
#   long_horizon.py at z = 1e6, which measures beta independently. The test is whether
#   beta of the larger-margin group matches the gamma ratio stage A measured.
#
# THE PREDICTION BEING TESTED (parameter-free)
#   Alignment identity, neurips_2026.tex: gamma_min = max(1, alpha_xi(1+delta_maj)
#   - delta_min) ~= max(1, alpha). So:
#     * stage A measures gamma_min/gamma_maj from the frozen features alone;
#     * Theorem 5.3 then says the LARGER-margin group's beta equals that number and
#       the other group's beta equals exactly 1;
#     * stage B measures both betas. Nothing is fitted in between.
#   A tie in stage A (ratio 1.000) predicts beta = 1 for both, i.e. the alpha < 1
#   branch and exactly what sessions 1-3 measured.
#
# WHY DEGRADE GROUP 0 BY DEFAULT
#   The bundle convention is g == 0 = G_maj, g == 1 = G_min, and Theorem 5.3 labels
#   G_min as the LARGER-margin group (WLOG gamma~_maj = 1, gamma~_min >= 1). Degrading
#   g == 0 therefore makes the script's "min" the theorem's "min", and the reports read
#   straight. Group SIZE is irrelevant to the branch -- eps is free in [0,1] and
#   long_horizon sweeps it by reweighting anyway.
#
# WHAT YOU GET, all under results/
#   results/logs/session4<stage>_<stamp>/SUMMARY.md   every step, command, duration,
#                                                     exit code, files produced, tail
#   results/v4_group_margins.{md,json}                stage A: the margin ladder
#   results/v4_group_margins_final.{md,json}          stage B: margins of the chosen
#                                                     levels, with the LP cross-check
#   results/waterbirds_degrade_v4.{md,json}           stage B: long_horizon tables
#   results/waterbirds_degrade_v4_curves.json         every recorded point
#
# Environment overrides:
#   LEVELS   space-separated degradation levels. Stage A default is the ladder below;
#            stage B has no default and MUST be set from stage A's table.
#   BACKBONE frozen backbone for the ladder (default dinov2; dinov2_l is 3x slower
#            per level but starts from a larger margin, 0.839 vs 0.551)
#   KIND     resolution (default) or blur
#   GROUP    which group is degraded (default 0)
#   MAX_HOURS cap on the long GD run (default 5.0; rerun the same command to resume)
#   PUSH=1   commit and push results/ at the end

set -u
cd "$(dirname "$0")"
PY=${PYTHON:-python}
STAGE=${1:-A}
BACKBONE=${BACKBONE:-dinov2}
KIND=${KIND:-resolution}
GROUP=${GROUP:-0}
MAX_HOURS=${MAX_HOURS:-5.0}
PREFIX=features_v4
STAMP=$(date +%Y%m%d_%H%M%S)
LOG=results/logs/session4${STAGE}_${STAMP}
mkdir -p "$LOG"
SUMMARY=$LOG/SUMMARY.md
START_ALL=$(date +%s)

# The stage-A ladder. 1.0 is the undegraded control and MUST stay in the list: it is
# the only row that says whether the pipeline reproduces session 3's tie.
if [ "$STAGE" = "A" ]; then
  LEVELS=${LEVELS:-"1.0 0.60 0.40 0.28 0.20 0.14 0.10"}
else
  LEVELS=${LEVELS:-""}
  if [ -z "$LEVELS" ]; then
    echo "stage B needs LEVELS, chosen from results/v4_group_margins.md." >&2
    echo 'e.g.  LEVELS="1.0 0.35 0.20" bash run_session4.sh B' >&2
    exit 2
  fi
fi

echo "# Session 4 stage $STAGE summary ($STAMP)" > "$SUMMARY"
{
  echo ""
  echo "Started $(date '+%Y-%m-%d %H:%M:%S %Z') on $(hostname)."
  echo ""
  echo "- backbone: \`$BACKBONE\`   degradation: \`$KIND\` on group \`g == $GROUP\`"
  echo "- levels: \`$LEVELS\`"
  echo ""
} >> "$SUMMARY"

{
  echo "== date";   date
  echo "== git";    git rev-parse HEAD 2>&1; git status --short 2>&1 | head -50
  echo "== python"; $PY --version 2>&1
  $PY -c "import numpy, sklearn; print('numpy', numpy.__version__, 'sklearn', sklearn.__version__)" 2>&1
  $PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())" 2>&1
  echo "== nvidia-smi"; nvidia-smi 2>&1
} > "$LOG/environment.txt" 2>&1

( nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw \
      --format=csv -l 60 > "$LOG/gpu.csv" 2>/dev/null ) &
MON_PID=$!

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
    tail -n 40 "$LOG/$name.log" 2>/dev/null
    echo '```'
    echo ""
  } >> "$SUMMARY"
  rm -f "$LOG/$name.files.before" "$LOG/$name.files.after"
  return $rc
}

lvl_flag() {   # "1.0" -> no flag at all, so the control is byte-identical to a plain run
  case "$1" in
    1.0|1|1.00) [ "$KIND" = "resolution" ] && echo "" || echo "--degrade-kind $KIND --degrade-level $1 --degrade-group $GROUP" ;;
    *) echo "--degrade-kind $KIND --degrade-level $1 --degrade-group $GROUP" ;;
  esac
}

# =========================== stage 0: self-checks =====================================
run_step 00_validate_degrade $PY degrade.py
DEG_OK=$?
run_step 01_validate_degrade_path $PY validate_degrade_path.py
PATH_OK=$?
run_step 02_validate_margins $PY validate_group_margins.py
MAR_OK=$?
if [ "$DEG_OK" -ne 0 ] || [ "$PATH_OK" -ne 0 ] || [ "$MAR_OK" -ne 0 ]; then
  { echo "## ABORTED"; echo ""; echo "A self-check failed. Nothing measured would mean anything, so no extraction was run."; } >> "$SUMMARY"
  kill $MON_PID 2>/dev/null
  echo "ABORTED: self-check failed. Read $SUMMARY"
  exit 1
fi

if [ "$STAGE" = "A" ]; then
  # ---------- stage A: the margin ladder, train split only, frozen backbone ----------
  run_step 03_smoke_degrade $PY extract_features.py --dataset waterbirds \
      --backbone "$BACKBONE" --degrade-kind "$KIND" --degrade-level 0.2 \
      --degrade-group "$GROUP" --splits train --out-prefix smoke4 --no-save-model \
      --max-batches 3
  rm -f smoke4_waterbirds_*.npz

  for L in $LEVELS; do
    # shellcheck disable=SC2046
    run_step 10_extract_$(echo "$L" | tr -d '.') $PY extract_features.py \
        --dataset waterbirds --backbone "$BACKBONE" --splits train \
        --out-prefix "$PREFIX" $(lvl_flag "$L")
  done

  run_step 20_group_margins $PY group_margins.py --no-lp --tag v4_group_margins \
      --bundles "${PREFIX}_waterbirds_${BACKBONE}"'*_train.npz'

  run_step 21_pick_levels $PY pick_levels.py --json results/v4_group_margins.json

  {
    echo "## WHAT TO DO NEXT"
    echo ""
    echo "\`21_pick_levels\` above applied the selection rules to the ladder and printed"
    echo "either the exact stage-B command to paste, or a recommendation NOT to run stage B."
    echo "Read that block: nothing else has to be decided by hand."
    echo ""
    echo "The full ladder, with the cross-check columns, is in \`results/v4_group_margins.md\`."
    echo "Two rows there are worth checking by eye whatever the picker says:"
    echo ""
    echo "- the \`level = 1.0\` CONTROL must report a **tie**. If it does not, something"
    echo "  changed since session 3 and stage B is premature whatever the other rows show."
    echo "- any row with \`plateau\` above 1e-2 is not a measurement. The picker drops those,"
    echo "  but if it dropped a row you wanted, rerun \`group_margins.py\` on that bundle"
    echo "  alone with a longer \`--C\` ladder rather than quoting it."
    echo ""
  } >> "$SUMMARY"

else
  # ---------- stage B: full splits at the chosen levels, then the long GD run --------
  for L in $LEVELS; do
    # shellcheck disable=SC2046
    run_step 30_extract_full_$(echo "$L" | tr -d '.') $PY extract_features.py \
        --dataset waterbirds --backbone "$BACKBONE" --splits train,val,test \
        --out-prefix "$PREFIX" $(lvl_flag "$L")
  done

  BUNDLES=""
  for L in $LEVELS; do
    case "$L" in
      1.0|1|1.00) [ "$KIND" = "resolution" ] && T="" || T="$($PY -c "import degrade,sys; print(degrade.tag('$KIND', float('$L')))")g$GROUP" ;;
      *) T="$($PY -c "import degrade,sys; print(degrade.tag('$KIND', float('$L')))")g$GROUP" ;;
    esac
    B="${PREFIX}_waterbirds_${BACKBONE}${T:+_$T}_train.npz"
    [ -e "$B" ] && BUNDLES="$BUNDLES $B" || echo "MISSING bundle for level $L: $B"
  done
  echo "bundles for the GD run:$BUNDLES"

  # Margins again on exactly the bundles that go into the GD run, this time with the
  # LP and logistic cross-checks. These are the numbers the prediction is read from.
  # shellcheck disable=SC2086
  run_step 31_group_margins_final $PY group_margins.py --tag v4_group_margins_final \
      --bundles $BUNDLES

  # shellcheck disable=SC2086
  run_step 40_degrade_speed $PY long_horizon.py --device cuda --T 20000000 \
      --max-hours "$MAX_HOURS" --bundles $BUNDLES --tag waterbirds_degrade_v4

  {
    echo "## WHAT TO DO NEXT"
    echo ""
    echo "If \`40_degrade_speed\` stopped on the time cap rather than finishing, rerun the"
    echo "SAME command and it resumes from saved state:"
    echo ""
    echo '```'
    echo "LEVELS=\"$LEVELS\" bash run_session4.sh B"
    echo '```'
    echo ""
    echo "When it has finished, the test is one comparison, per level:"
    echo ""
    echo "| from | quantity |"
    echo "|---|---|"
    echo "| \`results/v4_group_margins_final.md\` | \`beta predicted\` (= gamma_min/gamma_maj) and which group is larger |"
    echo "| \`results/waterbirds_degrade_v4.md\` | \`beta_min\` and \`beta_maj\` at the largest FULL-window z (not the half-window last row) |"
    echo ""
    echo "Theorem 5.3 predicts: the larger-margin group's beta equals the predicted value,"
    echo "the other group's beta equals 1, and NEITHER beta depends on eps. Apply the"
    echo "session-3 detector before claiming an escape: beta must exceed 1 by more than the"
    echo "eps-spread of that same beta, the other group must stay within that spread of 1,"
    echo "and both must hold at two consecutive z decades."
    echo ""
  } >> "$SUMMARY"
fi

kill $MON_PID 2>/dev/null
{
  echo "## Wall clock"
  echo ""
  echo "Total $(( ($(date +%s) - START_ALL) / 60 )) min."
  echo ""
} >> "$SUMMARY"

if [ "${PUSH:-0}" = "1" ]; then
  git add -A results/ >> "$LOG/push.log" 2>&1
  git commit -m "session 4 stage $STAGE: group-conditional degradation ($STAMP)" >> "$LOG/push.log" 2>&1
  git push >> "$LOG/push.log" 2>&1 || echo "push failed; see $LOG/push.log"
fi

echo "done. Read $SUMMARY"
