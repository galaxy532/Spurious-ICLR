#!/usr/bin/env bash
# run_session1.sh -- session 1 of the 17 Sept 2026 plan, unattended.
#
#   cd /notebooks/Spurious-ICLR          # the repo root, where the feature bundles are
#   nohup setsid bash run_session1.sh > session1_nohup.out 2>&1 &
#   (you can close the terminal and the browser tab)
#
# WHAT IT RUNS
#   0. checks (sequential)
#        long_horizon.py --validate-only --device cuda      (GPU GD engine vs reference)
#        extract_features.py smoke test on gdro_rn50         (5 batches, nothing saved)
#   then three streams IN PARALLEL:
#     GPU stream 1   points 1-2, Waterbirds: long_horizon.py on features_waterbirds_*_train.npz
#     GPU stream 2   points 3-4 features: extract_features.py for rwg_rn50, gdro_rn50,
#                    erm_rn50, dinov2 with --splits train,val,test --out-prefix features_v2
#     CPU stream     CelebA: celeba_subsample.py, sizes 5000,10000,20000, with separability
#   stream 1 is skipped if the GPU validation fails; the rwg/gdro extractions are skipped
#   if the smoke test fails (erm_rn50 and dinov2 are still attempted, and logged).
#   A failing step never stops the other steps.
#
# WHAT YOU GET, all under results/ (text files are git-trackable):
#   results/logs/session1_<stamp>/SUMMARY.md     one page: every step, command, start/end,
#                                                duration, exit code, the files it produced,
#                                                and the last lines of its log
#   results/logs/session1_<stamp>/<step>.log     full output of each step (progress bars
#                                                collapsed to their final state)
#   results/logs/session1_<stamp>/environment.txt  git commit, python/torch/GPU versions,
#                                                input bundle sizes and checksums
#   results/logs/session1_<stamp>/gpu.csv        GPU utilisation and memory every 60 s
#   results/logs/session1_<stamp>/bundles_v2.txt shape, eps, cells, standardisation flag of
#                                                every features_v2 bundle written
#   results/waterbirds_speed.{md,json}           long_horizon tables and numbers
#   results/waterbirds_speed_curves.json         every recorded point as JSON (the .npz
#                                                twin is git-ignored, the JSON is not)
#   results/celeba_sub_separability.{md,json}    CelebA subsample separability
#
# Optional:  PUSH=1 bash run_session1.sh   commits results/ and pushes at the end
#            (never prompts for credentials; a failed push is logged, not fatal).
#            MAX_HOURS=5.0 (default) caps the long GD run so it saves state before the
#            6 h Paperspace limit; rerunning long_horizon with the same command resumes.

set -u
cd "$(dirname "$0")"
PY=${PYTHON:-python}
MAX_HOURS=${MAX_HOURS:-5.0}
STAMP=$(date +%Y%m%d_%H%M%S)
LOG=results/logs/session1_${STAMP}
mkdir -p "$LOG"
SUMMARY=$LOG/SUMMARY.md
START_ALL=$(date +%s)

echo "# Session 1 summary ($STAMP)" > "$SUMMARY"
echo "" >> "$SUMMARY"
echo "Started $(date '+%Y-%m-%d %H:%M:%S %Z') on $(hostname)." >> "$SUMMARY"
echo "" >> "$SUMMARY"

# ---- environment record -----------------------------------------------------------
{
  echo "== date";   date
  echo "== git";    git rev-parse HEAD 2>&1; git status --short 2>&1 | head -50
  echo "== python"; $PY --version 2>&1
  $PY -c "import numpy, sklearn; print('numpy', numpy.__version__, 'sklearn', sklearn.__version__)" 2>&1
  $PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())" 2>&1
  echo "== nvidia-smi"; nvidia-smi 2>&1
  echo "== input bundles (size, md5 of first 1 MB)"
  for f in features_*_train.npz features_*_test.npz; do
    [ -e "$f" ] || continue
    printf "%s  %s  %s\n" "$(stat -c %s "$f")" "$(head -c 1048576 "$f" | md5sum | cut -c1-12)" "$f"
  done
} > "$LOG/environment.txt" 2>&1

# ---- GPU monitor ------------------------------------------------------------------
( nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw \
      --format=csv -l 60 > "$LOG/gpu.csv" 2>/dev/null ) &
MON_PID=$!

# ---- step runner ------------------------------------------------------------------
# run_step NAME COMMAND...   -> writes $LOG/NAME.log, appends a section to SUMMARY.md,
# returns the command's exit code. Progress bars still appear in session1_nohup.out;
# in the step log every carriage-return-redrawn bar is collapsed to its last state
# so the log stays readable.
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
  local sect="$LOG/$name.section"
  {
    echo "## $name"
    echo ""
    echo "- command: \`$*\`"
    echo "- start: $(date -d @$t0 '+%H:%M:%S'), end: $(date -d @$t1 '+%H:%M:%S'), duration: $(( (t1 - t0) / 60 )) min $(( (t1 - t0) % 60 )) s"
    echo "- exit code: **$rc** $( [ "$rc" -eq 0 ] && echo '(ok)' || echo '(FAILED)')"
    echo "- log: \`$LOG/$name.log\`"
    echo "- files that appeared in results/ during this step (streams run in parallel, so another stream may have written some): $(comm -13 "$LOG/$name.files.before" "$LOG/$name.files.after" | tr '\n' ' ')"
    echo ""
    echo '```'
    tail -n 40 "$LOG/$name.log" 2>/dev/null
    echo '```'
    echo ""
  } > "$sect"
  cat "$sect" >> "$SUMMARY" && rm -f "$sect"
  rm -f "$LOG/$name.files.before" "$LOG/$name.files.after"
  return $rc
}

# ---- 0. checks --------------------------------------------------------------------
run_step 00_validate_gd $PY long_horizon.py --validate-only --device cuda
VALID=$?
run_step 01_smoke_extract $PY extract_features.py --dataset waterbirds --backbone gdro_rn50 \
    --epochs 1 --max-batches 5 --splits train --out-prefix smoke --no-save-model
SMOKE=$?
rm -f smoke_waterbirds_gdro_rn50_*.npz

# ---- GPU stream 1: points 1-2 on Waterbirds -----------------------------------------
stream_gpu1() {
  if [ "$VALID" -ne 0 ]; then
    echo "## 10_waterbirds_speed" >> "$SUMMARY"
    echo "" >> "$SUMMARY"
    echo "SKIPPED: GPU validation failed (see 00_validate_gd)." >> "$SUMMARY"
    echo "" >> "$SUMMARY"
    return
  fi
  run_step 10_waterbirds_speed $PY long_horizon.py --device cuda \
      --bundles 'features_waterbirds_*_train.npz' --tag waterbirds_speed \
      --max-hours "$MAX_HOURS"
}

# ---- GPU stream 2: features for points 3 and 4 --------------------------------------
stream_gpu2() {
  for B in rwg_rn50 gdro_rn50 erm_rn50 dinov2; do
    if [ "$SMOKE" -ne 0 ] && { [ "$B" = rwg_rn50 ] || [ "$B" = gdro_rn50 ]; }; then
      { echo "## 20_extract_$B"; echo ""; echo "SKIPPED: smoke test failed (see 01_smoke_extract)."; echo ""; } >> "$SUMMARY"
      continue
    fi
    run_step 20_extract_$B $PY extract_features.py --dataset waterbirds --backbone "$B" \
        --splits train,val,test --out-prefix features_v2
  done
  run_step 29_check_v2_bundles $PY - <<'EOF'
import glob, numpy as np
from common import FeatureBundle
for p in sorted(glob.glob("features_v2_waterbirds_*.npz")):
    fb = FeatureBundle.load(p)
    cells = {f"y={yy},g={gg}": int(np.sum((fb.y == yy) & (fb.g == gg)))
             for yy in (-1, 1) for gg in (0, 1)}
    with np.load(p, allow_pickle=True) as z:
        has_stats = "std_mu" in z.files
    print(f"{p}: phi {fb.phi.shape}, eps {np.mean(fb.g == 1):.4f}, cells {cells}, "
          f"standardized_with={fb.meta.get('standardized_with')}, train_mode="
          f"{fb.meta.get('train_mode')}, std stats stored={has_stats}, "
          f"finite={bool(np.isfinite(fb.phi).all())}")
EOF
  cp "$LOG/29_check_v2_bundles.log" "$LOG/bundles_v2.txt" 2>/dev/null
}

# ---- CPU stream: CelebA subsample + separability ------------------------------------
stream_cpu() {
  run_step 30_celeba_subsample $PY celeba_subsample.py \
      --bundles 'features_celeba_*_train.npz' --sizes 5000,10000,20000
}

stream_gpu1 & P1=$!
stream_gpu2 & P2=$!
stream_cpu  & P3=$!
wait $P1 $P2 $P3

# ---- curves npz -> json (the npz files are git-ignored) ----------------------------
run_step 40_curves_to_json $PY - <<'EOF'
import glob, json, numpy as np
for p in sorted(glob.glob("results/*_curves.npz")):
    z = np.load(p, allow_pickle=False)
    out = {k: z[k].tolist() for k in z.files}
    q = p[:-4] + ".json"
    with open(q, "w") as f:
        json.dump(out, f)
    print(f"{p} -> {q} ({len(z.files)} arrays)")
EOF

kill "$MON_PID" 2>/dev/null

END_ALL=$(date +%s)
{
  echo "---"
  echo ""
  echo "Finished $(date '+%Y-%m-%d %H:%M:%S %Z'), total $(( (END_ALL - START_ALL) / 60 )) min."
  echo ""
  echo "Result files for review: results/waterbirds_speed.{md,json}, results/waterbirds_speed_curves.json,"
  echo "results/celeba_sub_separability.{md,json}, $LOG/bundles_v2.txt, and this file."
} >> "$SUMMARY"

if [ "${PUSH:-0}" = "1" ]; then
  {
    git add results/ && git commit -m "session 1 results ($STAMP)" && \
    GIT_TERMINAL_PROMPT=0 timeout 180 git push
  } > "$LOG/git_push.log" 2>&1 && echo "pushed" || echo "push failed: see $LOG/git_push.log"
fi
echo "done. Read $SUMMARY"
