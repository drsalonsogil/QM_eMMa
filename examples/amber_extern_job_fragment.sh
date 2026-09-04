#!/usr/bin/env bash
# Generic qm_emma setup fragment for one Amber/sander job.
# Source/activate Amber, CUDA and the Python environment containing
# GPU4PySCF before this fragment.
set -euo pipefail

RUN_DIR="${RUN_DIR:-$PWD}"
STATE_DIR="${QM_EMMA_STATE_DIR:-$RUN_DIR/.qm_emma}"
mkdir -p "$STATE_DIR/bin"

# Stock Amber EXTERN still looks for the historical executable name.
# The repository contains no such program: create a runtime-only symlink to
# the real qm_emma shell shim and put it first in PATH.
qm_emma install-amber-link "$STATE_DIR/bin" >/dev/null
export PATH="$STATE_DIR/bin:$PATH"

# Validated qm_emma production defaults. Override before this fragment if needed.
export QM_EMMA_USE_DF="${QM_EMMA_USE_DF:-1}"
export QM_EMMA_DM_PREDICTOR="${QM_EMMA_DM_PREDICTOR:-previous}"
export QM_EMMA_CONV_TOL="${QM_EMMA_CONV_TOL:-1e-9}"
export QM_EMMA_GRID_LEVEL="${QM_EMMA_GRID_LEVEL:-2}"
export QM_EMMA_MAX_CYCLE="${QM_EMMA_MAX_CYCLE:-100}"
export QM_EMMA_SYNC_TIMING="${QM_EMMA_SYNC_TIMING:-0}"
export QM_EMMA_CHECKPOINT_STRIDE="${QM_EMMA_CHECKPOINT_STRIDE:-100}"
export QM_EMMA_LIVE_STRIDE="${QM_EMMA_LIVE_STRIDE:-10}"
export QM_EMMA_BRIDGE_TIMEOUT="${QM_EMMA_BRIDGE_TIMEOUT:-900}"

REQ="$STATE_DIR/request.fifo"
DONE="$STATE_DIR/done.fifo"
RESULT="$STATE_DIR/result.fifo"
ERR="$STATE_DIR/error.log"
READY="$STATE_DIR/ready.json"
METRICS="$STATE_DIR/metrics.json"
LIVE="$STATE_DIR/live.json"

rm -f "$REQ" "$DONE" "$RESULT" "$ERR" "$READY"
mkfifo "$REQ" "$DONE" "$RESULT"

export QM_EMMA_BRIDGE_REQ_FIFO="$REQ"
export QM_EMMA_BRIDGE_DONE_FIFO="$DONE"
export QM_EMMA_BRIDGE_RESULT_FIFO="$RESULT"
export QM_EMMA_BRIDGE_ERROR_FILE="$ERR"

qm_emma-daemon \
  --request-fifo "$REQ" \
  --done-fifo "$DONE" \
  --result-fifo "$RESULT" \
  --error-file "$ERR" \
  --ready-file "$READY" \
  --metrics "$METRICS" \
  --live "$LIVE" \
  > "$STATE_DIR/daemon.log" 2>&1 &

QM_EMMA_DAEMON_PID=$!
export QM_EMMA_BRIDGE_PID="$QM_EMMA_DAEMON_PID"

cleanup_qm_emma() {
  if kill -0 "$QM_EMMA_DAEMON_PID" 2>/dev/null; then
    printf '__STOP__\n' > "$REQ" || true
    wait "$QM_EMMA_DAEMON_PID" || true
  fi
  rm -f "$REQ" "$DONE" "$RESULT"
}
trap cleanup_qm_emma EXIT

# Wait for daemon readiness.
for _ in $(seq 1 300); do
  [[ -s "$READY" ]] && break
  kill -0 "$QM_EMMA_DAEMON_PID" 2>/dev/null || {
    cat "$STATE_DIR/daemon.log" >&2
    exit 1
  }
  sleep 0.1
done
[[ -s "$READY" ]] || { echo "qm_emma daemon did not become ready" >&2; exit 1; }

# Example Amber invocation; replace the filenames with the user's own files.
# "$AMBERHOME/bin/sander" -O \
#   -i qmmm.in -o qmmm.out -p system.prmtop -c start.rst7 \
#   -r end.rst7 -x trajectory.nc -inf qmmm.mdinfo
