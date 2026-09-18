#!/usr/bin/env bash
# Run the minimal water-box QM_eMMa smoke test on a local GPU workstation.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# Optional convenience variables:
#   AMBER_SETUP=/path/to/amber.sh
#   QM_EMMA_VENV=$HOME/.venvs/qm_emma
if [[ -n "${AMBER_SETUP:-}" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "$AMBER_SETUP"
  set -u
fi
if [[ -n "${QM_EMMA_VENV:-}" ]]; then
  # shellcheck disable=SC1091
  source "$QM_EMMA_VENV/bin/activate"
fi

for exe in sander tleap qm_emma qm_emma-daemon; do
  command -v "$exe" >/dev/null 2>&1 || {
    echo "ERROR: $exe was not found in PATH." >&2
    exit 1
  }
done

[[ -f system.prmtop && -f start.rst7 ]] || ./build_system.sh

STATE_DIR="$HERE/.qm_emma"
mkdir -p "$STATE_DIR/bin"
rm -f "$STATE_DIR/request.fifo" "$STATE_DIR/done.fifo" \
      "$STATE_DIR/result.fifo" "$STATE_DIR/error.log" "$STATE_DIR/ready.json" \
      "$STATE_DIR/metrics.json" "$STATE_DIR/metrics.csv" "$STATE_DIR/live.json" \
      "$STATE_DIR/daemon.log"
mkfifo "$STATE_DIR/request.fifo" "$STATE_DIR/done.fifo" "$STATE_DIR/result.fifo"

qm_emma install-amber-link "$STATE_DIR/bin" >/dev/null
export PATH="$STATE_DIR/bin:$PATH"
export QM_EMMA_BRIDGE_REQ_FIFO="$STATE_DIR/request.fifo"
export QM_EMMA_BRIDGE_DONE_FIFO="$STATE_DIR/done.fifo"
export QM_EMMA_BRIDGE_RESULT_FIFO="$STATE_DIR/result.fifo"
export QM_EMMA_BRIDGE_ERROR_FILE="$STATE_DIR/error.log"
export QM_EMMA_BRIDGE_TIMEOUT="${QM_EMMA_BRIDGE_TIMEOUT:-900}"

qm_emma-daemon \
  --request-fifo "$STATE_DIR/request.fifo" \
  --done-fifo "$STATE_DIR/done.fifo" \
  --result-fifo "$STATE_DIR/result.fifo" \
  --error-file "$STATE_DIR/error.log" \
  --ready-file "$STATE_DIR/ready.json" \
  --metrics "$STATE_DIR/metrics.json" \
  --live "$STATE_DIR/live.json" \
  > "$STATE_DIR/daemon.log" 2>&1 &
DAEMON_PID=$!
export QM_EMMA_BRIDGE_PID="$DAEMON_PID"

cleanup() {
  if kill -0 "$DAEMON_PID" 2>/dev/null; then
    printf '__STOP__\n' > "$STATE_DIR/request.fifo" || true
    wait "$DAEMON_PID" || true
  fi
  rm -f "$STATE_DIR/request.fifo" "$STATE_DIR/done.fifo" "$STATE_DIR/result.fifo"
}
trap cleanup EXIT

for _ in $(seq 1 300); do
  [[ -s "$STATE_DIR/ready.json" ]] && break
  kill -0 "$DAEMON_PID" 2>/dev/null || {
    cat "$STATE_DIR/daemon.log" >&2
    exit 1
  }
  sleep 0.1
done
[[ -s "$STATE_DIR/ready.json" ]] || {
  echo "ERROR: qm_emma daemon did not become ready." >&2
  exit 1
}

sander -O \
  -i qmmm.in \
  -o qmmm.out \
  -p system.prmtop \
  -c start.rst7 \
  -r end.rst7 \
  -inf qmmm.mdinfo

echo "Smoke test finished. Inspect qmmm.out and .qm_emma/metrics.json."
