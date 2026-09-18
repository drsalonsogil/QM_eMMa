#!/usr/bin/env bash
# Build a tiny periodic TIP3P water box. Residue :1 is the seed water and is
# used as the QM region by qmmm.in; all other waters remain MM point charges.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

command -v tleap >/dev/null 2>&1 || {
  echo "ERROR: tleap was not found in PATH. Source your AMBER environment first." >&2
  exit 1
}

cat > leap.in <<'LEAP'
source leaprc.water.tip3p
seed = loadPdb water_seed.pdb
solvateBox seed TIP3PBOX 8.0
check seed
saveAmberParm seed system.prmtop start.rst7
quit
LEAP

tleap -f leap.in > leap.out

echo "Created system.prmtop and start.rst7"
echo "QM region for the smoke test: residue :1 (one neutral H2O molecule)."
