# QM_eMMa (`qm_emma`)

`qm_emma` is a persistent GPU4PySCF backend for **AMBER QM/MM** calculations
using AMBER's `EXTERN` interface.

The design goal is simple: AMBER remains responsible for molecular-mechanics
dynamics and QM/MM bookkeeping, while one long-lived Python/GPU process
performs the electronic-structure calculations. Reusing the previous converged
density matrix and avoiding a new Python/GPU4PySCF process at every MD step
removes avoidable per-step overhead.

> **Naming:** the project/manuscript name is **QM_eMMa**; the repository,
> Python package and command-line tools use the lowercase name `qm_emma`.

## Repository layout

```text
qm_emma/
├── README.md
├── pyproject.toml
├── requirements.txt              # default CUDA 12 bootstrap
├── requirements/
│   ├── cuda11.txt
│   ├── cuda12.txt
│   └── cuda13.txt
├── scripts/
│   ├── create_venv.sh
│   ├── check_environment.py
│   ├── capture_environment.sh
│   └── init_run_directory.sh
├── bin/
│   └── qm_emma                    # lightweight AMBER EXTERN shim
├── qm_emma/
│   ├── __init__.py
│   ├── daemon.py                  # persistent FIFO daemon
│   └── engine.py                  # GPU4PySCF electronic-structure engine
└── examples/
    ├── amber_extern_job_fragment.sh
    ├── local_water_test/              # 5-step local GPU smoke test
    │   ├── .gitignore
    │   ├── README.md
    │   ├── build_system.sh
    │   ├── qmmm.in
    │   ├── run_local.sh
    │   └── water_seed.pdb
    ├── slurm_water_test/              # same test through sbatch/Slurm
    │   ├── .gitignore
    │   ├── README.md
    │   ├── build_system.sh
    │   ├── qmmm.in
    │   ├── submit.slurm
    │   └── water_seed.pdb
    └── quickstart/
        ├── README.md
        ├── qmmm.in.template
        ├── run_qm_emma.sh
        └── qm_emma_bridge.sh
```

There is deliberately **no tracked executable named `terachem`** in the
repository.

## 1. Pre-installation requirements

Before installing `qm_emma`, the machine must provide:

- Linux;
- an NVIDIA GPU supported by GPU4PySCF;
- a working NVIDIA driver and CUDA runtime/toolkit appropriate for the chosen
  GPU4PySCF binary package;
- Python >= 3.10 with the standard `venv` module;
- AMBER with `sander` and QM/MM `EXTERN` support.

Check the GPU/driver first:

```bash
nvidia-smi
```

If the CUDA toolkit compiler is available, also record:

```bash
nvcc --version
```

GPU4PySCF distributes CUDA-specific binary packages. The official GPU4PySCF
installation currently distinguishes CUDA 11.x, CUDA 12.x and CUDA 13.x. Do
not mix CuPy/cuTENSOR packages from different CUDA families.

### Recommended: create a dedicated venv automatically

From the repository root:

```bash
./scripts/create_venv.sh --cuda 12
```

By default this creates:

```text
$HOME/.venvs/qm_emma
```

A different location can be requested with:

```bash
./scripts/create_venv.sh --venv $HOME/venvs/qm_emma --cuda 12
```

Use `--cuda 11` or `--cuda 13` on those software stacks. `--cuda auto` attempts
to detect the CUDA family from `nvcc` and then, if necessary, `nvidia-smi`.
Explicit selection is preferable on HPC systems that expose several CUDA
modules/toolkits.

After creation, activate the environment in every new shell/job:

```bash
source $HOME/.venvs/qm_emma/bin/activate
```

A shell script cannot leave a venv activated in its parent shell, so this
activation command is intentionally explicit.

### Optional full GPU smoke test

The installation script performs package/import checks that also work on an
HPC login node without a visible GPU. Inside a GPU allocation, perform the
full CUDA/device check and then, optionally, a tiny GPU4PySCF QM/MM
energy-and-gradient calculation:

```bash
python scripts/check_environment.py
python scripts/check_environment.py --smoke
```

This tests the part of the stack that `qm_emma` actually uses, including the
GPU4PySCF QM/MM interface and MM point-charge gradients.

### Manual installation

For CUDA 12.x:

```bash
python3 -m venv $HOME/.venvs/qm_emma
source $HOME/.venvs/qm_emma/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir -r requirements/cuda12.txt
python -m pip install -e . --no-deps
python scripts/check_environment.py --imports-only
```

Equivalent files are provided for CUDA 11.x and 13.x.

The root `requirements.txt` selects CUDA 12.x because CUDA 12 was used for the
initial validated `qm_emma` development/benchmark environment. The
CUDA-specific files are bootstrap requirements rather than immutable lock
files. For a reproducible calculation, save the exact installed environment:

```bash
./scripts/capture_environment.sh
```

which produces `qm_emma_environment.txt` by default.

## 2. Confirm AMBER and qm_emma in the job environment

AMBER and the Python venv must both be active in the shell that launches the
calculation.

Typical HPC pattern:

```bash
set +u
source /path/to/amber.sh
source $HOME/.venvs/qm_emma/bin/activate
set -u
```

The temporary `set +u` is useful on installations whose AMBER setup script
references shell variables that may be unset.

Verify:

```bash
command -v sander
qm_emma --version
qm_emma-daemon --help
python scripts/check_environment.py
```

### Minimal end-to-end water-box examples

Two deliberately small examples are provided under `examples/` so the AMBER
`EXTERN` bridge can be tested before preparing a real system. Both create a
periodic TIP3P water box with `tleap`, select residue `:1` (one neutral H2O) as
the QM region, leave the remaining waters as MM point charges, and run only
**5 QM/MM MD steps** with PBE/def2-SVP. These are integration/smoke tests, not
production simulation protocols.

For a local Linux GPU machine:

```bash
cd examples/local_water_test
./build_system.sh
AMBER_SETUP=/path/to/amber.sh \
QM_EMMA_VENV=$HOME/.venvs/qm_emma \
./run_local.sh
```

If AMBER and the `qm_emma` venv are already active, simply run
`./run_local.sh`; it will also build the water box automatically if
`system.prmtop` and `start.rst7` are absent.

For a Slurm/sbatch cluster:

```bash
cd examples/slurm_water_test
sbatch --export=ALL,AMBER_SETUP=/path/to/amber.sh,QM_EMMA_VENV=$HOME/.venvs/qm_emma submit.slurm
```

The Slurm script requests one GPU and is intentionally generic. Adjust the
`#SBATCH` resource lines and module/environment setup to the local scheduler.
If FIFO traffic on the submission filesystem is undesirable, set
`QM_EMMA_STATE_DIR` to node-local scratch. Each example has its own README with
the expected output and diagnostic files.

## 3. Prepare a QM/MM calculation directory

For every independent calculation, use a self-contained run directory. The
minimal recommended layout is:

```text
my_qmmm_run/
├── system.prmtop          # required: AMBER topology/parameters
├── start.rst7             # required: initial restart/coordinates
├── qmmm.in                # required: MD + QM/MM settings
├── run_qm_emma.sh         # required: launches daemon and sander
└── qm_emma_bridge.sh      # required: daemon/FIFO setup
```

Optional system/protocol files can be added alongside these, for example:

```text
plumed.dat                 # optional enhanced sampling / CVs
restraints.RST             # optional AMBER restraint input
reference.rst7             # optional reference coordinates
```

The runtime creates:

```text
.qm_emma/
├── daemon.log
├── metrics.json
├── metrics.csv
├── live.json
├── ready.json
└── transient FIFO files
```

The FIFO files are job-local communication objects. **Do not copy a stale
`.qm_emma/` directory between independent jobs.** It is safe to archive the
JSON/CSV/log diagnostics, but the bridge should create fresh FIFOs each time.
The previous SCF density is currently persistent within one daemon/job, not
across separate scheduler jobs.

### Create the directory automatically

The repository includes a small initializer:

```bash
./scripts/init_run_directory.sh \
  my_qmmm_run \
  /path/to/system.prmtop \
  /path/to/start.rst7
```

It copies the topology/restart and creates:

```text
qmmm.in
run_qm_emma.sh
qm_emma_bridge.sh
README_RUN.txt
```

The generated `qmmm.in` is intentionally a **10-step smoke-test template**, not
a production protocol.

## 4. Define the system-specific QM/MM model

Before running, edit `qmmm.in`. A schematic input is:

```text
&cntrl
  ifqnt=1,
  ...
/
&qmmm
  qmmask='YOUR_QM_MASK',
  qmcharge=YOUR_QM_CHARGE,
  spin=1,
  qm_theory='EXTERN',
  qmmm_int=1,
  qmcut=10.0,
  qm_ewald=0,
  qmshake=0,
/
&tc
  method='pbe',
  basis='def2-svp',
/
```

The `&tc` namelist name is inherited from AMBER's existing `EXTERN` interface;
it does **not** imply that TeraChem is installed or used.

The user must determine the physical settings for the actual system. In
particular, review:

- QM atom selection (`qmmask`);
- QM charge and spin multiplicity;
- DFT functional and basis set;
- QM/MM electrostatic treatment and cutoff;
- periodic/electrostatic settings (`qm_ewald`, `qm_pme`, etc., when relevant);
- covalent QM/MM boundaries and link-atom treatment;
- charge redistribution near boundaries;
- timestep and SHAKE/QM-SHAKE choices;
- ensemble, thermostat and barostat;
- restraints;
- PLUMED/enhanced-sampling inputs, if used.

Do **not** copy these physical parameters blindly from another enzyme/system.

## 5. Validated qm_emma numerical defaults

Unless deliberately testing a different numerical protocol, the current
production defaults are:

```text
QM_EMMA_USE_DF=1
QM_EMMA_DM_PREDICTOR=previous
QM_EMMA_CONV_TOL=1e-9
QM_EMMA_GRID_LEVEL=2
QM_EMMA_MAX_CYCLE=100
```

The electronic method, basis set, QM charge and spin multiplicity are read from
the EXTERN input generated by AMBER. `qm_emma` therefore does not hard-code a
particular enzyme, QM region, functional or basis.

Optional runtime variables include:

```text
QM_EMMA_DIIS_SPACE
QM_EMMA_SYNC_TIMING=0
QM_EMMA_CHECKPOINT_STRIDE=100
QM_EMMA_LIVE_STRIDE=10
QM_EMMA_BRIDGE_TIMEOUT=900
```

`QM_EMMA_DM_PREDICTOR=linear` is experimental. The validated production setting
is `previous`.

## 6. Run the first calculation

Inside the prepared run directory:

```bash
source /path/to/amber.sh
source $HOME/.venvs/qm_emma/bin/activate
./run_qm_emma.sh
```

Alternatively, the generic runner accepts setup paths from the environment:

```bash
export AMBER_SETUP=/path/to/amber.sh
export QM_EMMA_VENV=$HOME/.venvs/qm_emma
./run_qm_emma.sh
```

The runner starts one persistent `qm_emma-daemon`, creates the runtime AMBER
compatibility link, launches `sander`, and stops the daemon when `sander`
finishes.

### Why a runtime compatibility symlink is required

The stock AMBER `EXTERN` route used by this workflow invokes the historical
executable name `terachem`. `qm_emma` does not use TeraChem. The command:

```bash
qm_emma install-amber-link "$PWD/.qm_emma/bin"
```

creates only a runtime symlink:

```text
.qm_emma/bin/terachem -> .../bin/qm_emma
```

There is no tracked TeraChem executable in the repository and no TeraChem
license is required.

## 7. Validate before production

Use a staged workflow for every new system:

```text
GPU4PySCF smoke test
        ↓
10-step AMBER/QM_eMMa test
        ↓
100-1000 step stability test
        ↓
production MD / enhanced sampling
```

After the 10-step test inspect:

```text
qmmm.out
qmmm.mdinfo
.qm_emma/daemon.log
.qm_emma/metrics.json
.qm_emma/error.log   # if present/non-empty
```

For a normal trajectory, after the initial cold calculation the metrics should
usually show reuse of the previous density (`density_guess` based on the
previous DM), converged SCF calculations and no systematic cold fallbacks.
Before production also inspect temperature/pressure/energy behavior, the QM
region and boundary geometry, and the chosen scientific observables.

A changing number of MM point charges does not by itself force `qm_emma` to
discard the previous QM density, provided the QM atom list and electronic
condition are unchanged.

## 8. Persistent-density and geometry policy

When the QM atom list and electronic settings remain unchanged, `qm_emma`
reuses the previous converged AO density as the initial guess for the next SCF
calculation. If the warm-started SCF does not converge, it automatically retries
from a cold initial guess. Failure after the cold fallback is propagated to
AMBER.

`qm_emma` caches the CPU `Mole`/basis definition and updates coordinates in
place, but deliberately constructs fresh mean-field, density-fitting and QM/MM
embedding objects at every geometry. This conservative choice avoids retaining
stale geometry-dependent density-fitting state.

## 9. Repeating and archiving jobs

For a new simulation segment or a new system:

1. create a new run directory;
2. copy the required starting topology/restart and protocol inputs;
3. activate the same recorded Python environment;
4. allow `qm_emma_bridge.sh` to create a fresh `.qm_emma/` runtime state;
5. capture the software/hardware environment for the production run;
6. archive the AMBER outputs together with `.qm_emma/metrics.json`,
   `.qm_emma/metrics.csv`, `.qm_emma/daemon.log` and the environment capture.

For continuation MD, use the previous segment's AMBER restart as the new
`start.rst7` and adjust the AMBER restart flags/input as usual. A new daemon
will begin with a cold SCF evaluation and then resume density reuse within that
job.

## License

QM_eMMa is intended for distribution under the Apache License 2.0. 
