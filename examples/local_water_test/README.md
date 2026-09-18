# Local water-box smoke test

This is a deliberately tiny end-to-end AMBER/QM_eMMa test for a local Linux
machine with one NVIDIA GPU.

The builder creates a periodic TIP3P water box around one seed water molecule.
Residue `:1` (the seed H2O) is the QM region; all solvent waters are MM. The
input runs only 5 MD steps with PBE/def2-SVP, so it is intended to test the
EXTERN bridge, persistent daemon, gradients and density reuse rather than to
provide a production MD protocol.

From this directory, after AMBER and the `qm_emma` Python environment are
available:

```bash
./build_system.sh
./run_local.sh
```

Or let the runner source them for you:

```bash
AMBER_SETUP=/path/to/amber.sh \
QM_EMMA_VENV=$HOME/.venvs/qm_emma \
./run_local.sh
```

Expected diagnostic files include `qmmm.out`, `qmmm.mdinfo`,
`.qm_emma/daemon.log` and `.qm_emma/metrics.json`. After the first electronic
calculation, later requests should normally report reuse of the previous density
unless an SCF fallback is required.
