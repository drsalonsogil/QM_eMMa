# Slurm water-box smoke test

This folder contains the same 5-step water-box QM/MM smoke test as the local
example, wrapped in a simple single-GPU Slurm job.

The builder creates a periodic TIP3P water box. Residue `:1` is one neutral QM
water molecule and the remaining waters are MM point charges. PBE/def2-SVP is
used only as a small integration test.

Typical use:

```bash
./build_system.sh
sbatch --export=ALL,AMBER_SETUP=/path/to/amber.sh,QM_EMMA_VENV=$HOME/.venvs/qm_emma submit.slurm
```

If `tleap` is not available on the login node, skip the first command: the job
will build the topology after the AMBER environment is loaded.

Adapt the `#SBATCH` GPU/resource directives and environment setup to your site.
On clusters where the submission directory is on network storage, you can keep
the FIFOs on node-local scratch with, for example:

```bash
sbatch --export=ALL,AMBER_SETUP=/path/to/amber.sh,QM_EMMA_VENV=$HOME/.venvs/qm_emma,QM_EMMA_STATE_DIR=/local/scratch/qm_emma-water submit.slurm
```

Expected diagnostics are `qmmm.out`, `qmmm.mdinfo`, the Slurm output file, and
`.qm_emma/metrics.json` / `.qm_emma/daemon.log`. If `QM_EMMA_STATE_DIR` points
to node-local scratch, the batch script copies these small diagnostics back to
this folder before the job exits.
