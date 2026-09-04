from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass
from typing import Optional

import cupy as cp
import numpy as np
from pyscf import dft, gto
from gpu4pyscf.qmmm import itrf as gpu_qmmm


@dataclass
class EngineResult:
    energy: float
    grad_qm: np.ndarray
    grad_mm: np.ndarray
    converged: bool
    scf_cycles: Optional[int]
    molecule_update_seconds: float
    mf_setup_seconds: float
    scf_seconds: float
    qm_grad_seconds: float
    mm_grad_seconds: float
    d2h_seconds: float
    total_seconds: float
    density_guess: str
    fallback_used: bool
    warm_attempt_seconds: float
    warm_attempt_cycles: Optional[int]
    cold_attempt_seconds: float
    cold_attempt_cycles: Optional[int]


class QMEMMAEngine:
    """Persistent GPU4PySCF engine used by qm_emma.

    The engine is intentionally conservative with geometry-dependent objects:
    the CPU Mole/basis definition and converged density matrix are retained,
    while a fresh mean-field / density-fitting / QM-MM embedding object is
    constructed for every geometry.

    Defaults correspond to the validated qm_emma production configuration:
      - density fitting enabled
      - SCF convergence 1e-9 Eh
      - PySCF DFT grid level 2
      - maximum 100 SCF cycles
      - previous converged AO density reused as dm0
      - cold-SCF fallback if a warm start does not converge
    """

    def __init__(
        self,
        symbols,
        mm_charges,
        basis="def2-svp",
        xc="PBE",
        charge=0,
        spin=0,
        conv_tol=1e-9,
        max_cycle=100,
        grid_level=2,
        verbose=0,
        use_df=True,
        diis_space=None,
        dm_predictor="previous",
        sync_timing=False,
    ):
        self.symbols = list(symbols)
        self.mm_charges = np.asarray(mm_charges, dtype=float)
        self.basis = basis
        self.xc = xc
        self.charge = int(charge)
        self.spin = int(spin)
        self.conv_tol = float(conv_tol)
        self.max_cycle = int(max_cycle)
        self.grid_level = int(grid_level)
        self.verbose = int(verbose)
        self.use_df = bool(use_df)
        self.diis_space = None if diis_space is None else int(diis_space)
        self.dm_predictor = str(dm_predictor).lower()
        self.sync_timing = bool(sync_timing)

        self.previous_dm = None
        self.previous_previous_dm = None
        self.mol = None
        self.step = 0

        cp.cuda.Device().use()
        cp.zeros(1)
        cp.cuda.Stream.null.synchronize()

    def reset_density(self):
        self.previous_dm = None
        self.previous_previous_dm = None

    def _sync(self):
        if self.sync_timing:
            cp.cuda.Stream.null.synchronize()

    def _prepare_mol(self, qm_coords):
        coords = np.asarray(qm_coords, dtype=float)
        t0 = time.perf_counter()
        if self.mol is None:
            self.mol = gto.M(
                atom=[(s, xyz) for s, xyz in zip(self.symbols, coords)],
                basis=self.basis,
                charge=self.charge,
                spin=self.spin,
                unit="Angstrom",
                verbose=self.verbose,
            )
        else:
            self.mol.set_geom_(coords, unit="Angstrom", inplace=True)
        return self.mol, time.perf_counter() - t0

    def _build_mf(self, mol, mm_coords):
        t0 = time.perf_counter()
        mf = dft.RKS(mol) if self.spin == 0 else dft.UKS(mol)
        mf.xc = self.xc

        if self.use_df:
            # A fresh DF object is deliberately created for every geometry.
            # This avoids retaining geometry-dependent auxiliary-basis state.
            mf = mf.density_fit()

        mf = mf.to_gpu()

        mm_coords = np.asarray(mm_coords, dtype=float)
        if len(mm_coords):
            mf = gpu_qmmm.mm_charge(
                mf,
                mm_coords,
                self.mm_charges,
                unit="Angstrom",
            )

        mf.conv_tol = self.conv_tol
        mf.max_cycle = self.max_cycle
        mf.grids.level = self.grid_level
        if self.diis_space is not None:
            mf.diis_space = self.diis_space

        return mf, time.perf_counter() - t0

    @staticmethod
    def _cycles(mf):
        cycles = getattr(mf, "cycles", None)
        try:
            return int(cycles) if cycles is not None else None
        except Exception:
            return None

    def _initial_dm(self, use_previous_density):
        if not use_previous_density or self.previous_dm is None:
            return None, "cold"
        if self.dm_predictor == "linear" and self.previous_previous_dm is not None:
            return 2.0 * self.previous_dm - self.previous_previous_dm, "linear_dm"
        return self.previous_dm, "previous_dm"

    def _run_scf(self, mf, dm0=None):
        self._sync()
        t0 = time.perf_counter()
        energy = mf.kernel() if dm0 is None else mf.kernel(dm0=dm0)
        self._sync()
        seconds = time.perf_counter() - t0
        return energy, bool(mf.converged), self._cycles(mf), float(seconds)

    def compute(self, qm_coords, mm_coords, use_previous_density=True):
        ttotal = time.perf_counter()
        qm_coords = np.asarray(qm_coords, dtype=float)
        mm_coords = np.asarray(mm_coords, dtype=float)

        if len(qm_coords) != len(self.symbols):
            raise ValueError(
                f"QM atom count changed from {len(self.symbols)} to {len(qm_coords)}. "
                "A new qm_emma engine is required."
            )
        if len(mm_coords) != len(self.mm_charges):
            raise ValueError(
                f"MM coordinate/charge mismatch: {len(mm_coords)} coordinates, "
                f"{len(self.mm_charges)} charges."
            )

        mol, mol_s = self._prepare_mol(qm_coords)
        mf, setup_s = self._build_mf(mol, mm_coords)

        dm0, density_guess = self._initial_dm(use_previous_density)
        fallback_used = False
        warm_s = 0.0
        warm_cycles = None
        cold_s = 0.0
        cold_cycles = None

        energy, converged, cycles, this_s = self._run_scf(mf, dm0=dm0)
        if dm0 is None:
            cold_s, cold_cycles = this_s, cycles
        else:
            warm_s, warm_cycles = this_s, cycles

        if dm0 is not None and not converged and density_guess == "linear_dm":
            fallback_used = True
            del mf
            gc.collect()
            mol, extra_mol_s = self._prepare_mol(qm_coords)
            mf, extra_setup_s = self._build_mf(mol, mm_coords)
            mol_s += extra_mol_s
            setup_s += extra_setup_s
            energy, converged, cycles, retry_s = self._run_scf(
                mf, dm0=self.previous_dm
            )
            warm_s += retry_s
            warm_cycles = cycles
            density_guess = "linear_then_previous_dm"

        if dm0 is not None and not converged:
            fallback_used = True
            del mf
            gc.collect()
            mol, extra_mol_s = self._prepare_mol(qm_coords)
            mf, extra_setup_s = self._build_mf(mol, mm_coords)
            mol_s += extra_mol_s
            setup_s += extra_setup_s
            energy, converged, cold_cycles, cold_s = self._run_scf(mf, dm0=None)
            cycles = cold_cycles
            density_guess = density_guess + "_then_cold"

        if not converged:
            raise RuntimeError(
                f"SCF did not converge at qm_emma engine step {self.step}; "
                f"fallback_used={fallback_used}"
            )

        dm = mf.make_rdm1()

        self._sync()
        t0 = time.perf_counter()
        gobj = mf.nuc_grad_method()
        gq = gobj.kernel()
        self._sync()
        qmg_s = time.perf_counter() - t0

        self._sync()
        t0 = time.perf_counter()
        if len(mm_coords):
            gm = gobj.grad_hcore_mm(dm) + gobj.grad_nuc_mm()
        else:
            gm = np.zeros((0, 3), dtype=float)
        self._sync()
        mmg_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        if hasattr(gq, "get"):
            gq = gq.get()
        if hasattr(gm, "get"):
            gm = gm.get()
        gq = np.asarray(gq, dtype=float)
        gm = np.asarray(gm, dtype=float)
        d2h_s = time.perf_counter() - t0

        self.previous_previous_dm = self.previous_dm
        self.previous_dm = dm
        self.step += 1

        return EngineResult(
            energy=float(energy),
            grad_qm=gq,
            grad_mm=gm,
            converged=True,
            scf_cycles=cycles,
            molecule_update_seconds=float(mol_s),
            mf_setup_seconds=float(setup_s),
            scf_seconds=float(warm_s + cold_s),
            qm_grad_seconds=float(qmg_s),
            mm_grad_seconds=float(mmg_s),
            d2h_seconds=float(d2h_s),
            total_seconds=float(time.perf_counter() - ttotal),
            density_guess=density_guess,
            fallback_used=bool(fallback_used),
            warm_attempt_seconds=float(warm_s),
            warm_attempt_cycles=warm_cycles,
            cold_attempt_seconds=float(cold_s),
            cold_attempt_cycles=cold_cycles,
        )
