from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from . import __version__
from .engine import QMEMMAEngine


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def read_extern_input(path: Path):
    """Read the simple key/value input deck produced by Amber EXTERN."""
    opts = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        opts[parts[0].lower()] = parts[1].strip() if len(parts) > 1 else ""
    return opts


def read_xyz(path: Path):
    lines = path.read_text().splitlines()
    n = int(lines[0].split()[0])
    symbols = []
    xyz = np.empty((n, 3), dtype=float)
    j = 0
    for line in lines[2:]:
        if not line.strip():
            continue
        p = line.split()
        if len(p) < 4:
            continue
        symbols.append(p[0])
        xyz[j] = (float(p[1]), float(p[2]), float(p[3]))
        j += 1
        if j == n:
            break
    if j != n:
        raise RuntimeError(f"{path}: expected {n} atoms, parsed {j}")
    return symbols, xyz


def read_point_charges(path: Path):
    lines = path.read_text().splitlines()
    n = int(lines[0].split()[0])
    q = np.empty(n, dtype=float)
    xyz = np.empty((n, 3), dtype=float)
    j = 0
    for line in lines[1:]:
        if not line.strip():
            continue
        p = line.split()
        if len(p) < 4:
            continue
        q[j] = float(p[0])
        xyz[j] = (float(p[1]), float(p[2]), float(p[3]))
        j += 1
        if j == n:
            break
    if j != n:
        raise RuntimeError(f"{path}: expected {n} charges, parsed {j}")
    return q, xyz


def _f3(v):
    return f"{float(v[0]):16.10f} {float(v[1]):16.10f} {float(v[2]):16.10f} "


def write_amber_output_stream(handle, energy, gqm, gmm):
    """Write the output format expected by Amber's EXTERN interface."""
    handle.write(f"FINAL ENERGY: {float(energy):.14f} a.u.\n")
    handle.write("        dE/dX            dE/dY            dE/dZ\n")
    for v in np.asarray(gqm):
        handle.write(_f3(v) + "\n")
    if len(gmm):
        handle.write("        MM / Point charge part\n")
        for v in np.asarray(gmm):
            handle.write(_f3(v) + "\n")


def engine_signature(symbols, basis, method, charge, spin, conv_tol, max_cycle, grid_level):
    return (
        tuple(symbols),
        str(basis).lower(),
        str(method).upper(),
        int(charge),
        int(spin),
        float(conv_tol),
        int(max_cycle),
        int(grid_level),
        _env("QM_EMMA_USE_DF", "1"),
        _env("QM_EMMA_DIIS_SPACE", ""),
        _env("QM_EMMA_DM_PREDICTOR", "previous"),
    )


class QMEMMADaemon:
    def __init__(self, args):
        self.args = args
        self.engine = None
        self.signature = None
        self.prev_nmm = None
        self.rows = []
        self.started = time.perf_counter()
        self.result_fifo = Path(args.result_fifo)
        self.error_file = Path(args.error_file)
        self.metrics_path = Path(args.metrics)
        self.live_path = Path(args.live)
        self.checkpoint_stride = int(_env("QM_EMMA_CHECKPOINT_STRIDE", "100"))
        self.live_stride = int(_env("QM_EMMA_LIVE_STRIDE", "10"))

    def make_engine(self, symbols, q, basis, method, charge, spin, conv_tol, max_cycle, grid_level):
        use_df = _env("QM_EMMA_USE_DF", "1").lower() not in {"0", "false", "no"}
        diis_s = _env("QM_EMMA_DIIS_SPACE", "").strip()
        predictor = _env("QM_EMMA_DM_PREDICTOR", "previous")
        sync = _env("QM_EMMA_SYNC_TIMING", "0").lower() in {"1", "true", "yes"}
        return QMEMMAEngine(
            symbols=symbols,
            mm_charges=q,
            basis=basis,
            xc=method,
            charge=charge,
            spin=spin,
            conv_tol=conv_tol,
            max_cycle=max_cycle,
            grid_level=grid_level,
            verbose=0,
            use_df=use_df,
            diis_space=(int(diis_s) if diis_s else None),
            dm_predictor=predictor,
            sync_timing=sync,
        )

    def handle(self, token, workdir, input_name):
        t_req = time.perf_counter()
        wd = Path(workdir)
        inp = Path(input_name)
        if not inp.is_absolute():
            inp = wd / inp

        t0 = time.perf_counter()
        opts = read_extern_input(inp)
        crdfile = opts.get("coordinates")
        if not crdfile:
            raise RuntimeError("Amber EXTERN input has no coordinates entry")
        crdpath = Path(crdfile)
        if not crdpath.is_absolute():
            crdpath = wd / crdpath
        symbols, qm = read_xyz(crdpath)

        ptfile = opts.get("pointcharges")
        if ptfile:
            ptpath = Path(ptfile)
            if not ptpath.is_absolute():
                ptpath = wd / ptpath
            q, mm = read_point_charges(ptpath)
        else:
            q = np.zeros(0, dtype=float)
            mm = np.zeros((0, 3), dtype=float)
        parse_s = time.perf_counter() - t0

        run = opts.get("run", "gradient").lower()
        if run not in ("gradient", "energy"):
            raise RuntimeError(f"Unsupported EXTERN run type {run!r}")

        charge = int(opts.get("charge", "0").split()[0])
        multiplicity = int(opts.get("spinmult", "1").split()[0])
        spin = multiplicity - 1
        method = opts.get("method", "pbe").split()[0]
        basis = opts.get("basis", "def2-svp").split()[0]

        conv_tol = float(_env("QM_EMMA_CONV_TOL", "1e-9"))
        max_cycle = int(_env("QM_EMMA_MAX_CYCLE", "100"))
        grid_level = int(_env("QM_EMMA_GRID_LEVEL", "2"))

        sig = engine_signature(
            symbols, basis, method, charge, spin, conv_tol, max_cycle, grid_level
        )
        mode = "CONTINUE"
        if self.engine is None or sig != self.signature:
            self.engine = self.make_engine(
                symbols, q, basis, method, charge, spin,
                conv_tol, max_cycle, grid_level
            )
            self.signature = sig
            use_dm = False
            mode = "NEW_CONDITION"
        else:
            # A varying number of MM point charges does not invalidate the
            # previous QM density as long as the QM atom list is unchanged.
            self.engine.mm_charges = q
            use_dm = True

        nmm = int(len(q))
        delta = None if self.prev_nmm is None else nmm - self.prev_nmm

        result = self.engine.compute(qm, mm, use_previous_density=use_dm)

        t0 = time.perf_counter()
        with self.result_fifo.open("w") as handle:
            write_amber_output_stream(
                handle, result.energy, result.grad_qm, result.grad_mm
            )
        output_s = time.perf_counter() - t0

        row = {
            "request": len(self.rows),
            "mode": mode,
            "n_qm": int(len(qm)),
            "n_mm": nmm,
            "delta_n_mm": delta,
            "n_mm_changed": bool(delta not in (None, 0)),
            "parse_input_s": parse_s,
            "molecule_update_s": result.molecule_update_seconds,
            "mf_setup_s": result.mf_setup_seconds,
            "scf_s": result.scf_seconds,
            "scf_cycles": result.scf_cycles,
            "qm_grad_s": result.qm_grad_seconds,
            "mm_grad_s": result.mm_grad_seconds,
            "d2h_s": result.d2h_seconds,
            "compute_total_s": result.total_seconds,
            "format_output_s": output_s,
            "daemon_request_total_s": time.perf_counter() - t_req,
            "energy_Eh": result.energy,
            "density_guess": result.density_guess,
            "fallback_used": result.fallback_used,
            "use_df": self.engine.use_df,
            "dm_predictor": self.engine.dm_predictor,
            "diis_space": self.engine.diis_space,
            "sync_timing": self.engine.sync_timing,
            "conv_tol": self.engine.conv_tol,
            "max_cycle": self.engine.max_cycle,
            "grid_level": self.engine.grid_level,
        }
        self.rows.append(row)
        self.prev_nmm = nmm

        n = len(self.rows)
        if self.live_stride > 0 and n % self.live_stride == 0:
            self.write_live(row)
        if self.checkpoint_stride > 0 and n % self.checkpoint_stride == 0:
            self.write_metrics(checkpoint=True)
        if n == 1 or n % 100 == 0:
            print(
                f"QM_EMMA_STEP {n:6d} nQM={len(qm):4d} nMM={nmm:5d} "
                f"dMM={delta} cycles={result.scf_cycles} "
                f"compute={result.total_seconds:.3f}s "
                f"request={row['daemon_request_total_s']:.3f}s",
                flush=True,
            )

    def write_live(self, row):
        tmp = self.live_path.with_suffix(self.live_path.suffix + ".tmp")
        tmp.write_text(json.dumps({"requests": len(self.rows), "last": row}, indent=2))
        os.replace(tmp, self.live_path)

    def write_metrics(self, checkpoint=False):
        payload = {
            "program": "qm_emma",
            "version": __version__,
            "requests": len(self.rows),
            "wall_daemon_s": time.perf_counter() - self.started,
            "settings": {
                "use_df": _env("QM_EMMA_USE_DF", "1"),
                "dm_predictor": _env("QM_EMMA_DM_PREDICTOR", "previous"),
                "diis_space": _env("QM_EMMA_DIIS_SPACE", ""),
                "sync_timing": _env("QM_EMMA_SYNC_TIMING", "0"),
                "conv_tol": _env("QM_EMMA_CONV_TOL", "1e-9"),
                "max_cycle": _env("QM_EMMA_MAX_CYCLE", "100"),
                "grid_level": _env("QM_EMMA_GRID_LEVEL", "2"),
            },
            "rows": self.rows,
        }
        target = (
            self.metrics_path
            if not checkpoint
            else self.metrics_path.with_name(self.metrics_path.stem + "_checkpoint.json")
        )
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, target)

        csv_path = (
            self.metrics_path.with_suffix(".csv")
            if not checkpoint
            else self.metrics_path.with_name(self.metrics_path.stem + "_checkpoint.csv")
        )
        if self.rows:
            tmpcsv = csv_path.with_suffix(csv_path.suffix + ".tmp")
            with tmpcsv.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(self.rows[0].keys()))
                writer.writeheader()
                writer.writerows(self.rows)
            os.replace(tmpcsv, csv_path)


def main():
    parser = argparse.ArgumentParser(
        description="Persistent qm_emma GPU4PySCF daemon for Amber EXTERN"
    )
    parser.add_argument("--request-fifo", required=True)
    parser.add_argument("--done-fifo", required=True)
    parser.add_argument("--result-fifo", required=True)
    parser.add_argument("--error-file", required=True)
    parser.add_argument("--ready-file", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--live", required=True)
    args = parser.parse_args()

    req_fifo = Path(args.request_fifo)
    done_fifo = Path(args.done_fifo)
    daemon = QMEMMADaemon(args)

    Path(args.ready_file).write_text(
        json.dumps(
            {
                "program": "qm_emma",
                "version": __version__,
                "pid": os.getpid(),
                "request_fifo": str(req_fifo.resolve()),
                "done_fifo": str(done_fifo.resolve()),
                "mode": "persistent_direct_fifo",
            },
            indent=2,
        )
    )
    print(f"QM_EMMA_DAEMON_READY pid={os.getpid()} version={__version__}", flush=True)

    stopping = False
    while not stopping:
        with req_fifo.open("r") as req_handle:
            for raw in req_handle:
                line = raw.rstrip("\n")
                if not line:
                    continue
                if line == "__STOP__":
                    stopping = True
                    break

                parts = line.split("\t", 2)
                if len(parts) != 3:
                    print(f"BAD_REQUEST_LINE {line!r}", file=sys.stderr, flush=True)
                    continue

                token, workdir, inp = parts
                status = "OK"
                try:
                    daemon.handle(token, workdir, inp)
                except Exception as exc:
                    status = "ERROR"
                    daemon.error_file.write_text(traceback.format_exc())
                    # Release the shim blocked on the result FIFO. The ERROR
                    # status then makes the shim return non-zero to Amber.
                    with daemon.result_fifo.open("w"):
                        pass
                    print(
                        f"REQUEST_ERROR token={token} {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )

                with done_fifo.open("w") as done_handle:
                    done_handle.write(f"{token}\t{status}\n")
                    done_handle.flush()

    daemon.write_metrics(checkpoint=False)
    print(f"QM_EMMA_DAEMON_STOPPED requests={len(daemon.rows)}", flush=True)


if __name__ == "__main__":
    main()
