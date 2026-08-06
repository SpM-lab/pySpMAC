# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2024- SpM-lab
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from typing import Any, Dict, Tuple, Optional
import numpy as np
from numpy.typing import NDArray

from .util import dict_with_lowerkey
from .solver.base import SolverBase
from .solver.single import Solver as SingleSolver
from .solver.multiple import Solver as MultipleSolver
from .solver.pade import PadeAC


def optimize(
    solver: SolverBase,
    loglambdamin: float,
    loglambdamax: float,
    n_trials: int,
    verbose: bool = True,
) -> float:
    loglambdas = np.linspace(loglambdamin, loglambdamax, num=n_trials)
    fxs = np.zeros(n_trials, dtype=np.float64)
    for i, loglambda in enumerate(loglambdas):
        opt = solver.solve_one(10**loglambda)
        fxs[i] = np.log(opt([opt.x[0]]))
        if verbose:
            outdir = solver.outdir / str(i)
            outdir.mkdir(parents=True, exist_ok=True)
            solver.write_rhol(opt.x[0], loglambda=loglambda, outdir=outdir)
            rho = solver.predict_rho(opt.x[0])
            solver.write_rho(solver.ws, rho, loglambda=loglambda, outdir=outdir)

    gxs = np.zeros(n_trials, dtype=np.float64)
    with open(solver.outdir / "elbow.dat", "w") as output:
        coeff = (fxs[-1] - fxs[0]) / (loglambdamax - loglambdamin)
        # coeff = (np.log(fxs[-1]) - np.log(fxs[0])) / (loglambdamax - loglambdamin)
        for i, loglambda in enumerate(loglambdas):
            gx = coeff * (loglambda - loglambdamin) + fxs[0]
            # gx = coeff * (loglambda - loglambdamin) + np.log(fxs[0])
            gxs[i] = gx - fxs[i]
            # gxs[i] = gx - np.log(fxs[i])
            output.write(f"{loglambda} {fxs[i]} {gxs[i]}\n")
    i = np.argmax(gxs)
    return loglambdas[i]


def _load_gtau_from_file(
    filename: str, column: int, column_imag: Optional[int] = None
) -> NDArray:
    """Load G(τ) from a text file.

    If ``column_imag`` is given, return complex G = Re + i Im.
    Otherwise return the real column only (imaginary part is treated as 0).
    """
    data = np.loadtxt(filename)
    g = data[:, column]
    if column_imag is not None:
        g = g + 1j * data[:, column_imag]
    return g


def run(
    params: Dict[str, Any], Gtau: NDArray
) -> Tuple[SolverBase, NDArray, float]:
    params = dict_with_lowerkey(params)
    verbose = params.get("verbose", True)
    n_trials = params.get("num_trials", 10)
    nflavor = params.get("num_flavor", 1)
    force_multi = params.get("force_multi", 0)
    if nflavor == 1 and force_multi == 0:
        print("Using single")
        solver: SolverBase = SingleSolver(params, Gtau=Gtau)
    else:
        solver = MultipleSolver(params, Gtau=Gtau)

    beta: float = params["beta"]
    pade = PadeAC(Gtau, beta)
    rho_pade = pade(solver.ws)
    solver.write_rho(
        solver.ws, rho_pade, outdir=solver.outdir, filename="pade.dat", rescale_dw=True
    )

    if params.get("optimize", False):
        loglambdamax = params["max_loglambda"]
        loglambdamin = params["min_loglambda"]
        if loglambdamax < loglambdamin:
            loglambdamin, loglambdamax = loglambdamax, loglambdamin
        best_loglambda: float = optimize(
            solver,
            loglambdamin,
            loglambdamax,
            n_trials,
            verbose=verbose,
        )
    else:
        best_loglambda = params["loglambda"]
    opt = solver.solve_one(10**best_loglambda)
    rho_l = opt.x[0]
    return solver, rho_l, best_loglambda


def main():
    import sys
    import toml

    if len(sys.argv) == 1:
        inputfilename = "input.toml"
    else:
        inputfilename = sys.argv[1]
    with open(inputfilename, "r") as f:
        params = toml.load(f)
    params = dict_with_lowerkey(params)

    nflavor = params.get("num_flavor", 1)
    column: int = params["column"]
    column_imag: Optional[int] = params.get("column_imag")

    if column_imag is None:
        print(f"Loading real G_ij(τ) from {params["filein_g"]}.i_j column {column}")
    else:
        print(f"Loading complex G_ij(τ) from {params["filein_g"]}.i_j column {column} and {column_imag}")

    if nflavor == 1:
        Gtau = _load_gtau_from_file(params["filein_g"], column, column_imag)
    else:
        dtype = np.complex128 if column_imag is not None else np.float64
        Gtau = np.zeros((1, nflavor, nflavor), dtype=dtype)
        ntau = -1
        for i in range(nflavor):
            for j in range(nflavor):
                filename = params["filein_g"] + f".{i}_{j}"
                Gt = _load_gtau_from_file(filename, column, column_imag)
                if ntau < 0:
                    ntau = len(Gt)
                    Gtau = np.zeros((ntau, nflavor, nflavor), dtype=dtype)
                Gtau[:, i, j] = Gt[:]

    solver, rho_l, best_loglambda = run(params, Gtau)
    solver.write_rhol(rho_l, loglambda=best_loglambda, outdir=solver.outdir)
    rho = solver.predict_rho(rho_l)
    solver.write_rho(solver.ws, rho, loglambda=best_loglambda, outdir=solver.outdir)

    mean = solver.total_time / solver.total_niter
    std = solver.total_niter * (solver.total_time2 / solver.total_niter - mean ** 2) / (solver.total_niter - 1)
    with open(solver.outdir / "time.dat", "w") as f:
        f.write(f"{mean} {std} {solver.total_niter}\n")


if __name__ == "__main__":
    main()
