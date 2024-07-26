# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2024- SpM-lab
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from typing import Union, Any, Optional, Dict, List

import time
import pathlib
import itertools

import numpy as np
from numpy.typing import NDArray


import admmsolver
import admmsolver.optimizer
import admmsolver.objectivefunc
from admmsolver.optimizer import Problem, EqualityCondition
from admmsolver.objectivefunc import (
    L1Regularizer,
    LeastSquares,
    ConstrainedLeastSquares,
    SemiPositiveDefinitePenalty,
)
from admmsolver.matrix import identity, PartialDiagonalMatrix, DiagonalMatrix
from admmsolver.optimizer import SimpleOptimizer

import sparse_ir

from ..pade import Pade
from ..util import dict_with_lowerkey
from .base import SolverBase, _oversample


class Solver(SolverBase):
    nflavor: int
    __sumrule_weight: NDArray[np.float64]

    def __init__(
        self, params: Dict[str, Any], Gtau: Optional[NDArray[np.float64]]
    ) -> None:
        """
        Arguments
        -----------
        params: dict[str, Any]
            Note that parameter names are case-insensitive

            - num_flavor: int
                - Number of flavors (orbitals)
            - filein_g: str
                - Filename storing G_{ab}(τ)
            - column: int
                - Index of column storing G_{ab}(τ) (0-origin)

            See base.SolverBase.__init__ for other parameters

        Gtau: NDArray[np.float64], 3-dimensional array with (ntau, nflavor, nflavor)
            G_{ab}(τ). If None, read from files f"{a}{b}/{filein_g}" where a,b = 0, 1, ..., nflavor-1.
        """

        params = dict_with_lowerkey(params)
        nflavor: int = params["num_flavor"]

        self.Gtau = np.array(Gtau)
        assert self.Gtau.ndim == 3
        assert self.Gtau.shape[1] == nflavor
        assert self.Gtau.shape[1] == self.Gtau.shape[2]

        ntau = self.Gtau.shape[0]
        super().__init__(params, "tau", ntau)
        self.nflavor = nflavor

    def solve_one(
        self,
        lambda_: float,
        idx: Optional[Union[List[int], NDArray[np.int64]]] = None,
        maxiter: int = -1,
        initial_mu: float = 1.0,
        oversampling: int = 2,
    ) -> SimpleOptimizer:

        print(f"start with loglambda={np.log10(lambda_)}")

        if idx is None:
            idx = np.arange(self.ntau)
        if isinstance(idx, list):
            idx = np.array(idx)
        if maxiter < 0:
            maxiter = self.maxiter
        nflavor = self.Gtau.shape[1]
        nf2 = nflavor * nflavor
        nL = self.size * nf2

        # y = self.ref_Gl(idx).reshape(-1)
        # A = PartialDiagonalMatrix(DiagonalMatrix(-self.basis.s), (nflavor, nflavor))
        y = self.ref_input(idx).reshape(-1)
        taus = self.ts[idx]

        objs: List[admmsolver.objectivefunc.ObjectiveFunctionBase] = []
        eq_cond: List[EqualityCondition] = []
        objs_index: Dict[str, int] = {}

        if self.use_sparse_ir:
            A = PartialDiagonalMatrix(
                np.transpose(self.basis.u(taus)) * self.basis.s, (nflavor, nflavor)
            )
        else:
            A = PartialDiagonalMatrix(
                np.einsum("li,l -> il", self.u[:, idx], self.s), (nflavor, nflavor)
            )

        if self.sumrule:
            rho_sum = np.reshape(-(self.Gtau[0, :, :] + self.Gtau[-1, :, :]), nf2)
            C = PartialDiagonalMatrix(self.sumrule_weight.reshape(1,-1), (nflavor, nflavor))
            lstsq_F: LeastSquares = ConstrainedLeastSquares(
                0.5, A=A, y=y, C=C, D=rho_sum
            )
        else:
            lstsq_F = LeastSquares(0.5, A=A, y=y)
        objs_index["lstsq_F"] = len(objs)
        objs.append(lstsq_F)

        l1_F = L1Regularizer(lambda_, nL)
        objs_index["l1_F"] = len(objs)
        objs.append(l1_F)

        ec = EqualityCondition(
            objs_index["lstsq_F"],
            objs_index["l1_F"],
            identity(self.size * nflavor * nflavor),
            identity(self.size * nflavor * nflavor),
        )
        eq_cond.append(ec)

        spmpade_eta = self.pade_eta
        do_spmpade = spmpade_eta > 0.0
        if self.nonneg:
            if self.use_sparse_ir:
                ws = np.hstack(
                    [
                        -self.basis.wmax,
                        self.basis.default_omega_sampling_points(),
                        self.basis.wmax,
                    ]
                )
                ws = _oversample(ws, oversampling)
                wnum = len(ws)
                Aw = PartialDiagonalMatrix(self.basis.v(ws).T, (nflavor, nflavor))
            else:
                iw = range(0, len(self.ws), self.nonneg_freq_interval)
                wnum = len(iw)
                ws = self.ws[iw]
                Aw = PartialDiagonalMatrix(self.v[:, iw].transpose(), (nflavor, nflavor))
            semipos_F = SemiPositiveDefinitePenalty((wnum, nflavor, nflavor), axis=0)
            objs_index["semipos_F"] = len(objs)
            objs.append(semipos_F)
            ec = EqualityCondition(
                objs_index["lstsq_F"],
                objs_index["semipos_F"],
                Aw,
                identity(wnum * nflavor * nflavor),
            )
            eq_cond.append(ec)
        if do_spmpade:
            if self.use_sparse_ir:
                ws = np.hstack(
                    [
                        -self.basis.wmax,
                        self.basis.default_omega_sampling_points(),
                        self.basis.wmax,
                    ]
                )
                wnum = len(ws)
                Aw = PartialDiagonalMatrix(self.basis.v(ws).T, (nflavor, nflavor))
            else:
                wnum = len(self.ws)
                ws = self.ws[:]
                Aw = PartialDiagonalMatrix(self.v.transpose(), (nflavor, nflavor))
            if not self.spmpade_prepared:
                self.prepare_spmpade(ws, pade_sigma=self.pade_sigma, nsamples=self.pade_nsamples)
            pade_weight = np.sqrt(
                spmpade_eta / (1.0 + self.rhovar_pade / (self.rho_pade**2))
            )
            A = DiagonalMatrix(pade_weight.reshape(-1))
            y = (self.rho_pade * pade_weight).reshape(-1)
            spmpade_F = LeastSquares(0.5, A=A, y=y)
            objs_index["spmpade_F"] = len(objs)
            objs.append(spmpade_F)
            ec = EqualityCondition(
                objs_index["lstsq_F"],
                objs_index["spmpade_F"],
                Aw,
                identity(wnum * nflavor * nflavor),
            )
            eq_cond.append(ec)

        p = Problem(objs, eq_cond)
        opt = SimpleOptimizer(p, mu=initial_mu)

        self.time_start = time.time()
        self.time_prev = self.time_start
        self.niter = 0

        def callback():
            self.niter += 1
            now = time.time()
            elapsed = now - self.time_prev
            self.time_prev = now
            self.total_time += elapsed
            self.total_time2 += elapsed * elapsed
            if self.niter % 1000 == 0:
                elapsed = time.time() - self.time_start
                print(f"  {self.niter} iteration finished in {elapsed} seconds")

        opt.solve(maxiter, callback=callback)
        elapsed = time.time() - self.time_start
        print(f"  {elapsed/self.niter} seconds/iteration")

        self.elapsed.append(elapsed/self.niter)
        self.total_niter += self.niter

        rtol = 1e-12
        if opt.check_convergence(rtol):
            print("  Converged")
        else:
            print("  Not converged")
        return opt

    def predict_rho(self, rho_l) -> NDArray[np.float64]:
        r_l = rho_l.reshape((-1, self.nflavor, self.nflavor))
        if self.use_sparse_ir:
            return np.einsum("lab,lw->wab", r_l, self.basis.v(self.ws))
        else:
            return np.einsum("lab,lw->wab", r_l, self.v)

    def predict_Gtau(self, rho_l, idx=None):
        if idx is None:
            idx = np.arange(self.ntau)
        r_l = rho_l.reshape((-1, self.nflavor, self.nflavor))
        if self.use_sparse_ir:
            sampler = sparse_ir.TauSampling(self.basis, self.ts[idx])
            ret = np.zeros((len(idx), self.nflavor, self.nflavor))
            for i, j in itertools.product(range(self.nflavor), repeat=2):
                ret[:, i, j] = np.real(sampler.evaluate(self.basis.s * r_l[:, i, j]))
            return ret
        else:
            return np.einsum("lab, l, li -> iab", r_l, self.s, self.u[:,idx])

    def ref_Gl(self, idx=None) -> NDArray[np.float64]:
        if idx is None:
            idx = np.arange(self.ntau)
        if self.use_sparse_ir:
            sampler = sparse_ir.TauSampling(self.basis, self.ts[idx])
            ret = np.zeros((self.size, self.nflavor, self.nflavor))
            for i, j in itertools.product(range(self.nflavor), repeat=2):
                ret[:, i, j] = sampler.fit(self.Gtau[idx, i, j])
            return ret
        else:
            raise NotImplementedError()

    def ref_Gtau(self, idx=None):
        if idx is None:
            idx = np.arange(self.ntau)
        return -self.Gtau[idx, :, :]

    def write_Gtau(self, ts, Gts, outdir: pathlib.Path):
        for (
            ifl,
            jfl,
        ) in itertools.product(range(self.nflavor), repeat=2):
            with open(outdir / f"Gtau_{ifl}{jfl}.dat", "w") as f:
                N = len(ts)
                for i in range(N):
                    f.write(str(ts[i]))
                    for gt in Gts:
                        y = gt[i, ifl, jfl]
                        f.write(f" {np.real(y)}")
                    f.write("\n")

    def write_rhol(
        self,
        rs,
        *,
        loglambda: float = None,
        outdir: pathlib.Path = None,
        filename: str = None,
    ):
        if outdir is None:
            outdir = pathlib.Path(".")
        if filename is None:
            filename = "rho_l.dat"
        nflavor = self.nflavor
        r_l = rs.reshape((-1, nflavor, nflavor))
        outdir.mkdir(parents=True, exist_ok=True)
        for (
            ifl,
            jfl,
        ) in itertools.product(range(nflavor), repeat=2):
            specfile = outdir / (filename + f".{ifl}_{jfl}")
            with open(specfile, "w") as f:
                if loglambda is not None:
                    f.write(f"# log_lambda = {loglambda}\n")
                for i, r in enumerate(r_l[:, ifl, jfl]):
                    f.write(f"{i} {np.real(r)}\n")

    def write_rho(
        self,
        ws,
        rs,
        *,
        loglambda: float = None,
        outdir: pathlib.Path = None,
        filename: str = None,
        rescale_dw: bool = True,
    ):
        if outdir is None:
            outdir = pathlib.Path(".")
        if filename is None:
            filename = "spectrum.dat"
        outdir.mkdir(parents=True, exist_ok=True)
        nflavor = rs.shape[1]
        if rescale_dw:
            dw = ws[1] - ws[0]
        else:
            dw = 1.0
        for (
            ifl,
            jfl,
        ) in itertools.product(range(nflavor), repeat=2):
            specfile = outdir / (filename + f".{ifl}_{jfl}")
            with open(specfile, "w") as f:
                if loglambda is not None:
                    f.write(f"# log_lambda = {loglambda}\n")
                for w, r in zip(ws, rs[:, ifl, jfl]):
                    f.write(f"{w} {np.real(r)/dw}\n")
