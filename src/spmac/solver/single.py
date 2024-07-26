# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2024- SpM-lab
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from typing import Union, Any, Optional, Dict, List

import time
import pathlib

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
    NonNegativePenalty,
)
from admmsolver.matrix import identity, DiagonalMatrix
from admmsolver.optimizer import SimpleOptimizer

import sparse_ir

from ..util import dict_with_lowerkey
from .base import SolverBase, _oversample


class Solver(SolverBase):
    def __init__(
        self, params: Dict[str, Any], Gtau: NDArray[np.float64]
    ) -> None:
        """

        Arguments
        -----------
        params: dict[str, Any]
            Note that parameter names are case-insensitive

            - filein_g: str
                - Filename storing G(τ)
            - column: int
                - Index of column storing G(τ) (0-origin)

            See base.SolverBase.__init__ for other parameters

        Gtau: NDArray[np.float64], 1-dimensional array
            G(τ). If None, read from file.
        """
        params = dict_with_lowerkey(params)
        assert Gtau.ndim == 1 or Gtau.ndim == 3
        if Gtau.ndim == 1:
            self.Gtau = np.array(Gtau)
        else:
            assert Gtau.shape[0] == Gtau.shape[1] == 1
            self.Gtau = np.array(Gtau).flatten()
        ntau = self.Gtau.shape[0]
        super().__init__(params, "tau", ntau)


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
            idx = np.arange(len(self.Gtau))
        if isinstance(idx, list):
            idx = np.array(idx)
        if maxiter < 0:
            maxiter = self.maxiter

        # y = self.ref_Gl(idx)
        y = self.ref_input(idx)
        taus = self.ts[idx]

        objs: List[admmsolver.objectivefunc.ObjectiveFunctionBase] = []
        eq_cond: List[EqualityCondition] = []
        objs_index: Dict[str, int] = {}

        if self.use_sparse_ir:
            # A = np.transpose(self.basis.u(taus)) * self.basis.s
            A = np.einsum("li,l -> il", self.basis.u(taus), self.basis.s)
        else:
            A = np.einsum("li,l -> il", self.u[:, idx], self.s)

        if self.sumrule:
            rho_sum = -(self.Gtau[0] + self.Gtau[-1])
            C = self.sumrule_weight.reshape(1, -1)
            # C = np.einsum("m, m -> m", self.s, self.u[:, 0] + self.u[:, -1]).reshape(
            #     1, -1
            # )
            lstsq_F: LeastSquares = ConstrainedLeastSquares(
                0.5, A=A, y=y, C=C, D=np.array([rho_sum])
            )
        else:
            lstsq_F = LeastSquares(0.5, A=A, y=y)
        objs_index["lstsq_F"] = len(objs)
        objs.append(lstsq_F)

        l1_F = L1Regularizer(lambda_, self.size)
        objs_index["l1_F"] = len(objs)
        objs.append(l1_F)

        ec = EqualityCondition(
            objs_index["lstsq_F"],
            objs_index["l1_F"],
            identity(self.size),
            identity(self.size),
        )
        eq_cond.append(ec)

        spmpade_eta = self.pade_eta
        do_spmpade = spmpade_eta > 0.0
        if self.nonneg or do_spmpade:
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
                if self.nonneg:
                    nonneg_F = NonNegativePenalty(wnum)
                    objs_index["nonneg_F"] = len(objs)
                    objs.append(nonneg_F)
                    ec = EqualityCondition(
                        objs_index["lstsq_F"],
                        objs_index["nonneg_F"],
                        self.basis.v(ws).T,
                        identity(wnum),
                    )
                    eq_cond.append(ec)

                if do_spmpade:
                    if not self.spmpade_prepared:
                        self.prepare_spmpade(ws, pade_sigma=self.pade_sigma, nsamples=self.pade_nsamples)
                    pade_weight = np.sqrt(
                        spmpade_eta / (1.0 + self.rhovar_pade / (self.rho_pade**2))
                    )
                    A = DiagonalMatrix(pade_weight)
                    y = (self.rho_pade * pade_weight)[:,0,0]
                    spmpade_F = LeastSquares(0.5, A=A, y=y)
                    objs_index["spmpade_F"] = len(objs)
                    objs.append(spmpade_F)
                    ec = EqualityCondition(
                        objs_index["lstsq_F"],
                        objs_index["spmpade_F"],
                        self.basis.v(ws).T,
                        identity(wnum),
                    )
                    eq_cond.append(ec)
            else:
                iw = range(0, len(self.ws), self.nonneg_freq_interval)
                wnum = len(iw)
                ws = self.ws[iw]
                if self.nonneg:
                    nonneg_F = NonNegativePenalty(wnum)
                    objs_index["nonneg_F"] = len(objs)
                    objs.append(nonneg_F)
                    ec = EqualityCondition(
                        objs_index["lstsq_F"],
                        objs_index["nonneg_F"],
                        self.v[:, iw].transpose(),
                        identity(wnum),
                    )
                    eq_cond.append(ec)
                if do_spmpade:
                    if not self.spmpade_prepared:
                        self.prepare_spmpade(ws, pade_sigma=self.pade_sigma, nsamples=self.pade_nsamples)

                    pade_weight = np.sqrt(
                        spmpade_eta / (1.0 + self.rhovar_pade / (self.rho_pade**2))
                    )
                    A = DiagonalMatrix(pade_weight.reshape(-1))
                    y = (self.rho_pade * pade_weight)[:,0,0]
                    spmpade_F = LeastSquares(0.5, A=A, y=y)
                    objs_index["spmpade_F"] = len(objs)
                    objs.append(spmpade_F)
                    ec = EqualityCondition(
                        objs_index["lstsq_F"],
                        objs_index["spmpade_F"],
                        self.v[:, iw].transpose(),
                        identity(wnum),
                    )
                    eq_cond.append(ec)

        p = Problem(objs, eq_cond)
        opt = SimpleOptimizer(p, mu=initial_mu)

        self.time_start = time.time()
        self.time_prev = self.time_start
        self.iter = 0

        def callback():
            self.iter += 1
            now = time.time()
            elapsed = now - self.time_prev
            self.time_prev = now
            self.total_time += elapsed
            self.total_time2 += elapsed * elapsed
            if self.iter % 1000 == 0:
                elapsed = time.time() - self.time_start
                print(f"  {self.iter} iteration finished in {elapsed} seconds")

        opt.solve(maxiter, callback=callback)
        elapsed = time.time() - self.time_start
        print(f"  {elapsed/self.iter} seconds/iteration")

        self.elapsed.append(elapsed / self.iter)
        self.total_niter += self.iter

        rtol = 1e-12
        if opt.check_convergence(rtol):
            print("  Converged")
        else:
            print("  Not converged")
        return opt

    def predict_rho(self, rho_l: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.use_sparse_ir:
            return rho_l @ self.basis.v(self.ws)
        else:
            return rho_l @ self.v

    def predict_Gtau(self, rho_l: NDArray[np.float64], idx=None) -> NDArray[np.float64]:
        if idx is None:
            idx = np.arange(self.ntau)
        if self.use_sparse_ir:
            sampler = sparse_ir.TauSampling(self.basis, self.ts[idx])
            return sampler.evaluate(self.basis.s * rho_l)
        else:
            return rho_l @ self.u[:, idx]

    def ref_Gl(self, idx=None) -> NDArray[np.float64]:
        if idx is None:
            idx = np.arange(self.ntau)
        if self.use_sparse_ir:
            sampler = sparse_ir.TauSampling(self.basis, self.ts[idx])
            return sampler.fit(self.Gtau[idx])
        else:
            raise NotImplementedError()

    def ref_Gtau(self, idx=None) -> NDArray[np.float64]:
        if idx is None:
            idx = np.arange(self.ntau)
        return -self.Gtau[idx]

    def write_Gtau(self, ts, Gts, outdir: pathlib.Path):
        with open(outdir / "Gtau.dat", "w") as f:
            N = len(ts)
            for i in range(N):
                f.write(str(ts[i]))
                for gt in Gts:
                    y = gt[i]
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
        outdir.mkdir(parents=True, exist_ok=True)
        specfile = outdir / filename
        if rs.ndim > 1:
            rs = rs.reshape(-1)

        with open(specfile, "w") as f:
            if loglambda is not None:
                f.write(f"# log_lambda = {loglambda}\n")
            for i, r in enumerate(rs):
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
        if rescale_dw:
            dw = ws[1] - ws[0]
        else:
            dw = 1.0
        outdir.mkdir(parents=True, exist_ok=True)
        specfile = outdir / filename
        if rs.ndim > 1:
            rs = rs.reshape(-1)

        with open(specfile, "w") as f:
            if loglambda is not None:
                f.write(f"# log_lambda = {loglambda}\n")
            for w, r in zip(ws, rs):
                f.write(f"{w} {np.real(r)/dw}\n")
                # f.write(f"{w} {np.real(r)}\n")
