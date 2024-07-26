# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2024- SpM-lab
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import typing
from typing import Union, Any, Optional, Dict, List

from abc import ABCMeta, abstractmethod

import pathlib

import numpy as np
from numpy.typing import NDArray

from admmsolver.optimizer import SimpleOptimizer

import sparse_ir

from ..util import dict_with_lowerkey
from .pade import PadeAC


def fermion_kernel(tau, omega, beta):
    if omega < 0.0:
        return np.exp((beta - tau) * omega) / (1.0 + np.exp(beta * omega))
    else:
        return np.exp(-tau * omega) / (1.0 + np.exp(-beta * omega))


class SolverBase(metaclass=ABCMeta):
    outdir: pathlib.Path
    "Output directory"

    beta: float
    "Inverse temperature β"

    input_type: str
    'type of input: "tau"'

    Gtau: NDArray[np.float64]
    "Temperature Green's function G(τ_i)"

    ts: NDArray[np.float64]
    "Imaginary time τ_i"

    ntau: int
    "Number of ts"

    ws: NDArray[np.float64]
    "Real frequencies ω_j"

    spmpade_prepared: bool
    rho_pade: NDArray[np.float64]
    rhovar_pade: NDArray[np.float64]

    maxiter: int
    "Maximum number of iterations in ADMM"

    nonneg: bool
    "Impose nonnegativity/semipositive-ness or not"

    nonneg_freq_interval: int
    """Frequency interval to impose nonnegativity/semipositive-ness
    1 means all frequencies, 2 means only ω_0, ω_2, ω_4, ...
    """

    sumrule: bool
    "Impose sum-rule or not"

    sumrule_weight: NDArray[np.float64]
    "Weights used in calculating sumrule"

    log_id: int

    basis: sparse_ir.FiniteTempBasis
    "IR basis"

    size: int
    "The number of the remaining singular values"


    def __init__(self, params: Dict[str, Any], input_type: str, ntau: int) -> None:
        """

        Arguments
        -----------
        params: dict[str, Any]
            Note that parameter names are case-insensitive

            - output: str
                - Output directory name (default: output)
            - beta: float
                - Inverse temperature β
            - max_omega: float
                - Upper bound of ω
                - Note: Lower bound of ω will be ``-max_omega``
            - num_omega: int
                - Number of ωs
            - nonnegative: Bool
                - Impose nonnegativity/semipositiveness or not (default:True)
            - sumrule: Bool
                - Impose sum-rule or not (default: True)
            - min_sv: float
                - cutoff in singular value (default: 1e-10)
            - max_iteration: int
                - Maximum number of iterations of ADMM (default: 1000)
        """
        self.input_type = input_type
        params = dict_with_lowerkey(params)
        self.log_id = 0
        outdir = params.get("output", "output")
        self.outdir = pathlib.Path(outdir)
        self.beta = typing.cast(float, params["beta"])
        self.ntau = ntau
        self.ts = np.linspace(0.0, self.beta, num=self.ntau)

        self.use_sparse_ir = params.get("use_sparse_ir", False)

        wmax: float = params["max_omega"]
        wmin: float = params["min_omega"]
        wmax_ = max(np.abs(wmax), np.abs(wmin))
        # wmin = -wmax
        wnum: int = params["num_omega"]
        self.ws = np.linspace(wmin, wmax, num=wnum)

        self.spmpade_prepared = False
        self.rho_pade = np.zeros(0, dtype=np.float64)
        self.rhovar_pade = np.zeros(0, dtype=np.float64)

        self.maxiter = params.get("max_iteration", 1000)
        self.nonneg = params.get("nonnegative", True)
        self.nonneg_freq_interval = params.get("nonnegative_freq_interval", 1)
        self.sumrule = params.get("sumrule", True)
        self.nflavor = 1

        SVmin: float = params.get("min_sv", 1e-10)

        if self.use_sparse_ir:
            self.statistics = "F"
            self.basis = sparse_ir.FiniteTempBasis(
                self.statistics, self.beta, wmax_, eps=SVmin
            )
            self.sumrule_weight = typing.cast(
                NDArray[np.float64], self.basis.v.overlap(lambda _: 1.0)
            )
            self.size = self.basis.s.size
        else:
            ntau = len(self.ts)
            nomega = len(self.ws)
            domega = self.ws[1] - self.ws[0]
            K = np.zeros((ntau, nomega), dtype=np.float64)
            for itau, tau in enumerate(self.ts):
                for iomega, omega in enumerate(self.ws):
                    # K[itau, iomega] = fermion_kernel(tau, omega, self.beta) * domega
                    K[itau, iomega] = fermion_kernel(tau, omega, self.beta)
            U, S, V = np.linalg.svd(K)
            self.size = np.count_nonzero(S >= SVmin)
            self.s = S[: self.size]
            self.u = U[:, : self.size].transpose()
            self.v = V[: self.size, :]
            # self.sumrule_weight = np.einsum("li -> l", self.v) * domega
            self.sumrule_weight = np.einsum("li -> l", self.v)

            # uu = (self.u[:,0] + self.u[:,-1]).reshape(1, -1)
            # self.sumrule_weight = np.einsum("l,al->al", self.s, uu)
            # print(self.sumrule_weight.shape)
        self.pade_nsamples = params.get("spmpade_nsamples", 30)
        self.pade_sigma = params.get("pade_sigma", 1e-5)
        self.pade_eta = params.get("spmpade_eta", 0.0)
        self.elapsed: List[float] = []
        self.total_time = 0.0
        self.total_time2 = 0.0
        self.total_niter = 0

    def prepare_spmpade(
        self,
        ws: Optional[Union[List[float], NDArray[np.float64]]] = None,
        pade_sigma: float = 1e-6,
        nsamples: int = 30,
    ):
        if ws is None:
            ws = self.ws
        elif isinstance(ws, list):
            ws = np.array(ws)
        nw = len(ws)

        self.rho_pade = np.zeros((nw, self.nflavor, self.nflavor), dtype=np.float64)
        self.rhovar_pade = np.zeros((nw, self.nflavor, self.nflavor), dtype=np.float64)
        #
        for _ in range(nsamples):
            Gtau_pade = self.Gtau + pade_sigma * np.random.randn(*self.Gtau.shape)
            pade = PadeAC(Gtau_pade, self.beta)
            rho_pade = pade(ws)
            self.rho_pade += rho_pade
            self.rhovar_pade += rho_pade ** 2
        self.rhovar_pade = (
            self.rhovar_pade - self.rho_pade * self.rho_pade / nsamples
        ) / (nsamples - 1)
        self.rho_pade /= nsamples
        self.spmpade_prepared = True

    @abstractmethod
    def solve_one(
        self,
        lambda_: float,
        idx: Optional[Union[List[int], NDArray[np.int64]]] = None,
        niter: int = -1,
        initial_mu: float = 1.0,
        pade_eta: float = 0.0,
    ) -> SimpleOptimizer:
        """Optimize ρ_l by τ(idx) with L1-coefficient λ"""
        raise NotImplementedError()

    @abstractmethod
    def predict_rho(self, rho_l: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict ρ(ω) from ρ_l"""
        raise NotImplementedError()

    @abstractmethod
    def predict_Gtau(self, rho_l: NDArray[np.float64], idx=None) -> NDArray[np.float64]:
        """Predict G(τ) from ρ_l"""
        raise NotImplementedError()

    def ref_input(self, idx=None) -> NDArray[np.float64]:
        if self.input_type == "tau":
            return self.ref_Gtau(idx)
        else:
            raise NotImplementedError()

    @abstractmethod
    def ref_Gl(self, idx=None) -> NDArray[np.float64]:
        """Fit G_l from G(τ[idx])"""
        raise NotImplementedError()

    @abstractmethod
    def ref_Gtau(self, idx=None) -> NDArray[np.float64]:
        """Get a part of G(τ) (input data)"""
        raise NotImplementedError()

    def predict_input(self, opt_x, idx=None):
        if self.input_type == "tau":
            return self.predict_Gtau(opt_x, idx)
        else:
            raise NotImplementedError()

    # def predict_Gtau(self, opt_x, idx=None):
    #     if idx is None:
    #         idx = np.arange(self.ntau)
    #     tau = self.ts[idx]
    #     Gl = self.basis.s.reshape(-1, 1, 1) * opt_x.reshape(
    #         -1, self.nflavor, self.nflavor
    #     )
    #     return np.einsum("il, lab -> iab", self.basis.u(tau), Gl)

    @abstractmethod
    def write_Gtau(self, ts, Gts, outdir: pathlib.Path):
        """Write G(τ) to a file under ``outdir``"""
        raise NotImplementedError()

    @abstractmethod
    def write_rhol(
        self,
        rs,
        *,
        loglambda: float = None,
        outdir: pathlib.Path = None,
        filename: str = None,
        rescale_dw: bool = True,
    ):
        """Write ρ_l to a file under ``outdir``"""
        raise NotImplementedError()

    @abstractmethod
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
        """Write ρ(ω) to a file under ``outdir``"""
        raise NotImplementedError()


def __oversample(x: np.ndarray):
    xmid = 0.5 * (x[1:] + x[:-1])
    return np.unique(np.hstack((x, xmid)))


def _oversample(x: np.ndarray, n: int = 1):
    for i in range(n):
        x = __oversample(x)
    return x
