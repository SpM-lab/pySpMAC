# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2024- SpM-lab
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from typing import List, Union, Optional

import itertools

import numpy as np
from numpy.typing import NDArray

from ..pade import Pade
from ..matsubara import frequencies_fermion, t2w_fermion


class PadeAC:
    gs: NDArray[np.float64]
    beta: float

    def __init__(self, gs: NDArray[np.float64], beta: float, wmax: Optional[float] = None):
        assert gs.ndim == 1 or gs.ndim == 3
        if gs.ndim == 1:
            self.gs = gs.reshape(-1, 1, 1)
        else:
            assert gs.shape[1] == gs.shape[2]
            self.gs = gs
        self.beta = beta

    def __call__(self, ws: Union[float, List[float], NDArray[np.float64]]):
        if isinstance(ws, float):
            ws = np.array([ws])
        elif isinstance(ws, list):
            ws = np.array(ws)
        nw = ws.size
        dw = ws[1] - ws[0]
        ntau = self.gs.shape[0]
        nflavor = self.gs.shape[1]
        Gw = np.zeros((nw, nflavor, nflavor))
        for ifl, jfl in itertools.product(range(nflavor), repeat=2):
            gt = self.gs[:, ifl, jfl]
            niw = (ntau - 1)//2
            iws = frequencies_fermion(self.beta, niw)
            Giw = t2w_fermion(gt, self.beta)
            # gt_f = np.concatenate([gt, -gt])
            # Giw = fft.ifft(gt_f)[1 : ntau + 1 : 2]
            # iws = complex(0.0, np.pi / self.beta) * np.arange(1, ntau + 1, step=2)
            pade_ = Pade(iws[iws > 0], Giw[iws > 0])
            Gw[:, ifl, jfl] = (-dw/np.pi) * np.imag(pade_.evaluate(ws.astype(np.complex128)))
        return Gw
