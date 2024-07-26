# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2024- SpM-lab
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import numpy.fft as fft


def frequencies_fermion(beta, nw):
    """ Fermionic Matsubara frequencies

    Parameters
    ----------
    beta : float
        Inverse temperature
    nw : int
        Number of Matsubara frequencies

    Returns
    -------
    np.array[float]
        Matsubara frequencies
        [-iw_n, ..., -iw_1, iw_1, ..., iw_n]
    """
    iw_pos = np.array([(2*i+1) * np.pi / beta for i in range(nw)])
    iw = np.append(-iw_pos[::-1], iw_pos) * 1j
    return iw


def t2w_fermion(gt, beta):
    """FFT from G(tau) to G(iw) for fermion

    Parameters
    ------------
    gt: numpy.ndarray
        G(tau), length should be odd,
        and the first and last element should be G(tau=0) and G(tau=beta), respectively.

    beta: float
        Inverse temperature

    Returns
    -----------
    gw_fermion: numpy.ndarray
        G(iw) including w>0 and w<0
    """
    if gt.size % 2 == 0:
        raise ValueError("Length of gt should be odd")

    # subtract tail term and
    # impose anti-periodic boundary condition
    a =  -(gt[0]+gt[-1])
    gt2 = gt[:-1] + 0.5*a # remove G(tau=beta)
    gt_full = np.append(gt2, -gt2)

    # FFT
    gw = fft.ifft(gt_full)[1::2] * beta

    # change order of frequency: w<0, w>0
    nw = gw.size // 2
    gw = np.roll(gw, nw)

    # recover tail term
    iw = frequencies_fermion(beta, nw)
    gw += a / iw

    return gw