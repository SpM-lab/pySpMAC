# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2024- SpM-lab
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from typing import Union

import copy

import numpy as np
from numpy.typing import NDArray


class Pade:
    """
    F(z) = a0/[1 +a1(z-z1)/[1 +a2(z-z2)/[1 +...
    """

    N: int
    "datasize"
    a: NDArray[np.complex128]
    "coefficient"
    z: NDArray[np.complex128]
    "position"

    def __init__(self, z: NDArray[np.complex128], u: NDArray[np.complex128]):
        assert z.size == u.size
        self.N = z.size
        a = copy.copy(u)
        for p in range(1, self.N):
            for q in range(p, self.N):
                a[q] = (a[p - 1] - a[q]) / ((z[q] - z[p - 1]) * a[q])
        self.a = a
        self.z = copy.copy(z)

    def evaluate(
        self, w: Union[np.complex128, NDArray[np.complex128]]
    ) -> NDArray[np.complex128]:
        w = np.array(w).flatten()
        n = w.size

        P0: NDArray[np.complex128] = np.zeros(n, dtype=np.complex128)
        P1: NDArray[np.complex128] = np.zeros(n, dtype=np.complex128)
        P2: NDArray[np.complex128] = np.zeros(n, dtype=np.complex128)
        Q0: NDArray[np.complex128] = np.ones(n, dtype=np.complex128)
        Q1: NDArray[np.complex128] = np.ones(n, dtype=np.complex128)
        Q2: NDArray[np.complex128] = np.ones(n, dtype=np.complex128)

        P1[:] = self.a[0]
        for p in range(1, self.N):
            coeffs: NDArray[np.complex128] = self.a[p] * (w - self.z[p - 1])
            P2[:] = P1 + coeffs * P0
            Q2[:] = Q1 + coeffs * Q0

            P0[:] /= Q2
            P1[:] /= Q2
            P2[:] /= Q2
            Q0[:] /= Q2
            Q1[:] /= Q2
            Q2[:] = 1

            P0[:] = P1
            P1[:] = P2
            Q0[:] = Q1
            Q1[:] = Q2

        return P2 / Q2
