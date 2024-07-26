# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2024- SpM-lab
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from typing import Any, Dict

def dict_with_lowerkey(d_in: Dict[str, Any]) -> Dict[str, Any]:
    ret: Dict[str, Any] = {}
    for k, v in d_in.items():
        lk = k.lower()
        if lk in ret:
            raise RuntimeError(f"ERROR: parameter {lk} is duplicated")
        else:
            ret[k.lower()] = v
    return ret
