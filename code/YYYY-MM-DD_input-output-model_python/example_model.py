# -*- coding: utf-8 -*-
"""
example_model
=============
Example implementation of a macroeconomic input-output model in Python, using
economic data for Scotland ('Aggregate Tables 1998 to 2016'):

    https://www.gov.scot/publications/supply-use-input-output-tables-multipliers-scotland/

Download the data to this folder and run 'process_raw_data.py' first, to
generate the input data (in 'data.txt').

-------------------------------------------------------------------------------
MIT License

Copyright (c) 2020 Chris Thoung

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from typing import Any, Dict

import numpy as np

from iomodel import BaseMDModel
import iomodel


class ScottishIOModel(BaseMDModel):

    DIMENSIONS = {
        'Industries': ['Agriculture, forestry and fishing (A)',
                       'Mining and quarrying (B)',
                       'Manufacturing (C)',
                       'Energy supply (D)',
                       'Water and waste (E)',
                       'Construction (F)',
                       'Distribution, hotels and catering (GI)',
                       'Transport, storage and communication (HJ)',
                       'Financial, insurance and real estate (KL)',
                       'Professional and support activities (MN)',
                       'Government, health and education (OPQ)',
                       'Other services (RST)', ]
    }

    # Conventions:
    #  - uppercase letters e.g. Z : a matrix in period t
    #  - lowercase letters e.g. q : a vector in period t
    VARIABLES = {
        'Z': ('Industries', 'Industries'),  # Intermediate consumption
        'A': ('Industries', 'Industries'),  # Technical coefficients
        'L': ('Industries', 'Industries'),  # Leontief inverse

        'q': 'Industries',                  # Gross output

        'f': 'Industries',                  # Final demand (sum of all components)
        'c': 'Industries',                  # HHFCE
        'g': 'Industries',                  # Government expenditure
        'i': 'Industries',                  # Gross capital formation
        'x_nr': 'Industries',               # Exports - Non-residents
        'x_ruk': 'Industries',              # Exports - RUK
        'x_row': 'Industries',              # Exports - RoW
    }

    ENDOGENOUS = ['q']
    CHECK = ENDOGENOUS

    def _evaluate(self, t: int, **kwargs: Dict[str, Any]) -> None:
        # Sum of components of final demand
        self._f[t] = (self._c[t] +
                      self._g[t] +
                      self._i[t] +
                      self._x_nr[t] +
                      self._x_ruk[t] +
                      self._x_row[t])

        # Leontief inverse and implied gross output
        self._L[t] = np.linalg.inv(np.eye(self._A[t].shape[0]) - self._A[t])
        self._q[t] = self._L[t] @ self._f[t]

        # Intermediate consumption
        self._Z[t] = self._A[t] * self._q[t].reshape((1, -1))


if __name__ == '__main__':
    # Read input data
    with open('data.txt') as f:
        start_year, end_year = map(int, next(f).split())
        data = iomodel.read_data(f)

    # Pop variables that the model should recover in solution, for later
    # comparison
    q = data.pop('q')  # q = L @ f
    Z = data.pop('Z')  # Intermediate consumption = A * q

    # Set up the model and solve
    model = ScottishIOModel(range(start_year, end_year + 1), **data)
    model.solve()

    # Check results for output and intermediate consumption
    assert model.q.shape == q.shape
    assert np.allclose(model.q, q)

    assert model.Z.shape == Z.shape
    assert np.allclose(model.Z, Z)
