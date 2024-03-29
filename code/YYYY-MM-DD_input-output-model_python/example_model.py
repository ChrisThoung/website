# -*- coding: utf-8 -*-
"""
example_model
=============
Example implementation of a macroeconomic input-output model in Python, using
economic data for Scotland ('Aggregate tables 1998 to 2017'):

    https://www.gov.scot/publications/input-output-latest/

Download the data to this folder and run 'process_raw_data.py' first, to
generate the input data (in 'data.txt').

-------------------------------------------------------------------------------
MIT License

Copyright (c) 2020-21 Chris Thoung

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
from pandas import DataFrame

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

    # Naming conventions (personal: none are enforced by the code):
    #  - uppercase letters e.g. Z : a matrix in period t
    #  - lowercase letters e.g. q : a vector in period t
    #
    # Dictionary values consist of:
    #  - int, str or sequence of int/str: variable dimensions
    #  - array dtype
    #  - default starting value(s) for the variable
    VARIABLES = {
        'Z': (('Industries', 'Industries'), float, 0.0),  # Intermediate consumption
        'A': (('Industries', 'Industries'), float, 0.0),  # Technical coefficients
        'L': (('Industries', 'Industries'), float, 0.0),  # Leontief inverse

        'q': ('Industries', float, 0.0),                  # Gross output

        'f': ('Industries', float, 0.0),                  # Final demand (sum of all components)
        'c': ('Industries', float, 0.0),                  # HHFCE
        'g': ('Industries', float, 0.0),                  # Government expenditure
        'i': ('Industries', float, 0.0),                  # Gross capital formation
        'x_nr': ('Industries', float, 0.0),               # Exports - Non-residents
        'x_ruk': ('Industries', float, 0.0),              # Exports - RUK
        'x_row': ('Industries', float, 0.0),              # Exports - RoW
    }

    ENDOGENOUS = ['q']
    CHECK = ENDOGENOUS

    def _evaluate(self, t: int, **kwargs: Dict[str, Any]) -> None:
        # Sum of components of final demand
        # (Time is always the first dimension/axis. All others follow the order
        # in `VARIABLES`.)
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
    # Setup -------------------------------------------------------------------
    # Read input data
    with open('data.txt') as f:
        start_year, end_year = map(int, next(f).split())
        data = iomodel.read_data(f)

    # Pop variables that the model should recover in solution, for later
    # comparison
    q = data.pop('q')  # q = L @ f
    Z = data.pop('Z')  # Intermediate consumption = A * q

    # Baseline: Solve for historical values -----------------------------------
    # Instantiate a model object with data and solve
    baseline = ScottishIOModel(range(start_year, end_year + 1), **data)
    baseline.solve()

    # Check results for output and intermediate consumption
    # (Model variables are accessible as object attributes.)
    assert baseline.q.shape == q.shape
    assert np.allclose(baseline.q, q)

    assert baseline.Z.shape == Z.shape
    assert np.allclose(baseline.Z, Z)

    # Scenario: Higher government spending ------------------------------------
    # Copy the model instance to run an alternative scenario
    scenario = baseline.copy()

    # Increase government spending on government services by 1% in 2017
    # Object indexing works as follows:
    #  - the first index (required): the name of the variable e.g. 'g'
    #  - the second index (optional; use `:` to select all items): the period label(s) e.g. 2017, 2015:2017 etc
    #  - further indexes (optional): slice notation as per usual for NumPy arrays

    # Last-but-one industry (index -2): 'Government, health and education'
    scenario['g', 2017, -2] *= 1.01

    # Solve
    # (Only 2017 actually changes. Could've used `scenario.solve_period(2017)`)
    scenario.solve()

    # Compare total gross output (sum of `q`):
    #  - figures should be identical over 1998-2016 (and match the data)
    #  - total gross output in 2017 should be higher in the scenario, and by an
    #    amount equal to the increase in government spending multiplied by the
    #    corresponding column of the Leontief Inverse
    print(DataFrame({'Baseline': baseline.q.sum(axis=1),
                     'Scenario': scenario.q.sum(axis=1), },
                    index=baseline.span).round())
