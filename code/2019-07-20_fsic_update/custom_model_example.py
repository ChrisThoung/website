# -*- coding: utf-8 -*-
"""
custom_model_example
====================
Example of a hand-edited model definition based on a (further) simplified
version of Godley and Lavoie's (2007) Model *SIM*.

The code to generate the original model template is as follows:

    import fsic

    script = '''\
C = {alpha_1} * YD + {alpha_2} * H[-1]
YD = Y - T
Y = C + G
T = {theta} * Y
H = H[-1] + YD - C
'''

    symbols = fsic.parse_model(script)
    definition = fsic.build_model_definition(symbols)

Reference:

    Godley, W., Lavoie, M. (2007)
    *Monetary economics:  
    An integrated approach to credit, money, income, production and wealth*,
    Palgrave Macmillan

-------------------------------------------------------------------------------
MIT License

Copyright (c) 2019 Chris Thoung

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

from fsic import BaseModel


class CustomSIM(BaseModel):
    ENDOGENOUS = ['C', 'YD', 'H', 'Y', 'T']
    EXOGENOUS = ['G']

    PARAMETERS = ['alpha_1', 'alpha_2', 'theta']

    # Remove extraneous (empty) `ERRORS` attribute
    NAMES = ENDOGENOUS + EXOGENOUS + PARAMETERS

    # (Arbitrarily) reduce the variables to be checked for convergence during
    # solution
    CHECK = ['C', 'H', 'Y']

    LAGS = 1
    LEADS = 0

    # Extend function signature with a new keyword argument to apply exogenous
    # changes in household consumption expenditure
    # Not required, but note use of keyword-only argument in the function
    # signature:
    #     https://www.python.org/dev/peps/pep-3102/
    def _evaluate(self, t, *, exogenous_change_in_consumption=0):
        self._C[t] = self._alpha_1[t] * self._YD[t] + self._alpha_2[t] * self._H[t-1]

        # Apply exogenous changes in household consumption expenditure
        self._C[t] += exogenous_change_in_consumption

        self._YD[t] = self._Y[t] - self._T[t]
        self._H[t] = self._H[t-1] + self._YD[t] - self._C[t]
        self._Y[t] = self._C[t] + self._G[t]
        self._T[t] = self._theta[t] * self._Y[t]
