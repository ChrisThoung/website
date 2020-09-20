# -*- coding: utf-8 -*-
"""
process_raw_data
================
Extract the raw data from the input-output tables and form into a more
convenient data file. Data are for Scotland ('Aggregate tables 1998 to 2017'):

    https://www.gov.scot/publications/input-output-latest/

Figures are in £m.

Download the data to this folder and run this script, to generate the input
data (in 'data.txt'). Then run 'example_model.py' to run the model itself.

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

import numpy as np
import pandas as pd

from iomodel import write_data


# Identifiers for rows corresponding to industries
industry_codes = ['A', 'B', 'C', 'D', 'E', 'F', 'GI', 'HJ', 'KL', 'MN', 'OPQ', 'RST', ]


if __name__ == '__main__':
    # Read raw data tables ----------------------------------------------------
    tables = pd.read_excel('SUT-Agg-98-17.xlsx',
                           sheet_name='Aggregate IxI 1998-2017',
                           skiprows=3)

    # The original file has a merged cell at the end that prevents pandas
    # reading the final column title properly
    columns = list(tables.columns)
    columns[-1] = 'Total use for industry object'
    tables.columns = [x.split('\n')[0] for x in columns]

    # Extract components of industry output -----------------------------------

    # Individual stores for the matrix variables
    Z = []
    A = []

    # The format for the vector components is more standard: can loop through
    # these items and extract in a similar way for each
    components = {
        'q': (None, 'Total output at basic prices', ),
        'm_ruk': ('RUKImp', 'Imports from rest of UK', ),
        'm_row': ('RoWImp', 'Imports from rest of world', ),
        't_products': ('TlSPrds', 'Taxes less subsidies on products', ),
        't_production': ('TlSPrdn', 'Taxes less subsidies on production', ),
        'coe': ('CoE', 'Compensation of employees', ),
        'gos': ('GOS', 'Gross operating surplus', ),
    }

    # Initial container for vector components
    collect = {k: [] for k in components.keys()}

    # Extract data year by year
    for year, df in tables.groupby('Year'):
        df = df.set_index('Section').iloc[:, 2:]
        df = df.iloc[:, :len(industry_codes)]

        # Extract intermediate consumption and gross output separately, to then
        # calculate the input-output coefficients
        intermediate_consumption = df.loc[industry_codes, :].values
        Z.append(intermediate_consumption)

        gross_output = df.loc['TOut', :].values

        A.append(intermediate_consumption / gross_output.reshape((1, -1)))

        # Extract the vector components
        for name, (code, _) in components.items():
            if code is not None:
                collect[name].append(df.loc[code, :].values)

        # Already have gross output: no need to extract again
        collect['q'].append(gross_output)

    # -------------------------------------------------------------------------
    # Create stores for the data and metadata with the convention that the
    # outer/leftmost axis is time
    data = {
        'Z': np.array(Z),
        'A': np.array(A),
    }

    metadata = {
        'Z': 'Intermediate consumption',
        'A': 'Technical coefficients',
    }

    # Add the vector components
    for name, (_, description) in components.items():
        data[name] = np.array(collect[name])
        metadata[name] = description

    # Extract components of final demand --------------------------------------
    final_demand = {
        'c': 'Consumers',
        'g': 'Government',
        'i': 'Gross capital formation',
        'x_nr': 'Exports - Non-residents',
        'x_ruk': 'Exports - RUK',
        'x_row': 'Exports - RoW',
    }

    for code, name in final_demand.items():
        # Form a table of (industries x time)
        extract = tables.pivot(index='Section', columns='Year', values=name)

        data[code] = extract.loc[[x in industry_codes for x in extract.index], :].values.T
        metadata[code] = name

    # -------------------------------------------------------------------------
    # Write the start and end year as a single line to a text file and append
    # the data
    with open('data.txt', 'w') as f:
        f.write('{} {}\n\n'.format(tables['Year'].min(), tables['Year'].max()))
        write_data(f, data)
