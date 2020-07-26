# -*- coding: utf-8 -*-
"""
test_iomodel
============
Test suite for `iomodel` module.

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

import io
import os
from typing import Any, Dict
import unittest

import numpy as np

import iomodel


class TestIO(unittest.TestCase):

    def test_roundtrip_buffer(self):
        # Check that writing to a buffer and reading back in returns the
        # original data
        expected = {
            'A': np.arange( 6, dtype=float),
            'B': np.arange(12, dtype=float).reshape((3, 4)),
            'C': np.arange(24, dtype=float).reshape((2, 3, 4)),
            'D': np.arange(120, dtype=float).reshape((2, 3, 4, 5)),
        }

        buffer = io.StringIO()
        iomodel.write_data(buffer, expected)

        buffer.seek(0)
        result = iomodel.read_data(buffer)

        # Compare keys and contents
        self.assertEqual(result.keys(), expected.keys())

        for k in result.keys():
            r = result[k]
            x = expected[k]

            self.assertEqual(r.shape, x.shape)
            self.assertTrue(np.allclose(r, x))

    def test_roundtrip_disk(self):
        # Check that writing to disk and reading back in returns the original
        # data
        expected = {
            'A': np.arange(  6, dtype=float),
            'B': np.arange( 12, dtype=float).reshape((3, 4)),
            'C': np.arange( 24, dtype=float).reshape((2, 3, 4)),
            'D': np.arange(120, dtype=float).reshape((2, 3, 4, 5)),
        }

        iomodel.write_data('test.txt', expected)
        result = iomodel.read_data('test.txt')

        # Compare keys and contents
        self.assertEqual(result.keys(), expected.keys())

        for k in result.keys():
            r = result[k]
            x = expected[k]

            self.assertEqual(r.shape, x.shape)
            self.assertTrue(np.allclose(r, x))

        if os.path.exists('test.txt'):
            os.remove('test.txt')


class TestMultiDimensionalContainer(unittest.TestCase):

    def test_setattr_replace(self):
        # Check array replacement
        container = iomodel.MultiDimensionalContainer()
        container.add_variable('A', np.zeros((2, 3)))

        container.A = np.ones((2, 3))
        self.assertEqual(container.A.shape, (2, 3))
        self.assertTrue(np.allclose(container.A, 1))

        container.A = 2
        self.assertEqual(container.A.shape, (2, 3))
        self.assertTrue(np.allclose(container.A, 2))

    def test_get_set_item(self):
        # Check object indexing (`__getitem__()` and `__setitem__()`)
        container = iomodel.MultiDimensionalContainer()
        container.add_variable('A', np.full((2, 3, 4), 0.5))

        self.assertEqual(container['A'].shape, (2, 3, 4))
        self.assertTrue(np.allclose(container['A'], 0.5))

        container['A'] = 1
        self.assertEqual(container['A'].shape, (2, 3, 4))
        self.assertTrue(np.allclose(container['A'], 1.0))

        container['A'] = np.full((2, 3, 4), 1.5)
        self.assertEqual(container['A'].shape, (2, 3, 4))
        self.assertTrue(np.allclose(container['A'], 1.5))

    def test_key_errors(self):
        # Check that unrecognised variable names are caught
        container = iomodel.MultiDimensionalContainer()

        with self.assertRaises(KeyError):
            _ = container['A']

        with self.assertRaises(KeyError):
            container['A'] = 5

    def test_duplicate_name_error(self):
        # Check that adding duplicate names raises an error
        container = iomodel.MultiDimensionalContainer()
        container.add_variable('A', np.zeros((2, 3)))

        with self.assertRaises(iomodel.DuplicateNameError):
            container.add_variable('A', np.zeros((3, 4)))

    def test_dimension_error(self):
        # Check that array replacement fails if the dimensions differ
        container = iomodel.MultiDimensionalContainer()
        container.add_variable('A', np.zeros((2, 3)))

        with self.assertRaises(iomodel.DimensionError):
            container.A = np.ones((3, 4))

    def test_copy(self):
        # Check that modifying a copy leaves the original unchanged
        original = iomodel.MultiDimensionalContainer()
        original.add_variable('A', np.zeros((2, 3)))

        copy = original.copy()
        copy.A = 1

        self.assertEqual(original.A.shape, (2, 3))
        self.assertTrue(np.allclose(original.A, 0))

        self.assertEqual(copy.A.shape, (2, 3))
        self.assertTrue(np.allclose(copy.A, 1))


class TestBaseMDModel(unittest.TestCase):

    def test_init_wrong_shape(self):
        # Check that an initial value of the wrong dimensions raises an
        # exception

        class TestModel(iomodel.BaseMDModel):
            VARIABLES = {'A': (3, 4), 'B': (4, 3)}

        with self.assertRaises(iomodel.DimensionError):
            model = TestModel(range(2), A=np.zeros((2, 3, 4)),
                              # B is of the wrong shape: should be (2, 4, 3)
                              B=np.ones((2, 3, 4)))

    def test_init_undefined_dimension(self):
        # Check for an error if the user specifies a dimension name without
        # defining that dimension

        class TestModel(iomodel.BaseMDModel):
            VARIABLES = {'A': 'Industries'}

        with self.assertRaises(iomodel.UndefinedDimensionError):
            model = TestModel(range(5))

    def test_iter_periods(self):
        # Check that `iter_periods()` handles lags and leads properly

        class TestModel(iomodel.BaseMDModel):
            LAGS = 1   # Should skip the first period
            LEADS = 1  # Should skip the last period

        model = TestModel(list('ABCDEFG'))
        self.assertEqual(list(model.iter_periods()),
                         [(1, 'B'), (2, 'C'), (3, 'D'), (4, 'E'), (5, 'F')])

    def test_setattr(self):
        # Check `BaseMDModel` set attribute behaviour

        class TestModel(iomodel.BaseMDModel):
            DIMENSIONS = {
                'A': ['A', 'B', 'C', 'D'],
                'B': ['X', 'Y', 'Z'],
            }

            VARIABLES = {
                'AB': 'A',
                'CD': ('A', 'B', ),
                'EF': ('B', 'A', 'A', ),
                'GH': 5,
                'IJ': ('A', 2),
            }

        model = TestModel(range(6))

        # Variable dimensions should include the span (6)
        self.assertEqual(model.AB.shape, (6, 4, ))
        self.assertEqual(model.CD.shape, (6, 4, 3))
        self.assertEqual(model.EF.shape, (6, 3, 4, 4))
        self.assertEqual(model.GH.shape, (6, 5, ))
        self.assertEqual(model.IJ.shape, (6, 4, 2))

        self.assertTrue(np.allclose(model.AB, 0.0))
        self.assertTrue(np.allclose(model.CD, 0.0))
        self.assertTrue(np.allclose(model.EF, 0.0))
        self.assertTrue(np.allclose(model.GH, 0.0))
        self.assertTrue(np.allclose(model.IJ, 0.0))

        # Assign new values
        model.AB = 1
        model.CD = 2
        model.EF = 3
        model.GH = 4
        model.IJ = 5

        self.assertTrue(np.allclose(model.AB, 1.0))
        self.assertTrue(np.allclose(model.CD, 2.0))
        self.assertTrue(np.allclose(model.EF, 3.0))
        self.assertTrue(np.allclose(model.GH, 4.0))
        self.assertTrue(np.allclose(model.IJ, 5.0))

        # Variable dimensions should be unchanged
        self.assertEqual(model.AB.shape, (6, 4, ))
        self.assertEqual(model.CD.shape, (6, 4, 3))
        self.assertEqual(model.EF.shape, (6, 3, 4, 4))
        self.assertEqual(model.GH.shape, (6, 5, ))
        self.assertEqual(model.IJ.shape, (6, 4, 2))

        model.AB[:] = 6
        self.assertTrue(np.allclose(model.AB, 6.0))

        model.AB[0, :] = 7
        model.AB[:, 2] = 8
        self.assertEqual(model.AB.shape, (6, 4, ))

        expected = np.array([
            [7.0, 7.0, 8.0, 7.0, ],
            [6.0, 6.0, 8.0, 6.0, ],
            [6.0, 6.0, 8.0, 6.0, ],
            [6.0, 6.0, 8.0, 6.0, ],
            [6.0, 6.0, 8.0, 6.0, ],
            [6.0, 6.0, 8.0, 6.0, ], ])
        self.assertTrue(np.allclose(model.AB, expected))

    def test_getitem(self):
        # Check `__getitem__()` indexing

        class TestModel(iomodel.BaseMDModel):
            VARIABLES = {'X': (2, 3), }

        model = TestModel(range(2005, 2010 + 1),
                          X=np.arange(36).reshape((6, 2, 3)))

        # Access an entire variable
        self.assertEqual(model['X'].shape, (6, 2, 3))
        self.assertTrue(np.allclose(model['X'], np.arange(36).reshape((6, 2, 3))))

        # Access a variable for a single period
        self.assertEqual(model['X', 2006].shape, (2, 3))
        self.assertTrue(np.allclose(model['X', 2006],
                                    np.array([[6, 7, 8], [9, 10, 11]])))

        # Access a variable with a period slice
        self.assertEqual(model['X', 2006:2008].shape, (3, 2, 3))
        self.assertTrue(np.allclose(model['X', 2006:2008],
                                    np.array([[[ 6,  7,  8], [ 9, 10, 11]],
                                              [[12, 13, 14], [15, 16, 17]],
                                              [[18, 19, 20], [21, 22, 23]]])))

        # Access a variable with a stepped period slice
        self.assertEqual(model['X', 2006:2008:2].shape, (2, 2, 3))
        self.assertTrue(np.allclose(model['X', 2006:2008:2],
                                    np.array([[[ 6,  7,  8], [ 9, 10, 11]],
                                              [[18, 19, 20], [21, 22, 23]]])))

    def test_setitem(self):
        # Check `__setitem__()` indexing

        class TestModel(iomodel.BaseMDModel):
            VARIABLES = {'X': (2, 3), }

        model = TestModel(range(2005, 2010 + 1),
                          X=np.arange(36).reshape((6, 2, 3)))

        # Modify an entire variable
        model['X'] = np.arange(0, 72, 2).reshape((6, 2, 3))
        self.assertEqual(model['X'].shape, (6, 2, 3))
        self.assertTrue(np.allclose(model['X'], np.arange(0, 72, 2).reshape((6, 2, 3))))

        # Modify a variable for a single period
        model['X', 2006] = 0
        self.assertEqual(model['X', 2006].shape, (2, 3))
        self.assertTrue(np.allclose(model['X', 2006], 0.0))

        # Modify a variable with a period slice
        model['X', 2006:2008] = 1
        self.assertEqual(model['X', 2006:2008].shape, (3, 2, 3))
        self.assertTrue(np.allclose(model['X', 2006:2008], 1.0))

        # Modify a variable with a stepped period slice
        model['X', 2006:2008:2] = 2
        self.assertEqual(model['X', 2006:2008:2].shape, (2, 2, 3))
        self.assertTrue(np.allclose(model['X', 2006:2008:2], 2.0))

    def test_extended_get_set_item(self):
        # Check extended `__getitem__()` and `__setitem__()` indexing (beyond
        # the name and period label)

        class TestModel(iomodel.BaseMDModel):
            VARIABLES = {'X': (2, 3), }

        model = TestModel(range(2000, 2003 + 1), X=np.arange(24).reshape((4, 2, 3)))

        # Check initial values
        self.assertEqual(model.X.shape, (4, 2, 3))
        self.assertTrue(np.allclose(model['X'],
                                    np.array([[[ 0.0,  1.0,  2.0, ],
                                               [ 3.0,  4.0,  5.0, ], ],
                                              [[ 6.0,  7.0,  8.0, ],
                                               [ 9.0, 10.0, 11.0, ], ],
                                              [[12.0, 13.0, 14.0, ],
                                               [15.0, 16.0, 17.0, ], ],
                                              [[18.0, 19.0, 20.0, ],
                                               [21.0, 22.0, 23.0, ], ], ])))

        # Replace an entire period of data
        model['X', 2000] = 0.0
        self.assertTrue(np.allclose(model['X', 2000], 0.0))
        self.assertEqual(model.X.shape, (4, 2, 3))
        self.assertTrue(np.allclose(model['X'],
                                    np.array([[[ 0.0,  0.0,  0.0, ],
                                               [ 0.0,  0.0,  0.0, ], ],
                                              [[ 6.0,  7.0,  8.0, ],
                                               [ 9.0, 10.0, 11.0, ], ],
                                              [[12.0, 13.0, 14.0, ],
                                               [15.0, 16.0, 17.0, ], ],
                                              [[18.0, 19.0, 20.0, ],
                                               [21.0, 22.0, 23.0, ], ], ])))

        # Replace an entire row of data for a single period
        model['X', 2000, 1] = 1.0
        self.assertTrue(np.allclose(model['X', 2000, 1], 1.0))
        self.assertEqual(model.X.shape, (4, 2, 3))
        self.assertTrue(np.allclose(model['X'],
                                    np.array([[[ 0.0,  0.0,  0.0, ],
                                               [ 1.0,  1.0,  1.0, ], ],
                                              [[ 6.0,  7.0,  8.0, ],
                                               [ 9.0, 10.0, 11.0, ], ],
                                              [[12.0, 13.0, 14.0, ],
                                               [15.0, 16.0, 17.0, ], ],
                                              [[18.0, 19.0, 20.0, ],
                                               [21.0, 22.0, 23.0, ], ], ])))

        model['X', 2000, 1, :] = 2.0
        self.assertTrue(np.allclose(model['X', 2000, 1], 2.0))
        self.assertEqual(model.X.shape, (4, 2, 3))
        self.assertTrue(np.allclose(model['X'],
                                    np.array([[[ 0.0,  0.0,  0.0, ],
                                               [ 2.0,  2.0,  2.0, ], ],
                                              [[ 6.0,  7.0,  8.0, ],
                                               [ 9.0, 10.0, 11.0, ], ],
                                              [[12.0, 13.0, 14.0, ],
                                               [15.0, 16.0, 17.0, ], ],
                                              [[18.0, 19.0, 20.0, ],
                                               [21.0, 22.0, 23.0, ], ], ])))

        # Replace an entire column of data for a single period
        model['X', 2001, :, 1] = 3.0
        self.assertTrue(np.allclose(model['X', 2001, :, 1], 3.0))
        self.assertEqual(model.X.shape, (4, 2, 3))
        self.assertTrue(np.allclose(model['X'],
                                    np.array([[[ 0.0,  0.0,  0.0, ],
                                               [ 2.0,  2.0,  2.0, ], ],
                                              [[ 6.0,  3.0,  8.0, ],
                                               [ 9.0,  3.0, 11.0, ], ],
                                              [[12.0, 13.0, 14.0, ],
                                               [15.0, 16.0, 17.0, ], ],
                                              [[18.0, 19.0, 20.0, ],
                                               [21.0, 22.0, 23.0, ], ], ])))

    def test_solve(self):
        # Check solution methods

        class TestModel(iomodel.BaseMDModel):

            VARIABLES = {
                'X': (3, 4),
                'Y': (4, 3),
                'Z': (4, 3),
            }

            def _evaluate(self, t: int, **kwargs: Dict[str, Any]) -> None:
                self._Y[t] = 1 + self._X[t].T * 2 + self._Z[t]

        model = TestModel(range(2), X=0.5, Z=np.full((2, 4, 3), 5.0))

        self.assertEqual(model.X.shape, (2, 3, 4))
        self.assertTrue(np.allclose(model.X, 0.5))

        self.assertEqual(model.Y.shape, (2, 4, 3))
        self.assertTrue(np.allclose(model.Y, 0.0))

        self.assertEqual(model.Z.shape, (2, 4, 3))
        self.assertTrue(np.allclose(model.Z, 5.0))

        model.solve()

        self.assertEqual(model.X.shape, (2, 3, 4))
        self.assertTrue(np.allclose(model.X, 0.5))

        self.assertEqual(model.Y.shape, (2, 4, 3))
        self.assertTrue(np.allclose(model.Y, 7.0))

        self.assertEqual(model.Z.shape, (2, 4, 3))
        self.assertTrue(np.allclose(model.Z, 5.0))

    def test_copy(self):
        # Check that taking a copy and operating on the original leaves the
        # copy unchanged

        class TestModel(iomodel.BaseMDModel):

            VARIABLES = {
                'X': (3, 4),
                'Y': (4, 3),
                'Z': (4, 3),
            }

            def _evaluate(self, t: int, **kwargs: Dict[str, Any]) -> None:
                self._Y[t] = 1 + self._X[t].T * 2 + self._Z[t]

        model = TestModel(range(2), X=0.5, Z=np.full((2, 4, 3), 5.0))
        copy = model.copy()

        self.assertEqual(model.X.shape, (2, 3, 4))
        self.assertTrue(np.allclose(model.X, 0.5))

        self.assertEqual(model.Y.shape, (2, 4, 3))
        self.assertTrue(np.allclose(model.Y, 0.0))

        self.assertEqual(model.Z.shape, (2, 4, 3))
        self.assertTrue(np.allclose(model.Z, 5.0))

        # Copy is identical at this point
        self.assertEqual(copy.X.shape, (2, 3, 4))
        self.assertTrue(np.allclose(copy.X, 0.5))

        self.assertEqual(copy.Y.shape, (2, 4, 3))
        self.assertTrue(np.allclose(copy.Y, 0.0))

        self.assertEqual(copy.Z.shape, (2, 4, 3))
        self.assertTrue(np.allclose(copy.Z, 5.0))

        model.solve()

        self.assertEqual(model.X.shape, (2, 3, 4))
        self.assertTrue(np.allclose(model.X, 0.5))

        self.assertEqual(model.Y.shape, (2, 4, 3))
        self.assertTrue(np.allclose(model.Y, 7.0))

        self.assertEqual(model.Z.shape, (2, 4, 3))
        self.assertTrue(np.allclose(model.Z, 5.0))

        # Copy is unchanged
        self.assertEqual(copy.X.shape, (2, 3, 4))
        self.assertTrue(np.allclose(copy.X, 0.5))

        self.assertEqual(copy.Y.shape, (2, 4, 3))
        self.assertTrue(np.allclose(copy.Y, 0.0))

        self.assertEqual(copy.Z.shape, (2, 4, 3))
        self.assertTrue(np.allclose(copy.Z, 5.0))


if __name__ == '__main__':
    unittest.main()
