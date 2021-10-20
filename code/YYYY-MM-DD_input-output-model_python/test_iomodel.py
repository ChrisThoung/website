# -*- coding: utf-8 -*-
"""
test_iomodel
============
Test suite for `iomodel` module.

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

import copy
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

    def test_size(self):
        # Check that `size` returns the total number of array elements
        container = iomodel.MultiDimensionalContainer()
        container.add_variable('A', np.zeros((2, 3), dtype=float))
        container.add_variable('B', np.zeros((3, 4), dtype=float))
        container.add_variable('C', np.zeros((4, 5), dtype=float))
        container.add_variable('D', np.zeros((5, 6), dtype=float))

        self.assertEqual(container.size, sum([2 * 3, 3 * 4, 4 * 5, 5 * 6]))

    def test_nbytes(self):
        # Check that `nbytes` returns the total bytes consumed by the array
        # elements
        container = iomodel.MultiDimensionalContainer()
        container.add_variable('A', np.zeros((2, 3), dtype=float))
        container.add_variable('B', np.zeros((3, 4), dtype=float))
        container.add_variable('C', np.zeros((4, 5), dtype=float))
        container.add_variable('D', np.zeros((5, 6), dtype=float))

        self.assertEqual(container.nbytes, sum([2 * 3, 3 * 4, 4 * 5, 5 * 6]) * 8)

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

    def test_contains(self):
        # Test `in` (membership) operator
        container = iomodel.MultiDimensionalContainer()
        container.add_variable('A', np.full((2, 3, 4), 0.5))

        self.assertIn('A', container)
        self.assertNotIn('B', container)

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

    def test_nbytes(self):
        # Check that `nbytes` returns the total bytes consumed by the array
        # elements

        class TestModel(iomodel.BaseMDModel):
            VARIABLES = {'A': ((3, 4), float, 0.0),
                         'B': ((4, 3), float, 0.0), }

        model = TestModel(range(20))

        self.assertEqual(model.size, 480)
        self.assertEqual(model.nbytes,
                         (480 * 8) +  # Array elements
                         (20 * 8) +   # Iterations
                         (20 * 4))    # Status

    def test_init_wrong_shape(self):
        # Check that an initial value of the wrong dimensions raises an
        # exception

        class TestModel(iomodel.BaseMDModel):
            VARIABLES = {'A': ((3, 4), float, 0.0),
                         'B': ((4, 3), float, 0.0), }

        with self.assertRaises(iomodel.DimensionError):
            model = TestModel(range(2), A=np.zeros((2, 3, 4)),
                              # B is of the wrong shape: should be (2, 4, 3)
                              B=np.ones((2, 3, 4)))

    def test_init_undefined_dimension(self):
        # Check for an error if the user specifies a dimension name without
        # defining that dimension

        class TestModel(iomodel.BaseMDModel):
            VARIABLES = {'A': ('Industries', float, 0.0), }

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
                'AB': ('A', float, 0.0),
                'CD': (('A', 'B', ), float, 0.0),
                'EF': (('B', 'A', 'A', ), float, 0.0),
                'GH': (5, float, 0.0),
                'IJ': (('A', 2), float, 0.0),
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
            VARIABLES = {'X': ((2, 3), float, 0.0), }

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
            VARIABLES = {'X': ((2, 3), float, 0.0), }

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
            VARIABLES = {'X': ((2, 3), float, 0.0), }

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

    def test_size(self):
        # Check that the model correctly counts the number of elements in the
        # variable arrays

        class TestModel(iomodel.BaseMDModel):
            VARIABLES = {
                'X': ((3, 4), float, 0.0),
                'Y': ((4, 3), float, 0.0),
                'Z': ((4, 3), float, 0.0),
            }

        model = TestModel(range(10))
        self.assertEqual(model.size, 360)

    def test_contains(self):
        # Test `in` (membership) operator

        class TestModel(iomodel.BaseMDModel):
            VARIABLES = {
                'X': ((3, 4), float, 0.0),
                'Y': ((4, 3), float, 0.0),
                'Z': ((4, 3), float, 0.0),
            }

        model = TestModel(range(10))

        for name in ['X', 'Y', 'Z']:
            with self.subTest(name=name):
                self.assertIn(name, model)

        self.assertNotIn('A', model)

        self.assertNotIn('status', model)
        self.assertNotIn('iterations', model)

    def test_solve(self):
        # Check solution methods

        class TestModel(iomodel.BaseMDModel):
            VARIABLES = {
                'X': ((3, 4), float, 0.0),
                'Y': ((4, 3), float, 0.0),
                'Z': ((4, 3), float, 0.0),
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

    def test_copy_method(self):
        # Check that taking a copy and operating on the original leaves the
        # copy unchanged

        class TestModel(iomodel.BaseMDModel):
            VARIABLES = {
                'X': ((3, 4), float, 0.0),
                'Y': ((4, 3), float, 0.0),
                'Z': ((4, 3), float, 0.0),
            }

            def _evaluate(self, t: int, **kwargs: Dict[str, Any]) -> None:
                self._Y[t] = 1 + self._X[t].T * 2 + self._Z[t]

        model = TestModel(range(2), X=0.5, Z=np.full((2, 4, 3), 5.0))
        copied = model.copy()

        self.assertEqual(model.X.shape, (2, 3, 4))
        self.assertTrue(np.allclose(model.X, 0.5))

        self.assertEqual(model.Y.shape, (2, 4, 3))
        self.assertTrue(np.allclose(model.Y, 0.0))

        self.assertEqual(model.Z.shape, (2, 4, 3))
        self.assertTrue(np.allclose(model.Z, 5.0))

        # Copy is identical at this point
        self.assertEqual(copied.X.shape, (2, 3, 4))
        self.assertTrue(np.allclose(copied.X, 0.5))

        self.assertEqual(copied.Y.shape, (2, 4, 3))
        self.assertTrue(np.allclose(copied.Y, 0.0))

        self.assertEqual(copied.Z.shape, (2, 4, 3))
        self.assertTrue(np.allclose(copied.Z, 5.0))

        model.solve()

        self.assertEqual(model.X.shape, (2, 3, 4))
        self.assertTrue(np.allclose(model.X, 0.5))

        self.assertEqual(model.Y.shape, (2, 4, 3))
        self.assertTrue(np.allclose(model.Y, 7.0))

        self.assertEqual(model.Z.shape, (2, 4, 3))
        self.assertTrue(np.allclose(model.Z, 5.0))

        # Copy is unchanged
        self.assertEqual(copied.X.shape, (2, 3, 4))
        self.assertTrue(np.allclose(copied.X, 0.5))

        self.assertEqual(copied.Y.shape, (2, 4, 3))
        self.assertTrue(np.allclose(copied.Y, 0.0))

        self.assertEqual(copied.Z.shape, (2, 4, 3))
        self.assertTrue(np.allclose(copied.Z, 5.0))

    def test_copy_dunder_method(self):
        # Check that taking a copy and operating on the original leaves the
        # copy unchanged

        class TestModel(iomodel.BaseMDModel):

            VARIABLES = {
                'X': ((3, 4), float, 0.0),
                'Y': ((4, 3), float, 0.0),
                'Z': ((4, 3), float, 0.0),
            }

            def _evaluate(self, t: int, **kwargs: Dict[str, Any]) -> None:
                self._Y[t] = 1 + self._X[t].T * 2 + self._Z[t]

        model = TestModel(range(2), X=0.5, Z=np.full((2, 4, 3), 5.0))
        copied = copy.copy(model)

        self.assertEqual(model.X.shape, (2, 3, 4))
        self.assertTrue(np.allclose(model.X, 0.5))

        self.assertEqual(model.Y.shape, (2, 4, 3))
        self.assertTrue(np.allclose(model.Y, 0.0))

        self.assertEqual(model.Z.shape, (2, 4, 3))
        self.assertTrue(np.allclose(model.Z, 5.0))

        # Copy is identical at this point
        self.assertEqual(copied.X.shape, (2, 3, 4))
        self.assertTrue(np.allclose(copied.X, 0.5))

        self.assertEqual(copied.Y.shape, (2, 4, 3))
        self.assertTrue(np.allclose(copied.Y, 0.0))

        self.assertEqual(copied.Z.shape, (2, 4, 3))
        self.assertTrue(np.allclose(copied.Z, 5.0))

        model.solve()

        self.assertEqual(model.X.shape, (2, 3, 4))
        self.assertTrue(np.allclose(model.X, 0.5))

        self.assertEqual(model.Y.shape, (2, 4, 3))
        self.assertTrue(np.allclose(model.Y, 7.0))

        self.assertEqual(model.Z.shape, (2, 4, 3))
        self.assertTrue(np.allclose(model.Z, 5.0))

        # Copy is unchanged
        self.assertEqual(copied.X.shape, (2, 3, 4))
        self.assertTrue(np.allclose(copied.X, 0.5))

        self.assertEqual(copied.Y.shape, (2, 4, 3))
        self.assertTrue(np.allclose(copied.Y, 0.0))

        self.assertEqual(copied.Z.shape, (2, 4, 3))
        self.assertTrue(np.allclose(copied.Z, 5.0))

    def test_deepcopy_dunder_method(self):
        # Check that taking a copy and operating on the original leaves the
        # copy unchanged

        class TestModel(iomodel.BaseMDModel):

            VARIABLES = {
                'X': ((3, 4), float, 0.0),
                'Y': ((4, 3), float, 0.0),
                'Z': ((4, 3), float, 0.0),
            }

            def _evaluate(self, t: int, **kwargs: Dict[str, Any]) -> None:
                self._Y[t] = 1 + self._X[t].T * 2 + self._Z[t]

        model = TestModel(range(2), X=0.5, Z=np.full((2, 4, 3), 5.0))
        copied = copy.deepcopy(model)

        self.assertEqual(model.X.shape, (2, 3, 4))
        self.assertTrue(np.allclose(model.X, 0.5))

        self.assertEqual(model.Y.shape, (2, 4, 3))
        self.assertTrue(np.allclose(model.Y, 0.0))

        self.assertEqual(model.Z.shape, (2, 4, 3))
        self.assertTrue(np.allclose(model.Z, 5.0))

        # Copy is identical at this point
        self.assertEqual(copied.X.shape, (2, 3, 4))
        self.assertTrue(np.allclose(copied.X, 0.5))

        self.assertEqual(copied.Y.shape, (2, 4, 3))
        self.assertTrue(np.allclose(copied.Y, 0.0))

        self.assertEqual(copied.Z.shape, (2, 4, 3))
        self.assertTrue(np.allclose(copied.Z, 5.0))

        model.solve()

        self.assertEqual(model.X.shape, (2, 3, 4))
        self.assertTrue(np.allclose(model.X, 0.5))

        self.assertEqual(model.Y.shape, (2, 4, 3))
        self.assertTrue(np.allclose(model.Y, 7.0))

        self.assertEqual(model.Z.shape, (2, 4, 3))
        self.assertTrue(np.allclose(model.Z, 5.0))

        # Copy is unchanged
        self.assertEqual(copied.X.shape, (2, 3, 4))
        self.assertTrue(np.allclose(copied.X, 0.5))

        self.assertEqual(copied.Y.shape, (2, 4, 3))
        self.assertTrue(np.allclose(copied.Y, 0.0))

        self.assertEqual(copied.Z.shape, (2, 4, 3))
        self.assertTrue(np.allclose(copied.Z, 5.0))

    def test_add_variable(self):
        # Check `add_variable()` correctly extends the model store, leaving
        # other model instances unchanged

        class TestModel(iomodel.BaseMDModel):

            VARIABLES = {
                'A': ((2, 3), float, 0.0),
                'B': ((2, 4), float, 0.0),
                'C': ((3, 4), float, 0.0),
            }

        model = TestModel(range(5))
        self.assertEqual(model.names, ['A', 'B', 'C'])
        self.assertEqual(model.size, 130)

        # Copy should be identical to the original
        copied = model.copy()
        self.assertEqual(copied.names, ['A', 'B', 'C'])
        self.assertEqual(copied.size, 130)

        # Add new variables to the original
        model.add_variable('D', np.zeros((5, 2, 5)))
        self.assertEqual(model.names, ['A', 'B', 'C', 'D'])
        self.assertEqual(model.size, 180)

        # `broadcast=True` expands the variable to match the model's `span`
        model.add_variable('E', np.zeros((3, 5)), broadcast=True)
        self.assertEqual(model.names, ['A', 'B', 'C', 'D', 'E'])
        self.assertEqual(model.size, 255)

        # Copy should remain unchanged
        self.assertEqual(copied.names, ['A', 'B', 'C'])
        self.assertEqual(copied.size, 130)

        # A completely new instance should match both the original and the copy
        separate = TestModel(range(5))
        self.assertEqual(separate.names, ['A', 'B', 'C'])
        self.assertEqual(separate.size, 130)

    def test_add_variable_span_error(self):
        # Check that `add_variable()` raises an exception if the first
        # dimension is of a different length to the model's `span`
        model = iomodel.BaseMDModel(range(10))

        with self.assertRaises(iomodel.DimensionError):
            model.add_variable('A', np.zeros((3, 4)))

    def test_add_variable_inconsistent_number_of_dimensions_error(self):
        # Check that `add_variable()` raises an exception if the number of
        # stated dimensions is different to the number of actual dimensions
        model = iomodel.BaseMDModel(range(10))

        with self.assertRaises(iomodel.DimensionError):
            model.add_variable('A', np.zeros((10, 2, 3)), dimensions=['A', 'B', 'C'])

    def test_add_variable_wrong_dimension_size_error(self):
        # Check that `add_variable()` raises an exception if the stated
        # dimension length differs from the actual one
        model = iomodel.BaseMDModel(range(10))

        with self.assertRaises(iomodel.DimensionError):
            model.add_variable('A', np.zeros((10, 2, 3)), dimensions=(2, 4))

    def test_add_variable_unrecognised_dimension_error(self):
        # Check that `add_variable()` raises an exception if the stated
        # dimension labels are missing from the model's `dimensions` variable
        model = iomodel.BaseMDModel(range(10))

        with self.assertRaises(iomodel.DimensionError):
            model.add_variable('A', np.zeros((10, 2, 3)), dimensions=(2, 'X'))

    def test_solve_offset(self):
        # Check that the `offset` argument copies values into the current
        # period
        class TestModel(iomodel.BaseMDModel):
            VARIABLES = {'A': ((3, 4), float, 0.0),
                         'B': ((3, 4), float, 0.0), }
            ENDOGENOUS = ['A', 'B']
            LAGS = 1

            def _evaluate(self, t):
                self.A[t] = self.B[t]
                self.B[t] *= 1

        model = TestModel(['2000Q{}'.format(i) for i in range(1, 4 + 1)])

        model.B = 5
        model.solve()

        self.assertTrue(np.allclose(model.A[0], 0))
        self.assertTrue(np.allclose(model.A[1:], 5))

        self.assertTrue(np.allclose(model.B, 5))

        model.B[0] = 10
        self.assertTrue(np.allclose(model.B[0], 10))
        self.assertTrue(np.allclose(model.B[1:], 5))

        model.solve(offset=-1)

        self.assertTrue(np.allclose(model.A[0], 0))
        self.assertTrue(np.allclose(model.A[1:], 10))

        self.assertTrue(np.allclose(model.B, 10))


if __name__ == '__main__':
    unittest.main()
