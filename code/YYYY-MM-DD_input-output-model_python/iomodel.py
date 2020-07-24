# -*- coding: utf-8 -*-
"""
iomodel
=======
(Experimental) tools for macroeconomic input-output modelling.

Large chunks of the `MultiDimensionalContainer` and `BaseMDModel` code have
been adapted from the corresponding `VectorContainer` and `BaseModel` classes
of FSIC (version 0.6.2):

    https://github.com/ChrisThoung/fsic/tree/v0.6.2.dev

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

__version__ = '0.1.0.dev'


import copy
from typing import Any, Dict, Hashable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np


# Custom exceptions -----------------------------------------------------------

class IOModelError(Exception):
    pass

class DimensionError(IOModelError):
    pass

class DuplicateNameError(IOModelError):
    pass

class NonConvergenceError(IOModelError):
    pass

class UndefinedDimensionError(IOModelError):
    pass


# Rudimentary I/O -------------------------------------------------------------

def write_data(path_or_buffer: Any, data: Dict[str, np.ndarray]) -> None:
    """Write the contents of `data` to `path_or_buffer`.

    See notes for output format.

    Parameters
    ----------
    path_or_buffer : str (filepath) or file-like (buffer for writing)
        Where to write the data
    data : dictionary mapping variable names (keys) to data (values)
        Data to write

    Notes
    -----
    This function writes to a text file, with three lines per variable:
    1. the name of the variable
    2. the shape of the variable, as a sequence of integers separated by
       whitespace
    3. the (flattened) data, as a sequence of floats separated by whitespace

    This function writes a blank line to separate each variable but these are
    not strictly necessary for reading the data back in (see this function's
    counterpart, `read_data()`).
    """
    # Open the file for writing if needed
    if hasattr(path_or_buffer, 'write'):
        buffer = path_or_buffer
    else:
        buffer = open(path_or_buffer, 'w')

    # Write the data one variable at a time
    for name, values in data.items():
        buffer.write(name)
        buffer.write('\n')

        buffer.write(' '.join(map(str, values.shape)))
        buffer.write('\n')

        buffer.write(' '.join(map(str, values.flatten())))
        buffer.write('\n\n')

    # Close the file if needed
    if not hasattr(path_or_buffer, 'write'):
        buffer.close()

def read_data(path_or_buffer: Any) -> Dict[str, np.ndarray]:
    """Return the contents of `path_or_buffer`.

    Parameters
    ----------
    path_or_buffer : str (filepath) or file-like (buffer)
        Where to read the data from. See notes for details of the expected
        format

    Returns
    -------
    data : dictionary mapping variable names (keys) to data (values)

    Notes
    -----
    This function reads text files (e.g. those created by its counterpart,
    `write_data()`), extracting variables by reading three lines at a time:
    1. the name of the variable: a string
    2. the shape of the variable, interpreted as a sequence of integers
       separated by whitespace
    3. the data, as a sequence of floats separated by whitespace and then
       reshaped according to [2]

    Lines of whitespace are ignored.
    """
    # Open the file for reading if needed
    if hasattr(path_or_buffer, 'read'):
        buffer = path_or_buffer
    else:
        buffer = open(path_or_buffer)

    # Read the data
    data = {}

    for line in filter(len, map(str.strip, buffer)):
        name = line
        shape = tuple(map(int, next(buffer).split()))
        data[name] = np.array(tuple(map(float, next(buffer).split()))).reshape(shape)

    # Close the file if needed
    if not hasattr(path_or_buffer, 'read'):
        buffer.close()

    return data


# Data container for NumPy arrays of arbitrary dimensions ---------------------

class MultiDimensionalContainer:

    def __init__(self) -> None:
        self.__dict__['index'] = []

    def add_variable(self, name: str, array: np.ndarray) -> None:
        """Initialise a new variable in the container.

        Parameters
        ----------
        name : str
            The name of the new variable, which must not already exist in the
            container (raise a `DuplicateNameError` if the name already exists)
        array : NumPy ndarray
            Initial values for the new variable
        """
        if name in self.__dict__['index']:
            raise DuplicateNameError(
                "'{}' already defined in the current object".format(name))

        self.__dict__['_' + name] = array
        self.__dict__['index'].append(name)

    def __getattr__(self, name: str) -> Any:
        if name in self.__dict__['index']:
            return self.__dict__['_' + name]
        else:
            return self.__getattribute__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name not in self.__dict__['index']:
            super.__setattr__(self, name, value)
            return

        if isinstance(value, np.ndarray):
            if value.shape != self.__dict__['_' + name].shape:
                raise DimensionError(
                    "Invalid assignment for '{}': "
                    "must be either a single value or "
                    "an array of expected dimensions {}".format(
                        name, self.__dict__['_' + name].shape))
            self.__dict__['_' + name] = np.array(value,
                                                 dtype=self.__dict__['_' + name].dtype)
        else:
            self.__dict__['_' + name][:] = value

    def __getitem__(self, key: str) -> Any:
        if key not in self.__dict__['index']:
            raise KeyError("'{}' not recognised as a variable name".format(key))

        return self.__getattr__(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if key not in self.__dict__['index']:
            raise KeyError("'{}' not recognised as a variable name".format(key))

        self.__setattr__(key, value)

    def copy(self) -> 'MultiDimensionalContainer':
        """Return a copy of the current object."""
        copied = self.__class__()
        copied.__dict__.update(
            {k: copy.deepcopy(v)
             for k, v in self.__dict__.items()})
        return copied

    __copy__ = copy
    __deepcopy__ = copy

    def __dir__(self) -> List[str]:
        return sorted(
            dir(type(self)) + self.__dict__['index'] + ['index'])

    def _ipython_key_completions_(self) -> List[str]:
        return self.__dict__['index']


# Base class for individual models --------------------------------------------

class BaseMDModel(MultiDimensionalContainer):

    DIMENSIONS: Dict[str, Sequence[str]] = {}
    VARIABLES: Dict[str, Union[str, Sequence[Union[str, int]], int]] = {}

    ENDOGENOUS: List[str] = []
    CHECK: List[str] = ENDOGENOUS

    LAGS: int = 0
    LEADS: int = 0

    def __init__(self, span: Sequence[Hashable], *, dtype: Any = float, default_value: Union[int, float] = 0.0, **initial_values: Dict[str, Any]) -> None:
        """

        Parameters
        ----------
        span : iterable
            Sequence of periods that define the timespan of the model
        dtype : variable type
            Data type to impose on model variables (in NumPy arrays)
        default_value : number
            Default value with which to initialise model variables (if not
            specified in `initial_values`)
        **initial_values : keyword arguments of variable names and values
            Values with which to initialise specific named model variables
        """
        super().__init__()
        self.__dict__['span'] = span

        # Add solution tracking variables
        self.add_variable('status', np.full(len(self.__dict__['span']), '-', dtype=str))
        self.add_variable('iterations', np.full(len(self.__dict__['span']), -1, dtype=int))

        # Initialise model variables
        dimension_lengths = {k: len(v) for k, v in self.DIMENSIONS.items()}

        self.__dict__['names'] = []
        for name, dimensions in self.VARIABLES.items():
            # If not already a sequence, convert to a one-element list
            if isinstance(dimensions, str) or not isinstance(dimensions, Sequence):
                dimensions = [dimensions]

            # Get initial value
            value = initial_values.get(name, default_value)

            # Calculate variable shape
            shape = [len(self.__dict__['span'])]  # Span of the model

            for dim in dimensions:
                # Integer: no labels; just allocate the space
                if isinstance(dim, int):
                    shape.append(dim)

                # String: named dimension; get from `DIMENSIONS`
                elif isinstance(dim, str):
                    # Error if dimension not defined in `DIMENSIONS`
                    if dim not in dimension_lengths:
                        raise UndefinedDimensionError(
                            'Unrecognised dimension name '
                            '(not defined in `DIMENSIONS` attribute): {}'.format(dim))

                    shape.append(dimension_lengths[dim])

                else:
                    raise TypeError(
                        "Unrecognised type for dimension '{}': {}".format(dim, type(dim)))

            shape = tuple(shape)

            # If an array, check shape is as expected...
            if isinstance(value, np.ndarray):
                if value.shape != shape:
                    raise DimensionError(
                        "Invalid assignment for '{}': "
                        "must be either a single value or "
                        "an array of expected dimensions: "
                        "expected {} but found {}".format(
                            name, shape, value.shape))
            # ...otherwise, create an array of the expected shape
            else:
                value = np.full(shape, value, dtype=dtype)

            self.add_variable(name, value)
            self.__dict__['names'].append(name)

    def __getitem__(self, key: Union[str, Tuple[str, Union[Hashable, slice]]]) -> Any:
        # `key` is a string (variable name): return the corresponding array
        if isinstance(key, str):
            if key not in self.__dict__['index']:
                raise KeyError("'{}' not recognised as a variable name".format(key))
            return self.__getattr__(key)

        # `key` is a tuple (variable name plus index): return the selected
        # elements of the corresponding array
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError(
                    'Invalid index: must be of length one (variable name) '
                    'or length two (variable name, span index)')

            # Unpack the key
            name: str
            index: Union[Hashable, slice]
            name, index = key

            # Get the full array
            if name not in self.__dict__['index']:
                raise KeyError("'{}' not recognised as a variable name".format(name))
            values = self.__getattr__(name)

            # Extract and return the relevant subset
            if isinstance(index, slice):
                start, stop, step = index.start, index.stop, index.step

                if start is None:
                    start = self.__dict__['span'][0]
                if stop is None:
                    stop = self.__dict__['span'][-1]
                if step is None:
                    step = 1

                start_location = self.__dict__['span'].index(start)
                stop_location = self.__dict__['span'].index(stop) + 1

                return values[start_location:stop_location:step]

            else:
                location = self.__dict__['span'].index(index)
                return values[location]

        raise TypeError('Invalid index type ({}): `{}`'.format(type(key), key))

    def __setitem__(self, key: Union[str, Tuple[str, Union[Hashable, slice]]], value: Union[Any, Sequence[Any]]) -> None:
        # `key` is a string (variable name): update the corresponding array in
        # its entirety
        if isinstance(key, str):
            if key not in self.__dict__['index']:
                raise KeyError("'{}' not recognised as a variable name".format(key))
            self.__setattr__(key, value)
            return

        # `key` is a tuple (variable name plus index): update selected elements
        # of the corresponding array
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError(
                    'Invalid index: must be of length one (variable name) '
                    'or length two (variable name, span index)')

            # Unpack the key
            name: str
            index: Union[Hashable, slice]
            name, index = key

            # Modify the relevant subset
            if isinstance(index, slice):
                start, stop, step = index.start, index.stop, index.step

                if start is None:
                    start = self.__dict__['span'][0]
                if stop is None:
                    stop = self.__dict__['span'][-1]
                if step is None:
                    step = 1

                start_location = self.__dict__['span'].index(start)
                stop_location = self.__dict__['span'].index(stop) + 1

                self.__dict__['_' + name][start_location:stop_location:step] = value
                return

            else:
                location = self.__dict__['span'].index(index)
                self.__dict__['_' + name][location] = value
                return

        raise TypeError('Invalid index type ({}): `{}`'.format(type(key), key))

    def copy(self) -> 'BaseMDModel':
        """Return a copy of the current object."""
        copied = self.__class__(span=copy.deepcopy(self.__dict__['span']))
        copied.__dict__.update(
            {k: copy.deepcopy(v)
             for k, v in self.__dict__.items()})
        return copied

    __copy__ = copy
    __deepcopy__ = copy

    def __dir__(self) -> List[str]:
        return sorted(super().__dir__() + ['span'])

    def iter_periods(self, *, start: Optional[Hashable] = None, end: Optional[Hashable] = None) -> Iterator[Tuple[int, Hashable]]:
        """Return pairs of period indexes and labels.

        Parameters
        ----------
        start : element in the model's `span`
            First period to return. If not given, defaults to the first
            solvable period, taking into account any lags in the model's
            equations
        end : element in the model's `span`
            Last period to return. If not given, defaults to the last solvable
            period, taking into account any leads in the model's equations
        """
        # Default start and end periods
        if start is None:
            start = self.span[self.LAGS]
        if end is None:
            end = self.span[-1 - self.LEADS]

        # Convert to an integer range
        indexes = range(self.span.index(start),
                        self.span.index(end) + 1)

        for t in indexes:
            yield t, self.span[t]

    def solve(self, *, start: Optional[Hashable] = None, end: Optional[Hashable] = None, max_iter: int = 100, tol: Union[int, float] = 1e-10, failures: str = 'raise', **kwargs: Dict[str, Any]) -> Tuple[List[Hashable], List[int], List[bool]]:
        """Solve the model. Use default periods if none provided.

        Parameters
        ----------
        start : element in the model's `span`
            First period to solve. If not given, defaults to the first solvable
            period, taking into account any lags in the model's equations
        end : element in the model's `span`
            Last period to solve. If not given, defaults to the last solvable
            period, taking into account any leads in the model's equations
        max_iter : int
            Maximum number of iterations to solution each period
        tol : int or float
            Tolerance for convergence
        failures : str, one of {'raise', 'ignore'}
            Action should the solution fail to converge in a period (by
            reaching the maximum number of iterations, `max_iter`)
             - 'raise' (default): stop immediately and raise a
                                  `NonConvergenceError`
             - 'ignore': continue to the next period
        kwargs :
            Further keyword arguments to pass to the solution routines

        Returns
        -------
        Three lists, each of length equal to the number of periods to be
        solved:
         - the names of the periods to be solved, as they appear in the model's
           span
         - integers: the index positions of the above periods in the model's
           span
         - bools, one per period: `True` if the period solved successfully;
           `False` otherwise
        """
        indexes, labels = map(list, zip(*self.iter_periods(start=start, end=end)))
        solved = [False] * len(indexes)

        for i, t in enumerate(indexes):
            solved[i] = self.solve_t(t, max_iter=max_iter, tol=tol, **kwargs)

        return labels, indexes, solved

    def solve_period(self, period: Hashable, *, max_iter: int = 100, tol: Union[int, float] = 1e-10, failures: str = 'raise', **kwargs: Dict[str, Any]) -> bool:
        """Solve a single period.

        Parameters
        ----------
        period : element in the model's `span`
            Named period to solve
        max_iter : int
            Maximum number of iterations to solution each period
        tol : int or float
            Tolerance for convergence
        failures : str, one of {'raise', 'ignore'}
            Action should the solution fail to converge (by reaching the
            maximum number of iterations, `max_iter`)
             - 'raise' (default): stop immediately and raise a
                                  `NonConvergenceError`
             - 'ignore': do nothing
        kwargs :
            Further keyword arguments to pass to the solution routines

        Returns
        -------
        `True` if the model solved for the current period; `False` otherwise.
        """
        t = self.span.index(period)
        return self.solve_t(t, max_iter=max_iter, tol=tol, **kwargs)

    def solve_t(self, t: int, *, max_iter: int = 100, tol: Union[int, float] = 1e-10, failures: str = 'raise', **kwargs: Dict[str, Any]) -> bool:
        """Solve for the period at integer position `t` in the model's `span`.

        Parameters
        ----------
        t : int
            Position in `span` of the period to solve
        max_iter : int
            Maximum number of iterations to solution each period
        tol : int or float
            Tolerance for convergence
        failures : str, one of {'raise', 'ignore'}
            Action should the solution fail to converge (by reaching the
            maximum number of iterations, `max_iter`)
             - 'raise' (default): stop immediately and raise a
                                  `NonConvergenceError`
             - 'ignore': do nothing
        kwargs :
            Further keyword arguments to pass to the solution routines

        Returns
        -------
        `True` if the model solved for the current period; `False` otherwise.
        """
        def get_check_values() -> Dict[str, np.ndarray]:
            """Return a dictionary of variables to check for convergence in the current period."""
            return {x: self[x][t] for x in self.CHECK}

        status = '-'
        current_values = get_check_values()

        for iteration in range(1, max_iter + 1):
            previous_values = copy.deepcopy(current_values)
            self._evaluate(t, **kwargs)
            current_values = get_check_values()

            converged = {x: False for x in current_values.keys()}
            for k in converged.keys():
                if np.all(np.absolute(current_values[k] - previous_values[k]) < tol):
                    converged[k] = True

            if all(converged.values()):
                status = '.'
                break
        else:
            status = 'F'

        self.status[t] = status
        self.iterations[t] = iteration

        if status == 'F' and failures == 'raise':
            raise NonConvergenceError(
                'Solution failed to converge after {} iterations(s) '
                'in period with label: {} (index: {})'
                .format(iteration, self.span[t], t))

        if status == '.':
            return True
        else:
            return False

    def _evaluate(self, t: int, **kwargs: Dict[str, Any]) -> None:
        """Evaluate the system of equations for the period at integer position `t` in the model's `span`.

        Parameters
        ----------
        t : int
            Position in `span` of the period to solve
        kwargs :
            Further keyword arguments for solution
        """
        raise NotImplementedError('Method must be over-ridden by a child class')
