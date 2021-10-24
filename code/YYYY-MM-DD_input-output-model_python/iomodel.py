# -*- coding: utf-8 -*-
"""
iomodel
=======
(Experimental) tools for macroeconomic input-output modelling.

The code below builds on the latest release version (0.7.1.dev) of `fsic`:

    https://github.com/ChrisThoung/fsic/tree/v0.7.1.dev

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

__version__ = '0.1.0.dev'


import copy
from typing import Any, Dict, Hashable, Iterator, List, NamedTuple, Optional, Sequence, Tuple, Union
import warnings

import numpy as np

from fsic import FSICError, DimensionError, DuplicateNameError, NonConvergenceError, SolutionError
from fsic import SolverMixin


# Custom exceptions -----------------------------------------------------------

class IOModelError(FSICError):
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

    @property
    def size(self) -> int:
        """Total number of elements in the object's variable arrays."""
        return sum(self.__dict__['_' + k].size for k in self.__dict__['index'])

    @property
    def nbytes(self) -> int:
        """Total bytes consumed by the elements in the object's arrays."""
        return sum(self.__dict__['_' + k].nbytes for k in self.__dict__['index'])

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

    def __contains__(self, key: str) -> bool:
        return key in self.__dict__['index']

    def copy(self) -> 'MultiDimensionalContainer':
        """Return a copy of the current object."""
        copied = self.__class__()
        copied.__dict__.update(
            {k: copy.deepcopy(v)
             for k, v in self.__dict__.items()})
        return copied

    __copy__ = copy

    def __deepcopy__(self, *args: Any, **kwargs: Any) -> 'MultiDimensionalContainer':
        return self.copy()

    def __dir__(self) -> List[str]:
        return sorted(
            dir(type(self)) + self.__dict__['index'] + ['index'])

    def _ipython_key_completions_(self) -> List[str]:
        return self.__dict__['index']


# Base class for individual models --------------------------------------------

class VariableDefinition(NamedTuple):
    dimensions: Union[str, int, Sequence[Union[str, int]]]
    dtype: Any
    default_value: Any

class BaseMDModel(SolverMixin, MultiDimensionalContainer):

    DIMENSIONS: Dict[str, Sequence[str]] = {}
    VARIABLES: Dict[str, VariableDefinition] = {}

    ENDOGENOUS: List[str] = []
    CHECK: List[str] = ENDOGENOUS

    LAGS: int = 0
    LEADS: int = 0

    def __init__(self, span: Sequence[Hashable], **initial_values: Dict[str, Any]) -> None:
        """Initialise model variables.

        Parameters
        ----------
        span : iterable
            Sequence of periods that define the timespan of the model
        **initial_values : keyword arguments of variable names and values
            Values with which to initialise specific named model variables
        """
        super().__init__()
        self.__dict__['span'] = span

        # Add solution tracking variables
        super().add_variable('status', np.full(len(self.__dict__['span']), '-', dtype=str))
        super().add_variable('iterations', np.full(len(self.__dict__['span']), -1, dtype=int))

        # Initialise model variables
        self.__dict__['dimensions'] = copy.deepcopy(self.DIMENSIONS)
        self.__dict__['variables'] = copy.deepcopy({k: dimensions
                                                    for k, (dimensions, _, _) in self.VARIABLES.items()})
        self.__dict__['names'] = []

        dimension_lengths = {k: len(v) for k, v in self.__dict__['dimensions'].items()}

        for name, (dimensions, dtype, default_value) in self.VARIABLES.items():
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

            super().add_variable(name, value)
            self.__dict__['names'].append(name)

    def add_variable(self, name: str, array: np.ndarray, *, broadcast: bool = False, dimensions: Optional[Sequence[Hashable]] = None) -> None:
        """Add a new variable to the model.

        Parameters
        ----------
        name : str
            The name of the new variable, which must not already exist in the
            container (raise a `DuplicateNameError` if the name already exists)
        array : NumPy ndarray
            Initial values for the new variable
        broadcast : bool
            If `False` (default), take `array` to be complete and insert as-is
            into the object. The first/leftmost dimension must be of identical
            length to the `span` attribute. Raise a `DimensionError` if not.
            If `True`, take `array` to be the shape and values for a single
            period. Repeat (broadcast) `array` as many times as needed to fill
            out `span`.
        dimensions : sequence of dimension names (*excluding* time/span)
            If supplied, store these dimension labels to the `variables`
            attribute. The labels should *exclude* time/span. Raise a
            `DimensionError` if any named dimensions are missing from the
            `dimensions` attribute.
            If not supplied, just store the unlabelled dimension lengths to the
            `variables` attribute.
        """
        # Broadcast the array to match `self.span` as needed
        if broadcast:
            array = np.array([array] * len(self.__dict__['span']))
        else:
            # No broadcasting: Check outer dimension is of the right length
            if array.shape[0] != len(self.__dict__['span']):
                raise DimensionError(
                    "If `broadcast=False` (the default), the length of the first dimension of `array` ({}) "
                    "must match the length of the object's `span` attribute ({}): {} != {}"
                    .format(array.shape[0], len(self.__dict__['span']),
                            array.shape[0], len(self.__dict__['span'])))

        # Generate dimensions as needed
        if dimensions is None:
            dimensions = array.shape[1:]
        else:
            # Dimensions passed: Check validity
            if len(dimensions) != (array.ndim - 1):
                raise DimensionError(
                    '`array` has {} ({}-1) non-`span` dimensions '
                    'but `dimensions` has a different length ({}): '
                    '{} != {}'.format(
                        array.ndim - 1, array.ndim,
                        len(dimensions),
                        array.ndim - 1, len(dimensions)))

            for i, (label, length) in enumerate(zip(dimensions, array.shape[1:])):
                # `label` is an integer (dimension size rather than dimension
                # label): Check it matches the corresponding dimension length
                if isinstance(label, int):
                    if label != length:
                        raise DimensionError(
                            '`dimension[{}]` is an integer with value {} '
                            'but this differs from the corresponding dimension length '
                            'of {} in `array`'.format(i, label, length))

                # Check if `label` is defined in the current instance
                elif label not in self.__dict__['dimensions']:
                    raise DimensionError(
                        "`dimension[{}]` has label '{}' but "
                        "'{}' is not defined in the object's `dimensions` attribute"
                        .format(i, label, label))

        # All fine: Add the variable to the model instance
        super().add_variable(name, array)
        self.__dict__['variables'][name] = dimensions
        self.__dict__['names'].append(name)

    @property
    def size(self) -> int:
        """Total number of elements in the model's variable arrays."""
        return sum(self.__dict__['_' + k].size for k in self.__dict__['names'])

    def __getitem__(self, key: Union[str, Tuple[str, Union[Hashable, slice]]]) -> Any:
        # `key` is a string (variable name): return the corresponding array
        if isinstance(key, str):
            if key not in self.__dict__['index']:
                raise KeyError("'{}' not recognised as a variable name".format(key))
            return self.__getattr__(key)

        # `key` is a tuple (variable name plus index): return the selected
        # elements of the corresponding array
        if isinstance(key, tuple):
            # Unpack `key`
            name: str
            index: Union[Hashable, slice]
            remaining_dims: List

            name, index, *remaining_dims = key

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

                indexes = tuple([slice(start_location, stop_location, step)] + remaining_dims)

            else:
                location = self.__dict__['span'].index(index)
                indexes = tuple([location] + remaining_dims)

            return values[indexes]

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
            # Unpack `key`
            name: str
            index: Union[Hashable, slice]
            remaining_dims: List

            name, index, *remaining_dims = key

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

                indexes = tuple([slice(start_location, stop_location, step)] + remaining_dims)

            else:
                location = self.__dict__['span'].index(index)
                indexes = tuple([location] + remaining_dims)

            self.__dict__['_' + name][indexes] = value
            return

        raise TypeError('Invalid index type ({}): `{}`'.format(type(key), key))

    def __contains__(self, key: str) -> bool:
        return key in self.__dict__['names']

    def copy(self) -> 'BaseMDModel':
        """Return a copy of the current object."""
        copied = self.__class__(span=copy.deepcopy(self.__dict__['span']))
        copied.__dict__.update(
            {k: copy.deepcopy(v)
             for k, v in self.__dict__.items()})
        return copied

    __copy__ = copy

    def __deepcopy__(self, *args: Any, **kwargs: Any) -> 'BaseMDModel':
        return self.copy()

    def __dir__(self) -> List[str]:
        return sorted(super().__dir__() + ['span'])

    def solve_t(self, t: int, *, min_iter: int = 0, max_iter: int = 100, tol: Union[int, float] = 1e-10, offset: int = 0, failures: str = 'raise', errors: str = 'raise', **kwargs: Dict[str, Any]) -> bool:
        """Solve for the period at integer position `t` in the model's `span`.

        Parameters
        ----------
        t : int
            Position in `span` of the period to solve
        min_iter : int
            Minimum number of iterations to solution each period
        max_iter : int
            Maximum number of iterations to solution each period
        tol : int or float
            Tolerance for convergence
        offset : int
            If non-zero, copy an initial set of endogenous values from the
            period at position `t + offset`. For example, `offset=-1` copies
            the values from the previous period.
        failures : str, one of {'raise', 'ignore'}
            Action should the solution fail to converge (by reaching the
            maximum number of iterations, `max_iter`)
             - 'raise' (default): stop immediately and raise a
                                  `NonConvergenceError`
             - 'ignore': do nothing
        errors : str, one of {'raise', 'skip', 'ignore', 'replace'}
            User-specified treatment on encountering numerical solution errors
            e.g. NaNs and infinities
             - 'raise' (default): stop immediately and raise a `SolutionError`
                                  [set period solution status to 'E']
             - 'skip': stop solving the current period
                       [set period solution status to 'S']
             - 'ignore': continue solving, with no special treatment or action
                         [period solution statuses as usual i.e. '.' or 'F']
             - 'replace': each iteration, replace NaNs and infinities with
                          zeroes
                          [period solution statuses as usual i.e. '.' or 'F']
        kwargs :
            Further keyword arguments to pass to the solution routines

        Returns
        -------
        `True` if the model solved for the current period; `False` otherwise.
        """
        def get_check_values() -> Dict[str, np.ndarray]:
            """Return a dictionary of variables to check for convergence in the current period."""
            return {x: self[x][t] for x in self.CHECK}

        # Error if `min_iter` exceeds `max_iter`
        if min_iter > max_iter:
            raise ValueError(
                'Value of `min_iter` ({}) cannot exceed value of `max_iter` ({})'.format(
                    min_iter, max_iter))

        # Optionally copy initial values from another period
        if offset:
            t_check = t
            if t_check < 0:
                t_check += len(self.span)

            # Error if `offset` points prior to the current model span
            if t_check + offset < 0:
                raise IndexError(
                    '`offset` argument ({}) for position `t` ({}) '
                    'implies a period before the span of the current model instance: '
                    '{} + {} -> position {} < 0'.format(
                        offset, t, offset, t, offset + t_check))

            # Error if `offset` points beyond the current model span
            if t_check + offset >= len(self.span):
                raise IndexError(
                    '`offset` argument ({}) for position `t` ({}) '
                    'implies a period beyond the span of the current model instance: '
                    '{} + {} -> position {} >= {} periods in span'.format(
                        offset, t, offset, t, offset + t_check, len(self.span)))

            for name in self.ENDOGENOUS:
                self.__dict__['_' + name][t] = self.__dict__['_' + name][t + offset]

        status = '-'
        current_values = get_check_values()

        for iteration in range(1, max_iter + 1):
            previous_values = copy.deepcopy(current_values)

            with warnings.catch_warnings(record=True) as w:
                if errors == 'raise':
                    warnings.simplefilter('error')
                else:
                    warnings.simplefilter('always')

                try:
                    self._evaluate(t, **kwargs)
                except:
                    self.status[t] = 'E'
                    self.iterations[t] = iteration

                    raise SolutionError(
                        'Error after {} iterations(s) '
                        'in period with label: {} (index: {})'
                        .format(iteration, self.span[t], t))

                # `errors` argument not implemented yet. For now, just fail if
                # evaluation creates any problems
                if len(w):
                    raise NotImplementedError(
                        'Solution encountered {} warning(s) '
                        'after {} iterations(s) '
                        'in period with label: {} (index: {}). '
                        'Solution error handling not fully implemented yet.'
                        .format(len(w), iteration, self.span[t], t))

            current_values = get_check_values()

            if iteration < min_iter:
                continue

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

        return status == '.'

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
