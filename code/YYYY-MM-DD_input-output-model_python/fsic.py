# -*- coding: utf-8 -*-
"""
fsic
====
Tools for macroeconomic modelling in Python.

-------------------------------------------------------------------------------
Copied from:

    https://github.com/ChrisThoung/fsic/tree/v0.7.1.dev

Licence reproduced below.

-------------------------------------------------------------------------------
MIT License

Copyright (c) 2018-21 Chris Thoung

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

__version__ = '0.7.1.dev'


import copy
import enum
import itertools
import keyword
import re
import textwrap
from typing import Any, Callable, Dict, Hashable, Iterator, List, Match, NamedTuple, Optional, Sequence, Tuple, Union
import warnings

import numpy as np


# Custom exceptions -----------------------------------------------------------

class FSICError(Exception):
    pass


class BuildError(FSICError):
    pass

class DimensionError(FSICError):
    pass

class DuplicateNameError(FSICError):
    pass

class EvalError(FSICError):
    pass

class InitialisationError(FSICError):
    pass

class NonConvergenceError(FSICError):
    pass

class ParserError(FSICError):
    pass

class SolutionError(FSICError):
    pass

class SymbolError(FSICError):
    pass


# Compiled regular expressions ------------------------------------------------

# Equations are Python-like statements that can take up a single line:
#
#     C = {alpha_1} * YD + {alpha_2} * H[-1]
#
# or span multiple lines using parentheses:
#
#     C = ({alpha_1} * YD +
#          {alpha_2} * H[-1])
#
#     (C =
#          {alpha_1} * YD +
#          {alpha_2} * H[-1])

equation_re = re.compile(
    r'''
        (?: ^ \( .*?      [=]        .*? \) (?= \s* ) $ )|  # Brackets beginning on the left-hand side
        (?: ^    \S+? \s* [=] \s* \( .*? \) (?= \s* ) $ )|  # Brackets beginning on the right-hand side
        (?: ^    \S+? \s* [=] \s*    .*?    (?= \s* ) $ )   # Equation on a single line
    ''',
    re.DOTALL | re.MULTILINE | re.VERBOSE
)

# Terms are valid Python identifiers. If they aren't functions, they may also
# have an index. For example:
#
#     C
#     H_d
#     V[-1]
#     exp()
#
# Enclose parameters in braces: {alpha_1}
#
# and errors in angle brackets: <epsilon>

term_re = re.compile(
    # Match attempts to use reserved Python keywords as variable names (to be
    # raised as errors elsewhere)
    r'(?: (?P<_INVALID> (?: {}) \s* \[ .*? \]) )|'.format('|'.join(keyword.kwlist)) +

    # Match Python keywords
    r'(?: \b (?P<_KEYWORD> {} ) \b )|'.format('|'.join(keyword.kwlist)) +

    # Valid terms for the parser
    r'''
        (?: (?P<_FUNCTION> [_A-Za-z][_A-Za-z0-9.]*[_A-Za-z0-9]* ) \s* (?= \( ) )|

        (?:
            (?: \{ \s* (?P<_PARAMETER> [_A-Za-z][_A-Za-z0-9]* ) \s* \} )|
            (?: \< \s* (?P<_ERROR>     [_A-Za-z][_A-Za-z0-9]* ) \s* \> )|
            (?:        (?P<_VARIABLE>  [_A-Za-z][_A-Za-z0-9]* )        )
        )
        (?: \[ \s* (?P<INDEX> .*? ) \s* \] )?
    ''',
    re.VERBOSE
)


# Containers for term and symbol information ----------------------------------

class Type(enum.IntEnum):
    """Enumeration to track term/symbol types."""
    VARIABLE = enum.auto()
    EXOGENOUS = enum.auto()
    ENDOGENOUS = enum.auto()

    PARAMETER = enum.auto()
    ERROR = enum.auto()

    FUNCTION = enum.auto()
    KEYWORD = enum.auto()

    INVALID = enum.auto()


# Replacement functions for model solution
replacement_function_names = {
    'exp': 'np.exp',
    'log': 'np.log',
    'max': 'max',
    'min': 'min',
}

class Term(NamedTuple):
    """Container for information about a single term of an equation."""
    name: str
    type: Type
    index_: Optional[int]

    def __str__(self) -> str:
        """Standardised representation of the term."""
        expression = self.name

        # If neither a function nor a keyword, add the index
        if self.type not in (Type.FUNCTION, Type.KEYWORD):
            assert self.index_ is not None

            if self.index_ > 0:
                index = '[t+{}]'.format(self.index_)
            elif self.index_ == 0:
                index = '[t]'
            else:
                index = '[t{}]'.format(self.index_)

            expression += index

        return expression

    @property
    def code(self) -> str:
        """Representation of the term in code for model solution."""
        code = str(self)

        # If a function or a keyword, update the name; otherwise amend to be an
        # object attribute
        if self.type in (Type.FUNCTION, Type.KEYWORD):
            code = replacement_function_names.get(code, code)
        else:
            code = 'self._' + code

        return code


class Symbol(NamedTuple):
    """Container for information about a single symbol of an equation or model."""
    name: str
    type: Type
    lags: Optional[int]
    leads: Optional[int]
    equation: Optional[str]
    code: Optional[str]

    def combine(self, other: 'Symbol') -> 'Symbol':
        """Combine the attributes of two symbols."""

        def resolve_strings(old: Optional[str], new: Optional[str]) -> Optional[str]:
            resolved = old

            if old is not None and new is not None:
                if old != new:
                    raise ParserError(
                        "Endogenous variable '{}' defined twice:\n    {}\n    {}"
                        .format(self.name, old, new))

            elif old is None and new is not None:
                resolved = new

            return resolved

        assert self.name == other.name

        combined_type = self.type
        if self.type != other.type:
            if (self.type  not in (Type.VARIABLE, Type.EXOGENOUS, Type.ENDOGENOUS) or
                other.type not in (Type.VARIABLE, Type.EXOGENOUS, Type.ENDOGENOUS)):
                raise SymbolError('''\
Unable to combine the following pair of symbols:
 - {}
 - {}

Variables cannot appear in the input script as both endogenous/exogenous
variables and parameters/errors etc.'''.format(self, other))

            combined_type = max(self.type, other.type)

        if self.lags is None:
            assert other.lags is None
            lags = None
        else:
            lags = min(self.lags, other.lags, 0)

        if self.leads is None:
            assert other.leads is None
            leads = None
        else:
            leads = max(self.leads, other.leads, 0)

        equation = resolve_strings(self.equation, other.equation)
        code = resolve_strings(self.code, other.code)

        return Symbol(name=self.name,
                      type=combined_type,
                      lags=lags,
                      leads=leads,
                      equation=equation,
                      code=code)


# Parser functions to convert strings to objects ------------------------------

def split_equations_iter(model: str) -> Iterator[str]:
    """Return the equations of `model` as an iterator of strings."""

    def strip_comments(line: str) -> str:
        hash_position = line.find('#')
        if hash_position == -1:
            return line
        else:
            return line[:hash_position].rstrip()

    # Counter: Increment for every opening bracket and decrement for every
    # closing one. When zero, all pairs have been matched
    unmatched_parentheses = 0

    buffer = []

    for line in map(strip_comments, model.splitlines()):
        buffer.append(line)

        # Count unmatched parentheses
        for char in line:
            if char == '(':
                unmatched_parentheses += 1
            elif char == ')':
                unmatched_parentheses -= 1

            if unmatched_parentheses < 0:
                raise ParserError('Found closing bracket before opening bracket '
                                  'while attempting to read the following: {}'.format('\n'.join(buffer)))

        # If complete, combine and yield
        if unmatched_parentheses == 0:
            # Combine into a single string
            equation = '\n'.join(buffer)

            if equation.strip():  # Skip pure whitespace
                # Check that the string is a valid equation by testing against
                # the regular expression
                match = equation_re.search(equation)

                if not match:
                    # A failed match could occur because of unnecessary leading
                    # whitespace in the equation expression. Check for this and
                    # raise a slightly more helpful error if so
                    match = equation_re.search(equation.strip())

                    # Not compatible with Python 3.6: `if isinstance(match, re.Match):`
                    if match is not None:
                        raise IndentationError(
                            "Found unnecessary leading whitespace in equation: '{}'"
                            .format(equation))

                    # Otherwise, raise the general error
                    raise ParserError("Failed to parse equation: '{}'".format(equation))

                yield equation

            # Reset the buffer to collect another equation
            buffer = []

    # If `unmatched_parentheses` is non-zero at this point, there must have
    # been an error in the input script's syntax. Throw an error
    if unmatched_parentheses != 0:
        raise ParserError('Failed to identify any equations in the following, '
                          'owing to unmatched brackets: {}'.format('\n'.join(buffer)))

def split_equations(model: str) -> List[str]:
    """Return the equations of `model` as a list of strings."""
    return list(split_equations_iter(model))


def parse_terms(expression: str) -> List[Term]:
    """Return the terms of `expression` as a list of Term objects."""

    def process_term_match(match: Match[str]) -> Term:
        """Return the contents of `match` as a Term object."""
        groupdict = match.groupdict()

        # Filter to the named group that matched for this term
        type_key_list = [k for k, v in groupdict.items()
                         if k.startswith('_') and v is not None]
        assert len(type_key_list) == 1
        type_key = type_key_list[0]

        # Add implicit index (0) to non-function terms
        index: Optional[int] = None

        index_ = groupdict['INDEX']
        if type_key not in ('_FUNCTION', '_KEYWORD'):
            if index_ is None:
                index_ = '0'

            try:
                index = int(index_)
            except ValueError:
                raise ParserError("Unable to parse index '{}' of '{}' in '{}'".format(
                    index_, match.group(0), expression))

        return Term(name=groupdict[type_key],
                    type=Type[type_key[1:]],
                    index_=index)

    return [process_term_match(m)
            for m in term_re.finditer(expression)
            # Python keywords are unnamed groups: Filter these out
            if any(m.groups())]

def parse_equation_terms(equation: str) -> List[Term]:
    """Return the terms of `equation` as a list of Term objects."""

    def replace_type(term: Term, new_type: Type) -> Term:
        if term.type == Type.VARIABLE:
            term = term._replace(type=new_type)
        return term

    left, right = equation.split('=', maxsplit=1)

    try:
        lhs_terms = [replace_type(t, Type.ENDOGENOUS) for t in parse_terms(left)]
    except ParserError:
        # Catch any parser errors at a term level and raise a further exception
        # to print the expression that failed
        raise ParserError("Failed to parse left-hand side of: '{}'".format(equation))

    try:
        rhs_terms = [replace_type(t, Type.EXOGENOUS) for t in parse_terms(right)]
    except ParserError:
        # Catch any parser errors at a term level and raise a further exception
        # to print the expression that failed
        raise ParserError("Failed to parse right-hand side of: '{}'".format(equation))

    if (any(filter(lambda x: x.type == Type.KEYWORD, lhs_terms)) or
        any(filter(lambda x: x.type == Type.INVALID, rhs_terms))):
        raise ParserError(
            'Equation uses one or more reserved Python keywords as variable '
            'names - these keywords are invalid for this purpose: `{}`'
            .format(equation))

    return lhs_terms + rhs_terms

def parse_equation(equation: str) -> List[Symbol]:
    """Return the symbols of `equation` as a list of Symbol objects."""
    # Return an empty list if the string is empty / pure whitespace
    if len(equation.strip()) == 0:
        return []

    # Use `split_equations()` to check that the expression is valid and only
    # contains one equation
    equations = split_equations(equation)
    if len(equations) != 1:
        raise ParserError(
            '`parse_equation()` expects a string that defines a single equation '
            'but found {} instead'.format(len(equations)))

    terms = parse_equation_terms(equation)

    # Construct standardised and code representations of the equation
    template = equation
    for match in reversed(list(term_re.finditer(equation))):
        # Skip Python keywords, which yield unnamed groups
        if not any(match.groups()):
            continue

        start, end = match.span()
        template = '{}{}{}'.format(template[:start], '{}', template[end:])

    template = re.sub(r'\s+',   ' ', template)  # Remove repeated whitespace
    template = re.sub(r'\(\s+', '(', template)  # Remove space after opening brackets
    template = re.sub(r'\s+\)', ')', template)  # Remove space before closing brackets

    equation = template.format(*[str(t) for t in terms])
    code = template.format(*[t.code for t in terms])

    # `symbols` stores the final symbols and is successively updated in the
    # loop below
    symbols: Dict[str, Symbol] = {}

    # `functions` keeps track of functions seen, to avoid duplicating entries
    # in `symbols`
    functions: Dict[str, Symbol] = {}

    for term in terms:
        symbol = Symbol(name=term.name,
                        type=term.type,
                        lags=term.index_,
                        leads=term.index_,
                        equation=None,
                        code=None)

        name = symbol.name

        if symbol.type == Type.FUNCTION:
            # Function previously encountered: Test for equality against the
            # previous entry
            if name in functions:
                assert symbol == functions[name]
            # Otherwise, store
            else:
                symbols[name] = symbol
                functions[name] = symbol
            continue

        # Update endogenous variables with the equation and code information
        # from above
        if symbol.type == Type.ENDOGENOUS:
            symbol = symbol._replace(equation=equation, code=code)

        symbols[name] = symbols.get(name, symbol).combine(symbol)

    return list(symbols.values())

def parse_model(model: str, *, check_syntax: bool = True) -> List[Symbol]:
    """Return the symbols of `model` as a list of Symbol objects."""
    # Symbols: one list per model equation
    symbols_by_equation: List[List[Symbol]] = []

    # Store any statements that fail the (optional) syntax check as 2-tuples of
    #  - int: location of statement in `model`
    #  - str: the original statement
    #  - str: the (attempted) translation of the statement
    problem_statements: List[Tuple[int, str, str]] = []

    # Parse each statement (equation), optionally checking the syntax
    for i, statement in enumerate(split_equations_iter(model)):
        equation_symbols = parse_equation(statement)

        if check_syntax:
            equations = [s.equation for s in equation_symbols if s.equation is not None]

            for e in equations:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter('always')

                    # Check for exceptions when trying to run the current equation
                    try:
                        exec(e)
                    except NameError:  # Ignore name errors (undefined variables)
                        pass
                    except SyntaxError:
                        problem_statements.append((i, statement, e))
                        break

                    # Check for warnings and treat them as errors
                    if len(w) == 0:
                        pass
                    elif len(w) == 1:
                        if issubclass(w[0].category, SyntaxWarning):
                            problem_statements.append((i, statement, e))
                            break
                        else:
                            raise ParserError(
                                'Unexpected warning (error) when parsing: {}\n    {}'.format(statement, w[0]))
                    else:
                        raise ParserError(
                            'Unexpected number of warnings (errors) when parsing: {}'.format(statement))

        symbols_by_equation.append(equation_symbols)

    # Error if any problem statements found
    if problem_statements:
        # Construct report for each statement: number, original statement and
        # attempted equation
        lines = []
        for s in problem_statements:
            line = '    {}:  {}\n'.format(*s[:2])
            line += ' ' * line.index(':')
            line += '-> ' + s[-1]
            lines.append(line)

        raise ParserError(
            'Failed to parse the following {} statement(s):\n'.format(len(lines)) +
            '\n'.join(lines))

    # Store combined symbols to a dictionary, successively combining in the
    # loop below
    symbols: Dict[str, Symbol] = {}
    for symbol in itertools.chain(*symbols_by_equation):
        name = symbol.name
        symbols[name] = symbols.get(name, symbol).combine(symbol)

    return list(symbols.values())


# Labelled container for vector data (1D NumPy arrays) ------------------------

class VectorContainer:
    """Labelled container for vector data (1D NumPy arrays).

    Examples
    --------
    # Initialise an object with a span that attaches labels to the vector
    # elements; here: 11-element vectors labelled [2000, 2001, ..., 2009, 2010]
    >>> container = VectorContainer(range(2000, 2010 + 1))

    # Add some data, each with a fixed dtype (as in NumPy/pandas)
    # `VectorContainer`s automatically broadcast scalar values to vectors of
    # the previously specified length
    >>> container.add_variable('A', 2)                                        # Integer
    >>> container.add_variable('B', 3.0)                                      # Float
    >>> container.add_variable('C', list(range(10 + 1)), dtype=float)         # Force as floats
    >>> container.add_variable('D', [0, 1., 2, 3., 4, 5., 6, 7., 8, 9., 10])  # Cast to float

    # View the data, whether by attribute or key
    >>> container.A                                      # By attribute
    array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

    >>> container['B']                                   # By key
    array([3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.])

    >>> container.values  # Pack into a 2D array (casting to a common dtype)
    array([[ 2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.],
           [ 3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.],
           [ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.],
           [ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.]])

    # Vector replacement preserves both length and dtype, making it more
    # convenient than the NumPy equivalent: `container.B[:] = 4`
    >>> container.B = 4  # Standard object behaviour would assign `4` to `B`...
    >>> container.B      # ...but `VectorContainer`s broadcast and convert automatically
    array([4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4.])  # Still a vector of floats

    # NumPy-style indexing, whether by attribute or key (interchangeably)
    >>> container.C[2:4]  # By attribute
    array([2., 3.])

    >>> container.B[2:4] = 5  # As above, assignment preserves original dtype
    >>> container.B
    array([4., 4., 5., 5., 4., 4., 4., 4., 4., 4., 4.])

    >>> container['D'][3:-2]  # By key
    array([3., 4., 5., 6., 7., 8.])

    >>> container['C'][5:] = 9
    >>> container.C  # `container['C']` also works
    array([0., 1., 2., 3., 4., 9., 9., 9., 9., 9., 9.])

    # pandas-like indexing using the span labels (must be by key, as a 2-tuple)
    >>> container['D', 2005:2009]  # Second element (slice) refers to the labels of the object's span
    array([5., 6., 7., 8., 9.])

    >>> container['D', 2000:2008:2] = 12
    >>> container['D']  # As previously, `container.D` also works
    array([12.,  1., 12.,  3., 12.,  5., 12.,  7., 12.,  9., 10.])
    """

    def __init__(self, span: Sequence[Hashable]) -> None:
        """Initialise model variables.

        Parameter
        ---------
        span : iterable
            Sequence of periods that defines the timespan of the model
        """
        self.__dict__['span'] = span
        self.__dict__['index'] = []

    def add_variable(self, name: str, value: Union[Any, Sequence[Any]], *, dtype: Any = None) -> None:
        """Initialise a new variable in the container, forcing a `dtype` if needed.

        Parameters
        ----------
        name : str
            The name of the new variable, which must not already exist in the
            container (raise a `DuplicateNameError` if it already exists)
        value : single value or sequence of values
            Initial value(s) for the new variable. If a sequence, must yield a
            length equal to that of `span`
        dtype : variable type
            Data type to impose on the variable. The default is to use the
            type/dtype of `value`
        """
        if name in self.__dict__['index']:
            raise DuplicateNameError(
                "'{}' is already defined in the current object".format(name))

        # Cast to a 1D array
        if isinstance(value, Sequence) and not isinstance(value, str):
            value_as_array = np.array(value).flatten()
        else:
            value_as_array = np.full(len(self.__dict__['span']), value)

        # Impose `dtype` if needed
        if dtype is not None:
            value_as_array = value_as_array.astype(dtype)

        # Check dimensions
        if value_as_array.shape[0] != len(self.__dict__['span']):
            raise DimensionError("Invalid assignment for '{}': "
                                 "must be either a single value or "
                                 "a sequence of identical length to `span`"
                                 "(expected {} elements)".format(
                                     name, len(self.__dict__['span'])))

        self.__dict__['_' + name] = value_as_array
        self.__dict__['index'].append(name)

    def __getattr__(self, name: str) -> Any:
        if name in self.__dict__['index']:
            return self.__dict__['_' + name]
        else:
            return super().__getattribute__(name)

    def __setattr__(self, name: str, value: Union[Any, Sequence[Any]]) -> None:
        if name not in self.__dict__['index']:
            super.__setattr__(self, name, value)
            return

        elif isinstance(value, Sequence) and not isinstance(value, str):
            value_as_array = np.array(value,
                                      dtype=self.__dict__['_' + name].dtype)

            if value_as_array.shape[0] != len(self.__dict__['span']):
                raise DimensionError("Invalid assignment for '{}': "
                                     "must be either a single value or "
                                     "a sequence of identical length to `span`"
                                     "(expected {} elements)".format(
                                         name, len(self.__dict__['span'])))

            self.__dict__['_' + name] = value_as_array

        else:
            self.__dict__['_' + name][:] = value

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

    def replace_values(self, **new_values) -> None:
        """Convenience function to replace values in one or more series at once.

        Parameter
        ---------
        **new_values : key-value pairs of replacements

        Examples
        --------
        >>> container = VectorContainer(range(10))
        >>> container.add_variable('A', 3)
        >>> container.add_variable('B', range(0, 20, 2))
        >>> container.values
        array([[ 3,  3,  3,  3,  3,  3,  3,  3,  3,  3],
               [ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18]])

        >>> container.replace_values(A=4)
        array([[ 4,  4,  4,  4,  4,  4,  4,  4,  4,  4],
               [ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18]])

        >>> container.replace_values(A=range(10), B=5)
        >>> container.values
        array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
               [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]])

        >>> container.replace_values(**{'A': 6, 'B': range(-10, 0)})
        >>> container.values
        array([[  6,   6,   6,   6,   6,   6,   6,   6,   6,   6],
               [-10,  -9,  -8,  -7,  -6,  -5,  -4,  -3,  -2,  -1]])
        """
        for k, v in new_values.items():
            self.__setitem__(k, v)

    def __contains__(self, key: str) -> bool:
        return key in self.__dict__['index']

    def copy(self) -> 'VectorContainer':
        """Return a copy of the current object."""
        copied = self.__class__(span=copy.deepcopy(self.__dict__['span']))
        copied.__dict__.update(
            {k: copy.deepcopy(v)
             for k, v in self.__dict__.items()})
        return copied

    __copy__ = copy

    def __deepcopy__(self, *args, **kwargs) -> 'VectorContainer':
        return self.copy()

    def __dir__(self) -> List[str]:
        return sorted(
            dir(type(self)) + self.__dict__['index'] + ['span', 'index'])

    def _ipython_key_completions_(self) -> List[str]:
        return self.__dict__['index']

    @property
    def values(self) -> np.ndarray:
        """Container contents as a 2D (index x span) array."""
        return np.array([
            self.__getattribute__('_' + name) for name in self.__dict__['index']
        ])

    @values.setter
    def values(self, new_values: np.ndarray) -> None:
        """Replace all values (effectively, element-by-element), preserving the original data types in the container.

        Notes
        -----
        The replacement array must be 2D and have identical dimensions to
        `values`, and in the right order: (index x span).

        The method coerces each row to the data type of the corresponding item
        in the container.
        """
        if new_values.shape != self.values.shape:
            raise DimensionError(
                'Replacement array is of shape {} but expected shape is {}'
                .format(new_values.shape, self.values.shape))

        for name, series in zip(self.index, new_values):
            self.__setattr__(name, series.astype(self.__getattribute__('_' + name).dtype))

    def eval(self) -> None:
        raise NotImplementedError('`eval()` method (including API) not implemented yet')


# Model interface, wrapping the core `VectorContainer` ------------------------

class ModelInterface(VectorContainer):

    NAMES: List[str] = []

    def __init__(self, span: Sequence[Hashable], *, dtype: Any = float, default_value: Union[int, float] = 0.0, **initial_values: Dict[str, Any]) -> None:
        """Initialise model variables.

        Parameters
        ----------
        span : iterable
            Sequence of periods that defines the timespan of the model
        dtype : variable type
            Default data type to impose on model variables (in NumPy arrays)
        default_value : number
            Value with which to initialise model variables
        **initial_values : keyword arguments of variable names and values
            Values with which to initialise specific named model variables
        """
        # Set up data container
        super().__init__(span)

        # Store the `dtype` as the default for future values e.g. using
        # `add_variable()` after initialisation
        self.__dict__['dtype'] = dtype

        # Use the base class version of `add_variable()` because `self.names`
        # is set separately (whereas this class's version would attempt to
        # extend `self.names`)

        # Add solution tracking variables
        super().add_variable('status', '-')
        super().add_variable('iterations', -1)

        # Add model variables
        self.__dict__['names'] = copy.deepcopy(self.NAMES)
        for name in self.__dict__['names']:
            super().add_variable(name,
                                 initial_values.get(name, default_value),
                                 dtype=self.__dict__['dtype'])

    def add_variable(self, name: str, value: Union[Any, Sequence[Any]], *, dtype: Any = None) -> None:
        """Add a new variable to the model at runtime, forcing a `dtype` if needed.

        Parameters
        ----------
        name : str
            The name of the new variable, which must not already exist in the
            model (raise a `DuplicateNameError` if it already exists)
        value : single value or sequence of values
            Initial value(s) for the new variable. If a sequence, must yield a
            length equal to that of `span`
        dtype : variable type
            Data type to impose on the variable. The default is to use the
            `dtype` originally passed at initialisation. Otherwise, use this
            argument to set the type.

        Notes
        -----
        In implementation, as well as adding the data to the underlying
        container, this version of the method also extends the list of names in
        `self.names`, to keep the variable list up-to-date and for
        compatibility with, for example, `fsictools.model_to_dataframe()`.
        """
        # Optionally impose the `dtype`
        if dtype is None:
            dtype = self.__dict__['dtype']

        super().add_variable(name, value, dtype=dtype)
        self.__dict__['names'] += name

    def __contains__(self, key: str) -> bool:
        return key in self.__dict__['names']

    def __dir__(self) -> List[str]:
        return sorted(super().__dir__() + ['names'])

    @property
    def values(self) -> np.ndarray:
        """Model variable values as a 2D (names x span) array."""
        return np.array([
            self.__getattribute__('_' + name) for name in self.names
        ])

    @values.setter
    def values(self, new_values: np.ndarray) -> None:
        """Replace all values (effectively, element-by-element), preserving the original data types in the container.

        Notes
        -----
        The replacement array must be 2D and have identical dimensions to
        `values`, and in the right order: (names x span).

        The method coerces each row to the data type of the corresponding item
        in the container.
        """
        if new_values.shape != self.values.shape:
            raise DimensionError(
                'Replacement array is of shape {} but expected shape is {}'
                .format(new_values.shape, self.values.shape))

        for name, series in zip(self.names, new_values):
            self.__setattr__(name, series.astype(self.__getattribute__('_' + name).dtype))


# Mixin to define (but not fully implement) solver behaviour ------------------

class PeriodIter:
    """Iterator of (index, label) pairs returned by `BaseModel.iter_periods()`. Compatible with `len()`."""

    def __init__(self, *args):
        self._length = len(args[0])
        self._iter = list(zip(*args))

    def __iter__(self):
        yield from self._iter

    def __next__(self):
        return next(self._iter)

    def __len__(self):
        return self._length

class SolverMixin:
    """Requires `self.span` (a `Sequence` of `Hashable`s) as an attribute."""

    LAGS: int = 0
    LEADS: int = 0

    def iter_periods(self, *, start: Optional[Hashable] = None, end: Optional[Hashable] = None, **kwargs: Any) -> Iterator[Tuple[int, Hashable]]:
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
        **kwargs : not used
            Absorbs arguments passed from `solve()`. Available should the user
            want to over-ride this method in their own derived class.

        Returns
        -------
        A `PeriodIter` object, which is iterable and returns the period (index,
        label) pairs in sequence.
        """
        # Default start and end periods
        if start is None:
            start = self.span[self.LAGS]
        if end is None:
            end = self.span[-1 - self.LEADS]

        # Convert to an integer range
        indexes = range(self.span.index(start),
                        self.span.index(end) + 1)

        return PeriodIter(indexes, self.span[indexes.start:indexes.stop])

    def solve(self, *, start: Optional[Hashable] = None, end: Optional[Hashable] = None, min_iter: int = 0, max_iter: int = 100, tol: Union[int, float] = 1e-10, offset: int = 0, failures: str = 'raise', errors: str = 'raise', **kwargs: Dict[str, Any]) -> Tuple[List[Hashable], List[int], List[bool]]:
        """Solve the model. Use default periods if none provided.

        Parameters
        ----------
        start : element in the model's `span`
            First period to solve. If not given, defaults to the first solvable
            period, taking into account any lags in the model's equations
        end : element in the model's `span`
            Last period to solve. If not given, defaults to the last solvable
            period, taking into account any leads in the model's equations
        min_iter : int
            Minimum number of iterations to solution each period
        max_iter : int
            Maximum number of iterations to solution each period
        tol : int or float
            Tolerance for convergence
        offset : int
            If non-zero, copy an initial set of endogenous values from the
            relative period described by `offset`. For example, `offset=-1`
            initialises each period's solution with the values from the
            previous period.
        failures : str, one of {'raise', 'ignore'}
            Action should the solution fail to converge in a period (by
            reaching the maximum number of iterations, `max_iter`)
             - 'raise' (default): stop immediately and raise a
                                  `NonConvergenceError`
             - 'ignore': continue to the next period
        errors : str, one of {'raise', 'skip', 'ignore', 'replace'}
            User-specified treatment on encountering numerical solution errors
            e.g. NaNs and infinities
             - 'raise' (default): stop immediately and raise a `SolutionError`
                                  [set current period solution status to 'E']
             - 'skip': stop solving the current period and move to the next one
                       [set current period solution status to 'S']
             - 'ignore': continue solving, with no special treatment or action
                         [period solution statuses as usual i.e. '.' or 'F']
             - 'replace': each iteration, replace NaNs and infinities with
                          zeroes
                          [period solution statuses as usual i.e. '.' or 'F']
        kwargs :
            Further keyword arguments to pass on to other methods:
             - `iter_periods()`
             - `solve_t()`

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
        # Error if `min_iter` exceeds `max_iter`
        if min_iter > max_iter:
            raise ValueError(
                'Value of `min_iter` ({}) cannot exceed value of `max_iter` ({})'.format(
                    min_iter, max_iter))

        period_iter = self.iter_periods(start=start, end=end, **kwargs)

        indexes = [None] * len(period_iter)
        labels  = [None] * len(period_iter)
        solved  = [None] * len(period_iter)

        for i, (t, period) in enumerate(period_iter):
            indexes[i] = t
            labels[i] = period
            solved[i] = self.solve_t(t, min_iter=min_iter, max_iter=max_iter, tol=tol, offset=offset,
                                     failures=failures, errors=errors, **kwargs)

        return labels, indexes, solved

    def solve_period(self, period: Hashable, *, min_iter: int = 0, max_iter: int = 100, tol: Union[int, float] = 1e-10, offset: int = 0, failures: str = 'raise', errors: str = 'raise', **kwargs: Dict[str, Any]) -> bool:
        """Solve a single period.

        Parameters
        ----------
        period : element in the model's `span`
            Named period to solve
        min_iter : int
            Minimum number of iterations to solution each period
        max_iter : int
            Maximum number of iterations to solution each period
        tol : int or float
            Tolerance for convergence
        offset : int
            If non-zero, copy an initial set of endogenous values from the
            relative period described by `offset`. For example, `offset=-1`
            initialises each period's solution with the values from the
            previous period.
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
        t = self.span.index(period)
        return self.solve_t(t, min_iter=min_iter, max_iter=max_iter, tol=tol, offset=offset, failures=failures, errors=errors, **kwargs)

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
        raise NotImplementedError('Method must be over-ridden by a child class')


# Base class for individual models --------------------------------------------
class BaseModel(SolverMixin, ModelInterface):
    """Base class for economic models."""

    ENDOGENOUS: List[str] = []
    EXOGENOUS: List[str] = []

    PARAMETERS: List[str] = []
    ERRORS: List[str] = []

    NAMES: List[str] = ENDOGENOUS + EXOGENOUS + PARAMETERS + ERRORS
    CHECK: List[str] = ENDOGENOUS

    LAGS: int = 0
    LEADS: int = 0

    CODE: Optional[str] = None

    def __init__(self, span: Sequence[Hashable], *, engine: str = 'python', dtype: Any = float, default_value: Union[int, float] = 0.0, **initial_values: Dict[str, Any]) -> None:
        """Initialise model variables.

        Parameters
        ----------
        span : iterable
            Sequence of periods that defines the timespan of the model
        engine : str
            Signal of the (expected) underlying solution method/implementation
        dtype : variable type
            Data type to impose on model variables (in NumPy arrays)
        default_value : number
            Value with which to initialise model variables
        **initial_values : keyword arguments of variable names and values
            Values with which to initialise specific named model variables
        """
        self.__dict__['engine'] = engine

        super().__init__(span=span,
                         dtype=dtype,
                         default_value=default_value,
                         **initial_values)

    @classmethod
    def from_dataframe(cls: 'BaseModel', data: 'DataFrame', *args, **kwargs) -> 'BaseModel':
        """Initialise the model by taking the index and values from a `pandas` DataFrame(-like).

        Parameters
        ----------
        data : `pandas` DataFrame(-like)
            The DataFrame's index (once converted to a list) becomes the
            model's `span`.
            The DataFrame's contents set the model's values, as with
            `**initial_values` in the class `__init__()` method. Default values
            continue to be set for any variables not included in the DataFrame.
        *args, **kwargs : further arguments to the class `__init__()` method
        """
        return cls(list(data.index),
                   *args,
                   **{k: v.values for k, v in data.items()},
                   **kwargs)

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

        Notes
        -----
        As of version 0.3.0, FSIC provides (some) support (escape hatches) for
        numerical errors in solution.

        For example, there may be an equation that involves a division
        operation but the equation that determines the divisor follows
        later. If that divisor was initialised to zero, this leads to a
        divide-by-zero operation that NumPy evaluates to a NaN. This becomes
        problematic if the NaNs then propagate through the solution. Similar
        problems come about with infinities e.g. from log(0).

        The `solve_t()` method now catches such operations (after a full pass
        through / iteration over the system of equations).
        """
        def get_check_values() -> np.ndarray:
            """Return a 1D NumPy array of variable values for checking in the current period."""
            return np.array([
                self.__dict__['_' + name][t] for name in self.CHECK
            ])

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

        # Raise an exception if there are pre-existing NaNs or infinities, and
        # error checking is at its strictest ('raise')
        if errors == 'raise' and np.any(~np.isfinite(current_values)):
            raise SolutionError(
                'Pre-existing NaNs or infinities found '
                'in period with label: {} (index: {})'
                .format(self.span[t], t))

        # Run any code prior to solution
        self.solve_t_before(t, errors=errors, iteration=0, **kwargs)

        for iteration in range(1, max_iter + 1):
            previous_values = current_values.copy()

            with warnings.catch_warnings(record=True):
                warnings.simplefilter('always')

                try:
                    self._evaluate(t, errors=errors, iteration=iteration, **kwargs)
                except:
                    raise SolutionError(
                        'Error after {} iterations(s) '
                        'in period with label: {} (index: {})'
                        .format(iteration, self.span[t], t))

            current_values = get_check_values()

            # It's possible that the current iteration generated no NaNs or
            # infinities, but the previous one did: check and continue if
            # needed
            if np.any(~np.isfinite(previous_values)):
                continue

            if np.any(~np.isfinite(current_values)):

                if errors == 'raise':
                    self.status[t] = 'E'
                    self.iterations[t] = iteration

                    raise SolutionError(
                        'Numerical solution error after {} iterations(s) '
                        'in period with label: {} (index: {})'
                        .format(iteration, self.span[t], t))

                elif errors == 'skip':
                    status = 'S'
                    break

                elif errors == 'ignore':
                    if iteration == max_iter:
                        status = 'F'
                        break
                    continue

                elif errors == 'replace':
                    if iteration == max_iter:
                        status = 'F'
                        break
                    else:
                        current_values[~np.isfinite(current_values)] = 0.0
                        continue

                else:
                    raise ValueError('Invalid `errors` argument: {}'.format(errors))

            if iteration < min_iter:
                continue

            diff = current_values - previous_values
            if np.all(np.abs(diff) < tol):
                self.solve_t_after(t, errors=errors, iteration=iteration, **kwargs)
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

    def solve_t_before(self, t: int, *, errors: str = 'raise', iteration: Optional[int] = None, **kwargs: Dict[str, Any]) -> None:
        """Pre-solution method: This runs each period, before the iterative solution. Over-ride to implement custom behaviour."""
        pass

    def solve_t_after(self, t: int, *, errors: str = 'raise', iteration: Optional[int] = None, **kwargs: Dict[str, Any]) -> None:
        """Post-solution method: This runs each period, after the iterative solution. Over-ride to implement custom behaviour."""
        pass

    def _evaluate(self, t: int, *, errors: str = 'raise', iteration: Optional[int] = None, **kwargs: Dict[str, Any]) -> None:
        """Evaluate the system of equations for the period at integer position `t` in the model's `span`.

        Parameters
        ----------
        t : int
            Position in `span` of the period to solve
        errors : str
            User-specified treatment on encountering numerical solution
            errors. Note that it is up to the user's over-riding code to decide
            how to handle this.
        iteration : int
            The current iteration count. This is not guaranteed to take a
            non-`None` value if the user has over-ridden the default calling
            `solve_t()` method. Note that it is up to the user's over-riding
            code to decide how to handle this.
        kwargs :
            Further keyword arguments for solution
        """
        raise NotImplementedError('Method must be over-ridden by a child class')


# Model class generator -------------------------------------------------------

model_template_typed = '''\
class Model(BaseModel):
    ENDOGENOUS: List[str] = {endogenous}
    EXOGENOUS: List[str] = {exogenous}

    PARAMETERS: List[str] = {parameters}
    ERRORS: List[str] = {errors}

    NAMES: List[str] = ENDOGENOUS + EXOGENOUS + PARAMETERS + ERRORS
    CHECK: List[str] = ENDOGENOUS

    LAGS: int = {lags}
    LEADS: int = {leads}

    def _evaluate(self, t: int, *, errors: str = 'raise', iteration: Optional[int] = None, **kwargs: Dict[str, Any]) -> None:
{equations}\
'''

model_template_untyped = '''\
class Model(BaseModel):
    ENDOGENOUS = {endogenous}
    EXOGENOUS = {exogenous}

    PARAMETERS = {parameters}
    ERRORS = {errors}

    NAMES = ENDOGENOUS + EXOGENOUS + PARAMETERS + ERRORS
    CHECK = ENDOGENOUS

    LAGS = {lags}
    LEADS = {leads}

    def _evaluate(self, t, *, errors='raise', iteration=None, **kwargs):
{equations}\
'''

def build_model_definition(symbols: List[Symbol], *, converter: Optional[Callable[[Symbol], str]] = None, with_type_hints: bool = True) -> str:
    """Return a model class definition string from the contents of `symbols` (with type hints, by default).

    Parameters
    ----------
    symbols : list of fsic Symbol objects
        Constituent symbols of the model
    converter : (optional) callable, default `None`
        Mechanism for customising the Python code generator. If `None`,
        generates the equation as a comment followed by the equivalent Python
        code on the next line (see examples below). If a callable, takes a
        Symbol object as an input and must return a string of Python code for
        insertion into the model template.
    with_type_hints : bool
        If `True`, use the version of the model template with type hints. If
        `False`, exclude type hints (which may be convenient if looking to
        manipulate and/or execute the code directly).

    Notes
    -----
    The `converter` argument provides a way to alter the code generated at an
    equation level. The default converter just takes a Symbol object of type
    `ENDOGENOUS` and prints both the standardised equation as a comment, and
    the accompanying code. For example:

        # Y[t] = C[t] + I[t] + G[t] + X[t] - M[t]
        self._Y[t] = self._C[t] + self._I[t] + self._G[t] + self._X[t] - self._M[t]

    By passing a custom callable, you could generate something like the
    following instead, using the contents of the Symbol object:

        def custom_converter(symbol):
            lhs, rhs = map(str.strip, symbol.code.split('=', maxsplit=1))
            return '''\
# {}
_ = {}
if _ > 0:  # Ignore negative values
    {} = _'''.format(symbol.equation, rhs, lhs)

    Which would generate:

        # Y[t] = C[t] + I[t] + G[t] + X[t] - M[t]
        _ = self._C[t] + self._I[t] + self._G[t] + self._X[t] - self._M[t]
        if _ > 0:  # Ignore negative values
            self._Y[t] = _

    Examples
    --------
    >>> import fsic

    >>> symbols = fsic.parse_model('Y = C + I + G + X - M')
    >>> symbols
    [Symbol(name='Y', type=<Type.ENDOGENOUS: 3>, lags=0, leads=0, equation='Y[t] = C[t] + I[t] + G[t] + X[t] - M[t]', code='self._Y[t] = self._C[t] + self._I[t] + self._G[t] + self._X[t] - self._M[t]'),
     Symbol(name='C', type=<Type.EXOGENOUS: 2>, lags=0, leads=0, equation=None, code=None),
     Symbol(name='I', type=<Type.EXOGENOUS: 2>, lags=0, leads=0, equation=None, code=None),
     Symbol(name='G', type=<Type.EXOGENOUS: 2>, lags=0, leads=0, equation=None, code=None),
     Symbol(name='X', type=<Type.EXOGENOUS: 2>, lags=0, leads=0, equation=None, code=None),
     Symbol(name='M', type=<Type.EXOGENOUS: 2>, lags=0, leads=0, equation=None, code=None)]

    # With type hints (default)
    >>> print(fsic.build_model_definition(symbols))
    class Model(BaseModel):
        ENDOGENOUS: List[str] = ['Y']
        EXOGENOUS: List[str] = ['C', 'I', 'G', 'X', 'M']

        PARAMETERS: List[str] = []
        ERRORS: List[str] = []

        NAMES: List[str] = ENDOGENOUS + EXOGENOUS + PARAMETERS + ERRORS
        CHECK: List[str] = ENDOGENOUS

        LAGS: int = 0
        LEADS: int = 0

        def _evaluate(self, t: int, *, errors: str = 'raise', iteration: Optional[int] = None, **kwargs: Dict[str, Any]) -> None:
            # Y[t] = C[t] + I[t] + G[t] + X[t] - M[t]
            self._Y[t] = self._C[t] + self._I[t] + self._G[t] + self._X[t] - self._M[t]

    # Without type hints
    >>> print(fsic.build_model_definition(symbols, with_type_hints=False))
    class Model(BaseModel):
        ENDOGENOUS = ['Y']
        EXOGENOUS = ['C', 'I', 'G', 'X', 'M']

        PARAMETERS = []
        ERRORS = []

        NAMES = ENDOGENOUS + EXOGENOUS + PARAMETERS + ERRORS
        CHECK = ENDOGENOUS

        LAGS = 0
        LEADS = 0

        def _evaluate(self, t, *, errors='raise', iteration=None, **kwargs):
            # Y[t] = C[t] + I[t] + G[t] + X[t] - M[t]
            self._Y[t] = self._C[t] + self._I[t] + self._G[t] + self._X[t] - self._M[t]

    # Custom code generator
    >>> def custom_converter(symbol):
    ...     lhs, rhs = map(str.strip, symbol.code.split('=', maxsplit=1))
    ...     return '''\
    ... # {}
    ... _ = {}
    ... if _ > 0:  # Ignore negative values
    ...     {} = _'''.format(symbol.equation, rhs, lhs)

    >>> print(fsic.build_model_definition(symbols, converter=custom_converter))
    class Model(BaseModel):
        ENDOGENOUS: List[str] = ['Y']
        EXOGENOUS: List[str] = ['C', 'I', 'G', 'X', 'M']

        PARAMETERS: List[str] = []
        ERRORS: List[str] = []

        NAMES: List[str] = ENDOGENOUS + EXOGENOUS + PARAMETERS + ERRORS
        CHECK: List[str] = ENDOGENOUS

        LAGS: int = 0
        LEADS: int = 0

        def _evaluate(self, t: int, *, errors: str = 'raise', iteration: Optional[int] = None, **kwargs: Dict[str, Any]) -> None:
            # Y[t] = C[t] + I[t] + G[t] + X[t] - M[t]
            _ = self._C[t] + self._I[t] + self._G[t] + self._X[t] - self._M[t]
            if _ > 0:  # Ignore negative values
                self._Y[t] = _
    """

    def default_converter(symbol: Symbol) -> str:
        """Return Python code for the current equation as an `exec`utable string."""
        return '''\
# {}
{}'''.format(symbol.equation, symbol.code)

    if converter is None:
        converter = default_converter

    if with_type_hints:
        model_template = model_template_typed
    else:
        model_template = model_template_untyped

    # Separate variable names according to variable type
    endogenous = [s.name for s in symbols if s.type == Type.ENDOGENOUS]
    exogenous  = [s.name for s in symbols if s.type == Type.EXOGENOUS]
    parameters = [s.name for s in symbols if s.type == Type.PARAMETER]
    errors     = [s.name for s in symbols if s.type == Type.ERROR]

    # Set longest lag and lead
    non_indexed_symbols = [s for s in symbols
                           if s.type not in (Type.FUNCTION, Type.KEYWORD)]

    if len(non_indexed_symbols) > 0:
        lags =  abs(min(s.lags  for s in non_indexed_symbols))
        leads = abs(max(s.leads for s in non_indexed_symbols))
    else:
        lags = leads = 0

    # Generate code block of equations
    expressions = [converter(s) for s in symbols if s.type == Type.ENDOGENOUS]
    equations = '\n\n'.join(textwrap.indent(e, '        ') for e in expressions)

    # If there are no equations, insert `pass` instead
    if len(equations) == 0:
        equations = '        pass'

    # Fill in class template
    model_definition_string = model_template.format(
        endogenous=endogenous,
        exogenous=exogenous,

        parameters=parameters,
        errors=errors,

        lags=lags,
        leads=leads,

        equations=equations,
    )

    return model_definition_string

def build_model(symbols: List[Symbol], *, converter: Optional[Callable[[Symbol], str]] = None, with_type_hints: bool = True) -> 'BaseModel':
    """Return a model class definition from the contents of `symbols`. **Uses `exec()`.**

    Parameters
    ----------
    symbols : list of fsic Symbol objects
        Constituent symbols of the model
    converter : (optional) callable, default `None`
        Mechanism for customising the Python code generator. If `None`,
        generates the equation as a comment followed by the equivalent Python
        code on the next line. If a callable, takes a Symbol object as an input
        and must return a string of Python code for insertion into the model
        template.
        See the docstring for `build_model_definition()` for further details.
    with_type_hints : bool
        If `True`, use the version of the model template with type hints. If
        `False`, exclude type hints (which may be convenient if looking to
        manipulate and/or execute the code directly).

    Notes
    -----
    After constructing the class definition (as a string), this function calls
    `exec()` to define the class. In the event of a `SyntaxError`, the function
    loops through the `Symbol`s with defined equations and tries to construct
    and execute a class definition for each one, individually. The function
    then raises a `BuildError`, printing any `Symbol`s that failed to
    `exec`ute.
    """
    # Construct the class definition
    model_definition_string = build_model_definition(symbols,
                                                     converter=converter,
                                                     with_type_hints=with_type_hints)

    # Execute the class definition code
    try:
        exec(model_definition_string)
    except SyntaxError:
        failed_execs: List[str] = []

        symbols_with_equations = list(filter(lambda x: x.equation is not None, symbols))
        for s in symbols_with_equations:
            try:
                exec(build_model_definition([s]))
            except SyntaxError:
                failed_execs.append(str(s))

        if failed_execs:
            raise BuildError(
                'Failed to `exec`ute the following `Symbol` object(s):\n' +
                '\n'.join('    {}'.format(x) for x in failed_execs))

    # Otherwise, if here, assign the original code to an attribute and return
    # the class
    locals()['Model'].CODE = model_definition_string
    return locals()['Model']


# Base class to link models ---------------------------------------------------

class BaseLinker(SolverMixin, ModelInterface):
    """Base class to link economic models."""

    ENDOGENOUS: List[str] = []
    EXOGENOUS: List[str] = []

    PARAMETERS: List[str] = []
    ERRORS: List[str] = []

    NAMES: List[str] = ENDOGENOUS + EXOGENOUS + PARAMETERS + ERRORS
    CHECK: List[str] = ENDOGENOUS

    def __init__(self, submodels: Dict[Hashable, BaseModel], *, name: Hashable = '_', dtype: Any = float, default_value: Union[int, float] = 0.0, **initial_values: Dict[str, Any]) -> None:
        """Initialise linker with constituent submodels and core model variables.

        Parameters
        ----------
        submodels : dict
            Mapping of submodel identifiers (keys) to submodel instances
            (values)
        name :
            Identifier for the model embedded in the linker (in the same way
            that the submodels each have an identifier/key, as in `submodels`)
        dtype : variable type
            Data type to impose on core model variables (in NumPy arrays)
        default_value : number
            Value with which to initialise core model variables
        **initial_values : keyword arguments of variable names and values
            Values with which to initialise specific named core model variables

        Notes
        -----
        For now, the submodels must have identical `span` attributes. Method
        raises an `InitialisationError` if not.
        """
        self.__dict__['submodels'] = submodels
        self.__dict__['name'] = name

        # Get a list of submodel IDs and pop the first submodel as the one to
        # compare all the others against
        identifiers = iter(self.__dict__['submodels'])

        base_name = next(identifiers)
        base = self.__dict__['submodels'][base_name]

        lags = base.LAGS
        leads = base.LEADS

        # Check for a common span across submodels (error if not) and work out
        # the longest lags and leads
        for name in identifiers:
            # Check spans are identical
            comparator = self.__dict__['submodels'][name]
            if comparator.span != base.span:
                raise InitialisationError('''\
Spans of submodels differ:
 - '{}', {:,} period(s): {}
 - '{}', {:,} period(s): {}'''.format(
     base_name, len(base.span), base.span,
     name, len(comparator.span), comparator.span))

            # Update longest lags and leads
            lags = max(lags, comparator.LAGS)
            leads = max(leads, comparator.LEADS)

        # Store lags and leads
        self.__dict__['_LAGS'] = lags
        self.__dict__['_LEADS'] = leads

        # Initialise internal store for the core of the model (which is
        # separate from the individual submodels)
        super().__init__(span=base.span,
                         dtype=dtype,
                         default_value=default_value,
                         **initial_values)

    @property
    def LAGS(self) -> int:
        """Longest lag among submodels."""
        return self.__dict__['_LAGS']

    @property
    def LEADS(self) -> int:
        """Longest lead among submodels."""
        return self.__dict__['_LEADS']

    def copy(self) -> 'BaseLinker':
        """Return a copy of the current object."""
        copied = self.__class__(
            submodels={copy.deepcopy(k): copy.deepcopy(v)
                       for k, v in self.__dict__['submodels'].items()}
        )

        copied.__dict__.update(
            {k: copy.deepcopy(v)
             for k, v in self.__dict__.items()
             if k not in ['submodels']}
        )

        return copied

    __copy__ = copy

    def __deepcopy__(self, *args, **kwargs) -> 'BaseLinker':
        return self.copy()

    def solve(self, *, start: Optional[Hashable] = None, end: Optional[Hashable] = None, submodels: Optional[Sequence[Hashable]] = None, min_iter: int = 0, max_iter: int = 100, tol: Union[int, float] = 1e-10, offset: int = 0, failures: str = 'raise', errors: str = 'raise', **kwargs: Dict[str, Any]) -> Tuple[List[Hashable], List[int], List[bool]]:
        """Solve the linker and its constituent submodels. Use default periods if none provided.

        Parameters
        ----------
        start : element in the model's `span`
            First period to solve. If not given, defaults to the first solvable
            period, taking into account any lags in the model's equations
        end : element in the model's `span`
            Last period to solve. If not given, defaults to the last solvable
            period, taking into account any leads in the model's equations
        submodels : sequence of submodel identifiers (as in `self.submodels.keys()`), default `None`
            Submodels to solve for. If `None` (the default), solve them all.
        min_iter : int
            Minimum number of iterations to solution each period
        max_iter : int
            Maximum number of iterations to solution each period
        tol : int or float
            Tolerance for convergence
        offset : int
            If non-zero, copy an initial set of endogenous values from the
            relative period described by `offset`. For example, `offset=-1`
            initialises each period's solution with the values from the
            previous period.
        failures : str, one of {'raise', 'ignore'}
            Action should the solution fail to converge in a period (by
            reaching the maximum number of iterations, `max_iter`)
             - 'raise' (default): stop immediately and raise a
                                  `NonConvergenceError`
             - 'ignore': continue to the next period
        errors : str, one of {'raise', 'skip', 'ignore', 'replace'}
            User-specified treatment on encountering numerical solution errors
            e.g. NaNs and infinities
             - 'raise' (default): stop immediately and raise a `SolutionError`
                                  [set current period solution status to 'E']
             - 'skip': stop solving the current period and move to the next one
                       [set current period solution status to 'S']
             - 'ignore': continue solving, with no special treatment or action
                         [period solution statuses as usual i.e. '.' or 'F']
             - 'replace': each iteration, replace NaNs and infinities with
                          zeroes
                          [period solution statuses as usual i.e. '.' or 'F']
        kwargs :
            Further keyword arguments to pass on to other methods:
             - `iter_periods()`
             - `solve_t()`

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
        # Error if `min_iter` exceeds `max_iter`
        if min_iter > max_iter:
            raise ValueError(
                'Value of `min_iter` ({}) cannot exceed value of `max_iter` ({})'.format(
                    min_iter, max_iter))

        period_iter = self.iter_periods(start=start, end=end, **kwargs)

        indexes = [None] * len(period_iter)
        labels  = [None] * len(period_iter)
        solved  = [None] * len(period_iter)

        for i, (t, period) in enumerate(period_iter):
            indexes[i] = t
            labels[i] = period
            solved[i] = self.solve_t(t,
                                     submodels=submodels,
                                     min_iter=min_iter, max_iter=max_iter, tol=tol, offset=offset,
                                     failures=failures, errors=errors, **kwargs)

        return labels, indexes, solved

    def solve_t(self, t: int, *, submodels: Optional[Sequence[Hashable]] = None, min_iter: int = 0, max_iter: int = 100, tol: Union[int, float] = 1e-10, offset: int = 0, failures: str = 'raise', errors: str = 'raise', **kwargs: Dict[str, Any]) -> bool:
        """Solve for the period at integer position `t` in the linker's `span`.

        Parameters
        ----------
        t : int
            Position in `span` of the period to solve
        submodels : sequence of submodel identifiers (as in `self.submodels.keys()`), default `None`
            Submodels to solve for. If `None` (the default), solve them all.
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
        `True` if the linker solved for the current period; `False` otherwise.
        """
        if submodels is None:
            submodels = list(self.__dict__['submodels'].keys())


        def get_check_values() -> Dict[Hashable, np.ndarray]:
            """Return NumPy arrays of variable values for the current period, for checking."""
            check_values = {
                '_': np.array([self.__dict__['_' + name][t] for name in self.CHECK]),
            }

            for k, submodel in self.submodels.items():
                if k in submodels:
                    check_values[k] = np.array([submodel[name][t] for name in submodel.CHECK])

            return check_values


        status = '-'
        current_values = get_check_values()

        # Set iteration counters to zero for submodels to solve (also use this
        # as an opportunity to make sure specified submodels are valid)
        for name in submodels:
            try:
                submodel = self.__dict__['submodels'][name]
            except KeyError:
                raise KeyError("'{}' not found in list of submodels".format(name))

            submodel.iterations[t] = 0

        # Run any code prior to solution
        self.solve_t_before(t, submodels=submodels, errors=errors, iteration=0, **kwargs)

        for iteration in range(1, max_iter + 1):
            previous_values = copy.deepcopy(current_values)

            self.evaluate_t_before(t, submodels=submodels, errors=errors, iteration=iteration, **kwargs)
            self.evaluate_t(       t, submodels=submodels, errors=errors, iteration=iteration, **kwargs)
            self.evaluate_t_after( t, submodels=submodels, errors=errors, iteration=iteration, **kwargs)

            current_values = get_check_values()

            if iteration < min_iter:
                continue

            diff = {k: current_values[k] - previous_values[k]
                    for k in current_values.keys()}
            diff_squared = {k: v ** 2 for k, v in diff.items()}

            if all(np.all(v < tol) for v in diff_squared.values()):
                status = '.'
                self.solve_t_after(t, submodels=submodels, errors=errors, iteration=iteration, **kwargs)
                break

        else:
            status = 'F'

        self.status[t] = status
        self.iterations[t] = iteration

        for name in submodels:
            submodel = self.__dict__['submodels'][name]
            submodel.status[t] = status

        if status == 'F' and failures == 'raise':
            raise NonConvergenceError(
                'Solution failed to converge after {} iterations(s) '
                'in period with label: {} (index: {})'
                .format(iteration, self.span[t], t))

        return status == '.'

    def evaluate_t(self, t: int, *, submodels: Optional[Sequence[Hashable]] = None, errors: str = 'raise', iteration: Optional[int] = None, **kwargs: Dict[str, Any]) -> None:
        """Evaluate the system of equations for the period at integer position `t` in the linker's `span`.

        Parameters
        ----------
        t : int
            Position in `span` of the period to solve
        submodels : sequence of submodel identifiers (as in `self.submodels.keys()`), default `None`
            Submodels to evaluate. If `None` (the default), evaluate them all.
        errors : str
            User-specified treatment on encountering numerical solution
            errors, as passed to the individual submodels.
        iteration : int
            The current iteration count. This is not guaranteed to take a
            non-`None` value if the user has over-ridden the default calling
            `solve_t()` method. Note that it is up to the individual submodels'
            over-riding code to decide how to handle this.
        kwargs :
            Further keyword arguments for solution
        """
        if submodels is None:
            submodels = list(self.__dict__['submodels'].keys())

        for name in submodels:
            submodel = self.__dict__['submodels'][name]

            with warnings.catch_warnings(record=True):
                warnings.simplefilter('always')
                submodel._evaluate(t, errors=errors, iteration=iteration, **kwargs)

            submodel.iterations[t] += 1

    def solve_t_before(self, t: int, *, submodels: Optional[Sequence[Hashable]] = None, errors: str = 'raise', iteration: Optional[int] = None, **kwargs: Dict[str, Any]) -> None:
        """Pre-solution method: This runs each period, before the iterative solution. Over-ride to implement custom linker behaviour."""
        pass

    def solve_t_after(self, t: int, *, submodels: Optional[Sequence[Hashable]] = None, errors: str = 'raise', iteration: Optional[int] = None, **kwargs: Dict[str, Any]) -> None:
        """Post-solution method: This runs each period, after the iterative solution. Over-ride to implement custom linker behaviour."""
        pass

    def evaluate_t_before(self, t: int, *, submodels: Optional[Sequence[Hashable]] = None, errors: str = 'raise', iteration: Optional[int] = None, **kwargs: Dict[str, Any]) -> None:
        """Evaluate any linker equations before solving the individual submodels. Over-ride to implement custom linker behaviour."""
        pass

    def evaluate_t_after(self, t: int, *, submodels: Optional[Sequence[Hashable]] = None, errors: str = 'raise', iteration: Optional[int] = None, **kwargs: Dict[str, Any]) -> None:
        """Evaluate any linker equations after solving the individual submodels. Over-ride to implement custom linker behaviour."""
        pass
