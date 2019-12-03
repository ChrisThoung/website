# An initial foray into computer algebra with SymPy

Code to accompany the post from 03/12/2019 on [An initial foray into computer
algebra with SymPy](http://www.christhoung.com/2019/12/03/sympy-sim/).

The example is a simplified version of Model *SIM* from:

Godley, W., Lavoie, M. (2007)  
*Monetary economics: An integrated approach to credit, money, income, production and wealth*,  
Palgrave Macmillan

## Contents

* README.md : this file
* fsic.py : version 0.2.0 of the re-implemented FSIC module (no changes from
  the [previous post](http://www.christhoung.com/2019/07/27/fsic-update/) about
  FSIC)
* fsictools.py : new module of supporting tools for FSIC-based models
* test_fsic.py : test suite for fsic.py

## How to

* Run the test suite: Run `python test_fsics.py` or use a `unittest`-compatible
  test framework (e.g. `unittest`, `pytest`, `nose`/`nose2`; other test
  frameworks are available)
