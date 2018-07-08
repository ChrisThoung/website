# Implementing a Stock-Flow Consistent macroeconomic model in Python

Code to accompany post from 08/07/2018 on [Implementing a Stock-Flow Consistent
macroeconomic model in
Python](http://www.christhoung.com/2018/07/08/fsic-gl2007-pc/).

Model *PC* is a model of government money with portfolio choice from:

Godley, W., Lavoie, M. (2007)  
*Monetary economics: An integrated approach to credit, money, income, production and wealth*,  
Palgrave Macmillan

Some values for simulation come from Gennaro Zezza's [EViews
programs](http://gennaro.zezza.it/software/eviews/glch04.php).

## Contents

* README.md : this file
* fsic_pc.ipynb : Python notebook version of the blog post, which presents,
  recreates and solves Model *PC*
* fsics.py : barebones (and rough) re-implementation of the core FSIC interface
* test_fsics.py : test suite for fsics.py

## How to

* Run the model: Open and run the Jupyter notebook, 'fsic_pc.ipynb'
* Run the test suite: Run `python test_fsics.py` or use a `unittest`-compatible
  test framework (e.g. `unittest`, `pytest`, `nose`/`nose2`; other test
  frameworks are available)
