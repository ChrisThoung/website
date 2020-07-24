# Some thoughts on macroeconomic input-output modelling in Python

**This is work-in-progress code.** I'm still finalising this initial version
and will also change the name/location of this sub-folder once I have an
accompanying write-up.

Experimental code for a possible implementation of a macroeconomic input-output
model in Python, using similar principles to
[FSIC](https://github.com/ChrisThoung/fsic), my Python package for (aggregate)
economic modelling. As experimental code, *this is nowhere close to a complete
and operational macroeconomic model*, but is instead a testbed for a possible
implementation.

The example uses 12-industry Scottish Government input-output tables
('Aggregate Tables 1998 to 2016': input-output-1998-2016-aggtables.xlsx)
available from:  
[https://www.gov.scot/publications/supply-use-input-output-tables-multipliers-scotland/](https://www.gov.scot/publications/supply-use-input-output-tables-multipliers-scotland/)

You'll need to download the data and then run a Python script (see instructions
below) to process the data to a useable format.

## Contents

* README.md : this file
* example_model.py : example model and model run
* iomodel.py : experimental library of base classes for defining a
  macroeconomic input-output model (nominally, version 0.1.0) - various
  elements have been copied or adapted from
  [FSIC](https://github.com/ChrisThoung/fsic)
* process_raw_data.py : Python script to process the raw data - run this before
  running example_model.py, to create the input data (as a new file: data.txt)
* test_iomodel.py : test suite for iomodel.py (this script is compatible with
  `unittest` and similar test frameworks e.g. `unittest`, `pytest`,
  `nose`/`nose2`)

## Instructions

1. Download the 12-industry input-output tables ('Aggregate Tables 1998 to
   2016': input-output-1998-2016-aggtables.xlsx) from the Scottish Government
   website, saving to this folder:  
   [https://www.gov.scot/publications/supply-use-input-output-tables-multipliers-scotland/](https://www.gov.scot/publications/supply-use-input-output-tables-multipliers-scotland/)
2. Run process_raw_data.py (e.g. `python process_raw_data.py`) to generate a
   data file, data.txt, from the raw Scottish Government data
3. Run example_model.py (e.g. `python example_model.py`) to test the example
   model, which just implements the calculation for gross output using a vector
   of final demands and the (Type I) Leontief Inverse matrix: if it runs
   through without error, the model/'model' has successfully reproduced the
   gross output figures in the original data
