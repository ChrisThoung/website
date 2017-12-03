# Power series approximation in Fortran/Python

Code and example to accompany post from 03/12/2017
on
[Wrapping Fortran code for Python with F2PY](http://www.christhoung.com/2017/12/03/f2py-power-series/).

Instructions:

1. Compile 'leontief.f95' to an extension module using `f2py` (see 'makefile'):
   'leontief'
2. Run the example script: 'power_series_example.py'

The example in the Python script comes from the *Tiny* model in:

Almon, C. (2017)
*The craft of economic modeling*,
Third, enlarged edition, *Inforum*  
[http://www.inforum.umd.edu/papers/TheCraft.html](http://www.inforum.umd.edu/papers/TheCraft.html)
