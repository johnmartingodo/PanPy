from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os
import sys

if sys.platform == 'darwin':
	os.environ["CC"] = "gcc-5" 
	os.environ["CXX"] = "gcc-5"

ext = Extension("Computation",
                sources=["Computation.pyx"],
                libraries=[],
                language="c", 
                include_dirs = [np.get_include()],
                extra_compile_args=['-fopenmp'],
    		extra_link_args=['-fopenmp'])

setup(name="Computation",
      ext_modules=cythonize(ext)
    )
