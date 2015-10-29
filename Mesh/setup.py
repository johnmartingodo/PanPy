from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext = Extension("Mesh",
                sources=["Mesh.pyx"],
                libraries=[],
                language="c", 
                include_dirs = [np.get_include()])

setup(name="Mesh",
    ext_modules=cythonize(ext)
    )