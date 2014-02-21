from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import cython_gsl
import numpy 

ext_modules = [
    Extension("integralsIJ", ["integralsIJ.pyx"],
              libraries=cython_gsl.get_libraries(),
    		  library_dirs=[cython_gsl.get_library_dir()],
              include_dirs=[cython_gsl.get_cython_include_dir(), numpy.get_include()],
              extra_compile_args=['-fopenmp'],
              extra_link_args=['-fopenmp'],),
    Extension("power_dm", ["power_dm.pyx"],
              include_dirs=[numpy.get_include()]),
    Extension("kernels", ["kernels.pyx"]),
    Extension("cosmology", ["cosmology.pyx"],
              include_dirs=[numpy.get_include()]),
    ]


setup(
  name = 'integrals IJ',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules,
  include_dirs = [cython_gsl.get_include()],
)