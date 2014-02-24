from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import cython_gsl
import numpy 

ext_modules = [ Extension("cosmo_tools", ["cosmo_tools.pyx"],
                    include_dirs=[numpy.get_include()]),
                Extension("gsl_tools", ["gsl_tools.pyx"],
                    libraries=cython_gsl.get_libraries(),
                    library_dirs=[cython_gsl.get_library_dir()],
                    include_dirs=[cython_gsl.get_cython_include_dir(), numpy.get_include()]),
                Extension("linear_growth", ["linear_growth.pyx"],
                    libraries=cython_gsl.get_libraries(),
                    library_dirs=[cython_gsl.get_library_dir()],
                    include_dirs=[cython_gsl.get_cython_include_dir(), numpy.get_include()]) ]


setup(
  name = 'cosmology package',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules,
  include_dirs = [cython_gsl.get_include()],
)