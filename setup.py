from setuptools import setup, Extension
import os, sys
import numpy

# setup the cc and cxx flags
os.environ['CC']  = "/usr/bin/gcc"
os.environ['CXX'] = "/usr/bin/gcc"

try: 
    from Cython.Distutils import build_ext
    import cython_gsl
except ImportError:
    print "You don't seem to have Cython installed. Please get a"
    print "copy from www.cython.org and install it"
    sys.exit(1)

# scan the 'pyPT' directory for extension files, converting
# them to extension names in dotted notation
def scandir(dir, files=[]):
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if os.path.isfile(path) and path.endswith(".pyx"):
            files.append(path.replace(os.path.sep, ".")[:-4])
        elif os.path.isdir(path):
            scandir(path, files)
    return files


parallel_exts = ['integralsIJ', 'integralsK']
# generate an Extension object from its dotted name
def makeExtension(extName):
    extPath = extName.replace(".", os.path.sep)+".pyx"
    
    cargs = ["-O3", '-w']
    largs = ['-g']
    if any(par_ext in extName for par_ext in parallel_exts):
        cargs.append('-fopenmp')
        largs.append('-fopenmp')
    
    sourceFiles = [extPath]
    if 'growth' in extPath:
        sourceFiles += ['pyRSD/cosmology/power_tools.c', 'pyRSD/cosmology/transfer.c']
    return Extension(
        extName,
        sourceFiles,
        extra_compile_args = cargs,
        extra_link_args = largs,
        libraries=cython_gsl.get_libraries(),
        library_dirs=[cython_gsl.get_library_dir(), '.'],
        include_dirs=[cython_gsl.get_cython_include_dir(), numpy.get_include(), "."]
        )

# get the list of extensions
extNames = scandir("pyRSD")

# and build up the set of Extension objects
extensions = [makeExtension(name) for name in extNames]

# finally, we can pass all this to distutils
setup(
  name="pyRSD",
  version='1.0',
  author='Nick Hand',
  author_email='nicholas.adam.hand@gmail.com',
  packages=['pyRSD', 'pyRSD.cosmology', 'pyRSD.rsd'],
  ext_modules=extensions,
  include_dirs = [cython_gsl.get_include(), '.'],
  cmdclass = {'build_ext': build_ext},
  description='python package for redshift space power spectra using perturbation theory',
  long_description=open('README.md').read()
)
