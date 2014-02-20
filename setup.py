from setuptools import setup, find_packages
import os

setup(
    name='pyPT',
    version='0.1',
    author='Nick Hand',
    author_email='nicholas.adam.hand@gmail.com',
    packages=find_packages(),
    description='python package for redshift space power spectra using perturbation theory',
    long_description=open('README.md').read()
)