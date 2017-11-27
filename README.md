pyRSD
=======

[![DOI](https://zenodo.org/badge/19336205.svg)](https://zenodo.org/badge/latestdoi/19336205)

pyRSD is a Python package for computing the theoretical predictions of the
redshift-space power spectrum of galaxies. The package also includes
functionality for fitting data measurements and finding the optimal model
parameters, using both MCMC and nonlinear optimization techniques.

The software is compatible with Python versions 2.7, 3.5, and 3.6.

Testing is performed via the continuous integration service for Python version 2.7, 3.5, and 3.6. The 
build status of those tests is below.

[![Build Status](https://travis-ci.org/nickhand/pyRSD.svg?branch=master)](https://travis-ci.org/nickhand/pyRSD)

Installation
============

The package is installable via the ``conda`` utility as

```bash
conda install -c nickhand -c astropy pyrsd
```

Reference
==========
The theoretical models used in this paper are described in more detail
in [Hand et al. 2017](https://arxiv.org/abs/1706.02362). Please cite
this work if you use this package in your research.

Documentation
=============

For installation instructions, examples, and full API documentation, please see [Read the Docs](http://pyrsd.readthedocs.io/en/latest/).
