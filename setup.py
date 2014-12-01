#! /usr/bin/env python
from distutils.core import setup
import os

descr = """pyRSD

Algorithms to compute the redshift space matter power spectra using 
perturbation theory and the redshift space distortion (RSD) model based
on a distribution function velocity moments approach
"""

DISTNAME            = 'pyRSD'
DESCRIPTION         = 'Redshift space power spectra in python'
LONG_DESCRIPTION    = descr
MAINTAINER          = 'Nick Hand'
MAINTAINER_EMAIL    = 'nicholas.adam.hand@gmail.com'
VERSION             = '0.10dev'

    
setup(  name=DISTNAME,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        version=VERSION,
        packages=['pyRSD', 'pyRSD.data', 'pyRSD.rsd']
    )
