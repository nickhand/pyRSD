API
---

.. currentmodule:: pyRSD.rsd.hzpt

The power spectrum classes are:

.. autosummary::

  HaloZeldovichP00
  HaloZeldovichP01
  HaloZeldovichP11
  HaloZeldovichPhm

Each of these objects provides three main functions to compute the various
HZPT terms:

.. currentmodule:: pyRSD.rsd.hzpt.HaloZeldovichP00

.. autosummary::

  __call__
  zeldovich
  broadband

The correlation function classes are:

.. currentmodule:: pyRSD.rsd.hzpt

.. autosummary::

  HaloZeldovichCF00
  HaloZeldovichCFhm

Similary, the three main functions to compute the various
HZPT terms are:

.. currentmodule:: pyRSD.rsd.hzpt.HaloZeldovichCF00

.. autosummary::

  __call__
  zeldovich
  broadband

HZPT Classes
~~~~~~~~~~~~

.. currentmodule:: pyRSD.rsd.hzpt

.. autoclass:: HaloZeldovichP00
  :members: __init__, __call__, zeldovich, broadband

.. autoclass:: HaloZeldovichP01
  :members: __init__, __call__, zeldovich, broadband

.. autoclass:: HaloZeldovichP11
  :members: __init__, __call__, zeldovich, broadband

.. autoclass:: HaloZeldovichPhm
  :members: __init__, __call__, zeldovich, broadband

.. autoclass:: HaloZeldovichCF00
  :members: __init__, __call__, zeldovich, broadband

.. autoclass:: HaloZeldovichCFhm
  :members: __init__, __call__, zeldovich, broadband
