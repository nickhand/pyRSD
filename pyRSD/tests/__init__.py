import matplotlib as mpl
import pytest

from pyRSD import data_dir
import os
# update the rc params
mpl.rc_file(os.path.join(data_dir, 'tests', 'matplotlibrc'))
mpl.use("Agg")

from .utils.cache import cache_manager
