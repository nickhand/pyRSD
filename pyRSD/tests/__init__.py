import matplotlib as mpl
mpl.use("Agg")
mpl.rc('font', family='Times New Roman')

import pytest
from pyRSD import data_dir
import os
from .utils.cache import cache_manager
