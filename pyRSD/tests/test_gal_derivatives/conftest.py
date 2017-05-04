from .. import pytest
from .. import cache_manager
from pyRSD.rsdfit import FittingDriver
from pyRSD import data_dir
import os

@pytest.fixture(scope='session', autouse=True)
def driver(request):

    from pyRSD.rsd import GalaxySpectrum

    # add the PYRSD_DATA env var
    os.environ['PYRSD_DATA'] = data_dir

    # inititalize the model
    config                   = {}
    config['z']              = 0.55
    config['cosmo_filename'] = 'runPB.ini'
    config['kmin']           = 1e-3
    config['kmax']           = 0.6
    config['interpolate']    = True
    m = GalaxySpectrum(**config)

    # load the model
    with cache_manager(m, "runPB_galaxy.npy") as model:
        pass

    # initialize the driver
    path = os.path.join(data_dir, 'examples', 'params.dat')
    driver = FittingDriver(path, init_model=False)
    driver.model = model

    # set fiducial
    driver.set_fiducial()

    return driver
