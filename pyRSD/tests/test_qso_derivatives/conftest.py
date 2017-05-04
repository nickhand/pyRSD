from .. import pytest
from pyRSD.rsdfit import FittingDriver
from pyRSD import data_dir
import os

@pytest.fixture(scope='session', autouse=True)
def driver(request):

    from pyRSD.rsd import QuasarSpectrum

    # add the PYRSD_DATA env var
    os.environ['PYRSD_DATA'] = data_dir

    # inititalize the model
    config                   = {}
    config['z']              = 0.55
    config['cosmo_filename'] = 'runPB.ini'
    config['kmin']           = 1e-3
    config['kmax']           = 0.6
    m = QuasarSpectrum(**config)

    # initialize the driver
    path = os.path.join(data_dir, 'examples', 'params_qso.dat')
    driver = FittingDriver(path, init_model=False)
    driver.model = m

    # set fiducial
    driver.set_fiducial()

    return driver
