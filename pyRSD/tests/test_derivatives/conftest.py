from .. import pytest
from .. import cache_manager
from pyRSD.rsdfit import FittingDriver 

@pytest.fixture(scope='session', autouse=True)
def driver(request):
        
    from pyRSD.rsd import GalaxySpectrum
    
    # inititalize the model
    config                   = {}
    config['z']              = 0.55
    config['cosmo_filename'] = 'runPB.ini'
    config['kmin']           = 1e-3
    config['kmax']           = 0.6
    m                        = GalaxySpectrum(**config)
    
    # load the model
    with cache_manager(m, "runPB_galaxy.npy") as model:
        pass
    
    # initialize the driver
    driver = FittingDriver("params/params.dat", init_model=False)
    driver.model = model
    
    # set fiducial
    driver.set_fiducial()
    
    return driver