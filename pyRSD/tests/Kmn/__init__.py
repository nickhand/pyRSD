from .. import pytest
from .. import cache_manager

@pytest.fixture(scope='module')
def model():
        
    from pyRSD.rsd import HaloSpectrum
    
    # inititalize the model
    config                       = {}
    config['include_2loop']      = False
    config['transfer_fit']       = 'CLASS'
    config['cosmo_filename']     = 'teppei_sims.ini'
    config['max_mu']             = 4
    config['use_P00_model']      = False
    config['use_P01_model']      = False
    config['use_P11_model']      = False
    config['use_Pdv_model']      = False
    config['interpolate']        = False
    config['vel_disp_from_sims'] = True
    m = HaloSpectrum(**config)
    
    # load the model
    with cache_manager(m, "vlahetal_halo.npy") as model:
        
        # set klims
        model.kmin = 0.01
        model.kmax = 0.5
        
        # update redshift-dependent quantities
        model.z        = 0.
        model.sigma8_z = model.cosmo.Sigma8_z(0.)
        model.f        = model.cosmo.f_z(0.)
    
    return model