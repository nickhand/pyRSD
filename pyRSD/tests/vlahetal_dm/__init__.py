from .. import pytest
from .. import cache_manager
import os

redshifts = [0., 0.509, 0.989, 2.070]

@pytest.fixture(scope='module', params=redshifts)
def redshift(request):
    return request.param

@pytest.fixture(scope='module')
def model(redshift):
        
    from pyRSD.rsd import DarkMatterSpectrum
    
    # inititalize the model
    config                   = {}
    config['include_2loop']  = True
    config['transfer_fit']   = 'CLASS'
    config['cosmo_filename'] = 'teppei_sims.ini'
    config['max_mu']         = 6
    config['use_P00_model']  = False
    config['use_P01_model']  = False
    config['use_P11_model']  = False
    config['use_Pdv_model']  = False
    config['interpolate']    = False
    config['z']              = redshift
    m = DarkMatterSpectrum(**config)
    
    # load the model
    with cache_manager(m, "vlahetal_dm.npy") as model:
        
        # set klims
        model.kmin = 0.01
        model.kmax = 0.5
        
        # update redshift-dependent quantities
        model.z        = redshift
        model.sigma8_z = model.cosmo.Sigma8_z(redshift)
        model.f        = model.cosmo.f_z(redshift)
    
    return model

#------------------------------------------------------------------------------
# TOOLS
#------------------------------------------------------------------------------
def get_sigma_bv2(z):
    """
    Return the value of `sigma_bv2` based on the input redshift
    """
    sigmas = [375., 356., 282., 144.]
    return sigmas[redshifts.index(z)]
    
def get_sigma_v2(z):
    """
    Return the value of `sigma_v2` based on the input redshift
    """
    sigmas = [209., 198., 159., 80.]
    return sigmas[redshifts.index(z)]

def get_sigma_bv4(z):
    """
    Return the value of `sigma_bv4` based on the input redshift
    """
    sigmas = [432., 382., 315., 144.]
    return sigmas[redshifts.index(z)]