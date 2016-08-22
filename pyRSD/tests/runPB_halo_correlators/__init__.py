from .. import pytest
from .. import cache_manager

import os
from lsskit import data as lss_data

# load the biases globally
root   = os.path.join(os.environ['RSD_DIR'], 'runPB_mocks/Results/nbodykit')
data   = lss_data.RunPBHaloMomentum(root)
biases = lss_data.RunPBHaloPower(root).get_fof_halo_biases()

# scale factor and mass bins
scale_factors = biases['a'].values.tolist()
mass_bins = biases['mass'].values.tolist()

@pytest.fixture(scope='module', params=scale_factors)
def a(request):
    return request.param
    
@pytest.fixture(scope='module', params=mass_bins)
def mass(request):
    return request.param

@pytest.fixture(scope='module')
def model():
        
    from pyRSD.rsd import HaloSpectrum
    
    # inititalize the model
    config                       = {}
    config['include_2loop']      = False
    config['transfer_fit']       = 'CLASS'
    config['cosmo_filename']     = 'runPB.ini'
    config['max_mu']             = 4
    config['interpolate']        = False
    config['kmax']               = 0.95
    config['vel_disp_from_sims'] = False
    m = HaloSpectrum(**config)
    
    # load the model
    with cache_manager(m, "runPB_halo.npy") as model:
        
        # set klims
        model.kmin = 0.02
        model.kmax = 0.5
    
    return model

#------------------------------------------------------------------------------
# TOOLS
#------------------------------------------------------------------------------
def update_model(model, a_str, mass):
    """
    Update the redshift-dependent quantities
    """
    z = 1./float(a_str) - 1.
    
    # update redshift-dependent quantities
    model.z        = z
    model.sigma8_z = model.cosmo.Sigma8_z(z)
    model.f        = model.cosmo.f_z(z)
    model.sigma_v  = model.sigma_lin
    
    # set the bias
    b1 = biases.sel(a=a_str, mass=mass).values.tolist()
    model.b1 = b1
    
def get_valid_data(model, data, a_str, mass, subtract_shot_noise=False):
    """
    Return the valid data
    """
    d = data.sel(a=a_str, mass=mass).get()
    valid = (d['k'] >= model.kmin)&(d['k'] <= model.kmax)
    x, y, yerr = d['k'][valid], d['power'][valid], d['error'][valid]
    
    if subtract_shot_noise:
        y -= lss_data.tools.get_Pshot(d)
    
    return x, y, yerr





