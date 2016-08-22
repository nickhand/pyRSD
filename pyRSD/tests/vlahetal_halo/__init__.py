from .. import pytest
from .. import cache_manager
import numpy

sim_tags  = ['teppei_lowz', 'teppei_midz', 'teppei_highz']
redshifts = [0., 0.509, 0.989]
vlah_bins = [(0, [0, 1, 2, 3]), (1, [0, 1, 2, 3]), (2, [0, 1, 2])]

@pytest.fixture(scope='module', params=vlah_bins)
def binning(request):
    return request.param

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
    config['vel_disp_from_sims'] = False
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

#------------------------------------------------------------------------------
# TOOLS
#------------------------------------------------------------------------------    
def get_sigma_v(z_str, mass):
    """
    Return the sigma_v for this bin
    """
    d = {}
    d['0.000'] = [306., 302., 296., 288.]
    d['0.509'] = [357., 352., 346., 339.]
    d['0.989'] = [340., 337., 330.]
    
    return d[z_str][mass]
    
def get_linear_bias(z_str, mass):
    """
    Return the linear bias for this bin
    """
    d = {}
    d['0.000'] = [1.18, 1.47, 2.04, 3.05]
    d['0.509'] = [1.64, 2.18, 3.13, 4.82]
    d['0.989'] = [2.32, 3.17, 4.64]
    
    return d[z_str][mass]
    
def get_stochasticity(z_str, mass):
    """
    Return the stochasticity function for this bin
    """
    d = {}
    d['0.000'] = [(83.2376, -246.78), (-645.891, -365.149), (-3890.46, -211.452), (-19202.9, 111.395)]
    d['0.509'] = [(-168.216, -137.268), (-1335.54, -104.929), (-6813.82, -241.446), (-35181.3, -5473.79)]
    d['0.989'] = [(-427.779, -41.3676), (-2661.11, -229.627), (-16915.5, -3215.98)]
    
    params = dict(zip(['A0', 'A1'], d[z_str][mass]))
    return lambda k: params['A0'] + params['A1'] * numpy.log(k)
    
def get_nonlinear_biases(z_str, mass):
    """
    Get the nonlinear biases for this bin
    """
    d = {}
    d['0.000'] = [(-0.39, -0.45), (-0.08, -0.35), (0.91, 0.14), (3.88, 2.00)]
    d['0.509'] = [(0.18, -0.20), (1.29, 0.48), (4.48, 2.60), (12.70, 9.50)]
    d['0.989'] = [(1.75, 0.80), (4.77, 3.15), (12.80, 10.80)]

    return dict(zip(['b2_00', 'b2_01'], d[z_str][mass]))

def get_params(z, mass):
    """
    Return the model parameters as measured by 
    """
    z_str = "%.3f" %z
    
    toret = get_nonlinear_biases(z_str, mass)
    toret['stochasticity'] = get_stochasticity(z_str, mass)
    toret['sigma_v'] = get_sigma_v(z_str, mass)
    toret['b1'] = get_linear_bias(z_str, mass)
    
    return toret
    
def update_model(model, z, b1):
    """
    Update the redshift-dependent quantities
    """
    # update redshift-dependent quantities
    model.z        = z
    model.sigma8_z = model.cosmo.Sigma8_z(z)
    model.f        = model.cosmo.f_z(z)
    
    # set the bias
    model.b1 = b1