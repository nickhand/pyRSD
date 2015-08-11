base_model_params = {'sigma8': 'mass variance at r = 8 Mpc/h',
                    'f': 'growth rate, f = dlnD/dlna', 
                    'alpha_perp': 'perpendicular Alcock-Paczynski effect parameter', 
                    'alpha_par': 'parallel Alcock-Paczynski effect parameter', 
                    'b1_sA': 'linear bias of sats w/o other sats in same halo',
                    'b1_sB': 'linear bias of sats w/ other sats in same halo',
                    'b1_cA': 'linear bias of cens w/o sats in same halo',
                    'b1_cB': 'linear bias of cens w/ sats in same halo',
                    'fcB': 'fraction of cens with sats in same halo',
                    'fsB': 'fraction of sats with other sats in same halo', 
                    'fs': 'fraction of total galaxies that are satellites',
                    'NcBs': 'amplitude of 1-halo power for cenB-sat in (Mpc/h)^3', 
                    'NsBsB': 'amplitude of 1-halo power for satB-satB in (Mpc/h)^3', 
                    'sigma_c': 'centrals FOG damping in Mpc/h',
                    'sigma_s': 'satellite FOG damping in Mpc/h',
                    'sigma_sA': 'satA FOG damping in Mpc/h', 
                    'sigma_sB': 'satB FOG damping in Mpc/h',
                    'small_scale_sigma': 'additional small scale velocity in km/s',
                    'N' : 'constant offset to model, in (Mpc/h)^3',
                    'fso' : 'so satelltie fraction around type A centrals',
                    'sigma_cA' : 'FOG damping in Mpc/h of SO satellites around type A centrals',
                    'sigma_so' : 'FOG damping in Mpc/h due to SO satellites'}

extra_model_params = {'b1_s': 'linear bias of satellites',
                    'b1_c': 'linear bias of centrals', 
                    'b1': 'the total linear bias', 
                    'fsigma8' : 'f(z)*sigma8(z) at z of measurement'}
                    
from .power_gal import GalaxyPowerParameters, GalaxyPowerTheory