"""
 write_spectra.py
 pyPT: write out power spectra measured from simulations
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 03/18/2014
"""
from pyRSD import rsd
from pyRSD import cosmology

import scipy.interpolate as interp
import numpy as np
import matplotlib.pyplot as plt
import argparse

import sys
sys.path.append('/Users/Nick/research/codes/utils/')
from utils import mpfit

#-------------------------------------------------------------------------------
def transition(k, k_transition, b):
    return 0.5 + 0.5*np.tanh((k-k_transition)/b)

#-------------------------------------------------------------------------------
def model(k, p, PTmodel):
    return np.exp(p[0]*k*k)*PTmodel
#end model

#-------------------------------------------------------------------------------
def fitting_function(p, fjac=None, k=None, y=None, err=None, PTmodel=None):
    m = model(k, p, PTmodel)
    status = 0
    return ([status, (y-m)/err])
#end fitting_function

#-------------------------------------------------------------------------------
def fit_P00(params, k_transition=0.1, plot=False, save=True):
    
    zs = ['0.000', '0.509','0.989']
    zlabels = ['007', '005', '004']
    for z, zlabel in zip(zs, zlabels):
        
        # read in the data
        data = np.loadtxt("./pkmu_chi_00_m_m_z%s_1-3_02-13binaaQ" %zlabel)
        x = data[:,0]
        y = data[:,-2]
        err = data[:,-1]
        
        # find where k < 0.1
        inds = np.where(x < k_transition)

        # get the PT model
        Pspec = rsd.power_dm.DMSpectrum(k=x[inds], z=float(z), transfer_fit='CAMB', include_2loop=False, num_threads=4, cosmo=params)
        PTmodel = Pspec.P00.total.mu0

        # measure the exponential scale
        parinfo = [{'value':1.0, 'fixed':0, 'limited':[0,0], 'limits':[0.,0.]}]
        fa = {'k': x[inds], 'y': y[inds], 'err': err[inds], 'PTmodel': PTmodel}
        m = mpfit.mpfit(fitting_function, parinfo=parinfo, functkw=fa)
        
        # get final k
        k_final = np.logspace(-4, np.log10(np.amax(x)), 500)
        
        # get full model for PT
        Pspec = rsd.power_dm.DMSpectrum(k=k_final, z=float(z), transfer_fit='CAMB', include_2loop=False, num_threads=4, cosmo=params)
        PTmodel = Pspec.P00.total.mu0
        PTmodel   = Pspec.P00.total.mu0
        pt_spline = interp.InterpolatedUnivariateSpline(Pspec.k, PTmodel)
        
        # and the full sim model
        sim_spline = interp.InterpolatedUnivariateSpline(x, y, w=1/err)
        
        # now transition nicely
        switch = transition(k_final, k_transition, 0.01)
        P00 = (1-switch)*model(k_final, m.params, pt_spline(k_final)) + switch*sim_spline(k_final)
        
        if plot:
            norm = Pspec.D**2*Pspec.power_lin.power
                        
            plt.plot(k_final, P00/norm)
            plt.plot(k_final, PTmodel/norm)
            
            norm_spline = interp.InterpolatedUnivariateSpline(k_final, norm)
            plt.errorbar(x, y/norm_spline(x), err/norm_spline(x), c='k', marker='.', ls='')
            
            plt.xlim(0., 0.4)
            plt.ylim(0.9, 2.0)
            plt.xlabel(r"$\mathrm{k \ (h/Mpc)}$", fontsize=16)
            plt.ylabel(r"$P_{00} / (D^2 P_\mathrm{lin}(z=0))$", fontsize=16)
            plt.title(r"$\mathrm{P_{00} \ at \ z \ = \ %s}$" %z)
            plt.show()
        
        if save:
            np.savetxt("./pkmu_P00_mu0_z_%s.dat" %z, zip(k_final, P00))
#end fit_P00

#-------------------------------------------------------------------------------
def fit_P01(params, k_transition=0.1, plot=False, save=True):
    
    zs = ['0.000', '0.509','0.989']
    zlabels = ['007', '005', '004']
    for z, zlabel in zip(zs, zlabels):
        
        # read in the data
        data = np.loadtxt("./pkmu_chi_01_m_m_z%s_1-3_02-13binaaQ" %zlabel)
        x = data[:,0]
        y = data[:,-2]*2*x
        err = data[:,-1]*2*x
        
        # find where k < 0.1
        inds = np.where(x < k_transition)

        # get the PT model
        Pspec = rsd.power_dm.DMSpectrum(k=x[inds], z=float(z), transfer_fit="CAMB", include_2loop=False, num_threads=4, cosmo=params)
        PTmodel = Pspec.P01.total.mu2

        # measure the exponential scale
        parinfo = [{'value':1.0, 'fixed':0, 'limited':[0,0], 'limits':[0.,0.]}]
        fa = {'k': x[inds], 'y': y[inds], 'err': err[inds], 'PTmodel': PTmodel}
        m = mpfit.mpfit(fitting_function, parinfo=parinfo, functkw=fa)
        
        # get final k
        k_final = np.logspace(-4, np.log10(np.amax(x)), 500)
        
        # get full model for PT
        Pspec = rsd.power_dm.DMSpectrum(k=k_final, z=float(z), transfer_fit='CAMB', include_2loop=False, num_threads=4, cosmo=params)
        PTmodel    = Pspec.P01.total.mu2
        pt_spline = interp.InterpolatedUnivariateSpline(Pspec.k, PTmodel)
        
        # and the full sim model
        sim_spline = interp.InterpolatedUnivariateSpline(x, y, w=1/err)
        
        # now transition nicely
        switch = transition(k_final, k_transition, 0.01)
        P01 = (1-switch)*model(k_final, m.params, pt_spline(k_final)) + switch*sim_spline(k_final)

        if plot:
            norm = 2*Pspec.f*Pspec.D**2*Pspec.power_lin.power
        
            plt.plot(k_final, P01/norm)
            plt.plot(k_final, PTmodel/norm)
            
            norm_spline = interp.InterpolatedUnivariateSpline(k_final, norm)
            plt.errorbar(x, y/norm_spline(x), err/norm_spline(x), c='k', marker='.', ls='')
            
            plt.xlim(0., 0.4)
            plt.ylim(0.9, 4.0)
            plt.xlabel(r"$\mathrm{k \ (h/Mpc)}$", fontsize=16)
            plt.ylabel(r"$P_{01} / (2 f D^2 P_\mathrm{lin}(z=0))$", fontsize=16)
            plt.title(r"$\mathrm{P_{01} \ at \ z \ = \ %s}$" %z)
            plt.show()

        if save:
            np.savetxt("./pkmu_P01_mu2_z_%s.dat" %z, zip(k_final, P01))
        
#end fit_P01

#-------------------------------------------------------------------------------
def fit_P11(params, k_transition=0.1, plot=False, save=True):
    
    zs = ['0.000', '0.509','0.989']
    zlabels = ['007', '005', '004']
    mu = 'mu4'
    for z, zlabel in zip(zs, zlabels):
        
        # read in the data
        data = np.loadtxt("./pkmu_chi_11_m_m_z%s_1-3_02-13binaaQ" %zlabel)
        
        x = data[:,0]
        y = data[:,-3]*x*x
        err = data[:,-1]*x*x
    
        # find where k < 0.1
        inds = np.where(x < k_transition)

        # get the PT model
        Pspec = rsd.power_dm.DMSpectrum(k=x[inds], z=float(z), transfer_fit="CAMB", include_2loop=False, num_threads=4, cosmo=params)
        PTmodel = getattr(Pspec.P11.total, mu)

        # measure the exponential scale
        parinfo = [{'value':1.0, 'fixed':0, 'limited':[0,0], 'limits':[0.,0.]}]
        fa = {'k': x[inds], 'y': y[inds], 'err': err[inds], 'PTmodel': PTmodel}
        m = mpfit.mpfit(fitting_function, parinfo=parinfo, functkw=fa)

        # get final k
        k_final = np.logspace(-4, np.log10(np.amax(x)), 500)

        # get full model for PT
        Pspec = rsd.power_dm.DMSpectrum(k=k_final, z=float(z), transfer_fit="CAMB", include_2loop=False, num_threads=4, cosmo=params)
        PTmodel   = getattr(Pspec.P11.total, mu)
        pt_spline = interp.InterpolatedUnivariateSpline(Pspec.k, PTmodel)

        # and the full sim model
        sim_spline = interp.InterpolatedUnivariateSpline(x, y, w=1/err)

        # now transition nicely
        switch = transition(k_final, k_transition, 0.01)
        P11 = (1-switch)*model(k_final, m.params, pt_spline(k_final)) + switch*sim_spline(k_final)
        
        if plot:
            norm = (Pspec.f*Pspec.D)**2*Pspec.power_lin.power
            
            plt.plot(k_final, P11/norm)
            plt.plot(k_final, PTmodel/norm)
            
            norm_spline = interp.InterpolatedUnivariateSpline(k_final, norm)
            plt.errorbar(x, y/norm_spline(x), err/norm_spline(x), c='k', marker='.', ls='')
            
            plt.xlim(0., 0.4)
            plt.ylim(0.9, 3.0)
            plt.xlabel(r"$\mathrm{k \ (h/Mpc)}$", fontsize=16)
            plt.ylabel(r"$P_{11}[\mu^4] / (f^2 D^2 P_\mathrm{lin}(z=0))$", fontsize=16)
            plt.title(r"$\mathrm{P_{11}[\mu^4] \ at \ z \ = \ %s}$" %z)
            plt.show()

        if save:
            np.savetxt("./pkmu_P11_%s_z_%s.dat" %(mu, z), zip(k_final, P11))
#end fit_P11   

#-------------------------------------------------------------------------------
def fit_Pdv(params, k_transition=0.1, plot=False, save=True):
    
    zs = ['0.000', '0.509','0.989']
    zlabels = ['007', '005', '004']
    for z, zlabel in zip(zs, zlabels):
        
        # read in the data
        data = np.loadtxt("./pkmu_chi_01_m_m_z%s_1-3_02-13binvvQ" %zlabel)
        
        x = data[:,0]
        y = data[:,-2]*(-x)
        err = data[:,-1]*(-x)
        
        # find where k < 0.1
        inds = np.where(x < k_transition)

        # get the PT model
        k_lo = x[inds]
        
        # get the PT model
        Pspec = rsd.power_dm.DMSpectrum(k=x[inds], z=float(z), transfer_fit="CAMB", include_2loop=False, num_threads=4, cosmo=params)
        PTmodel = Pspec.Pdv

        # measure the exponential scale
        parinfo = [{'value':1.0, 'fixed':0, 'limited':[0,0], 'limits':[0.,0.]}]
        fa = {'k': x[inds], 'y': y[inds], 'err': err[inds], 'PTmodel': PTmodel}
        m = mpfit.mpfit(fitting_function, parinfo=parinfo, functkw=fa)

        # get final k
        k_final = np.logspace(-4, np.log10(np.amax(x)), 500)

        # get full model for PT
        Pspec = rsd.power_dm.DMSpectrum(k=k_final, z=float(z), transfer_fit="CAMB", include_2loop=False, num_threads=4, cosmo=params)
        PTmodel   = Pspec.Pdv
        pt_spline = interp.InterpolatedUnivariateSpline(Pspec.k, PTmodel)

        # and the full sim model
        sim_spline = interp.InterpolatedUnivariateSpline(x, y, w=1/err)

        # now transition nicely
        switch = transition(k_final, k_transition, 0.005)
        Pdv = (1-switch)*model(k_final, m.params, pt_spline(k_final)) + switch*sim_spline(k_final)
        #Pdv = (k_final < k_transition)*(1-switch)*model(k_final, m.params, pt_spline(k_final)) + (k_final >= k_transition)*switch*sim_spline(k_final)
        
        if plot:
            norm = (-Pspec.f)*Pspec.D**2*Pspec.power_lin.power
            
            plt.plot(k_final, Pdv/norm)
            plt.plot(k_final, PTmodel/norm)
            
            norm_spline = interp.InterpolatedUnivariateSpline(k_final, norm)
            plt.errorbar(x, y/norm_spline(x), err/norm_spline(x), c='k', marker='.', ls='')
            
            plt.xlim(0., 0.4)
            plt.ylim(0.6, 2.0)
            plt.xlabel(r"$\mathrm{k \ (h/Mpc)}$", fontsize=16)
            plt.ylabel(r"$P_{\delta \theta} / (-f H D^2 P_\mathrm{lin}(z=0))$", fontsize=16)
            plt.title(r"$\mathrm{P_{\delta \theta} \ at \ z \ = \ %s}$" %z)
            plt.show()
        
        if save:
            np.savetxt("./pkmu_Pdv_mu0_z_%s.dat" %z, zip(k_final, Pdv))
#end fit_Pdv

#-------------------------------------------------------------------------------
def main(args):
        
    # the cosmological parameters used in the simulations (close to WMAP5)
    cosmo_params = {'force_flat' : True, 'default' : 'WMAP5', 'h' : 0.701, 'n' : 0.96, 'sigma_8' : 0.807, 'omegac': (1-0.165)*0.279, 'omegal': 0.721, 'omegab': 0.165*0.279, 'omegar': 0.}
    params = cosmology.cosmo.Cosmology(cosmo_params)
    
    fit_P00(params, k_transition=args.k_transition, plot=args.plot, save=args.save)
    fit_P01(params, k_transition=args.k_transition, plot=args.plot, save=args.save)
    fit_P11(params, k_transition=args.k_transition, plot=args.plot, save=args.save)
    fit_Pdv(params, k_transition=args.k_transition, plot=args.plot, save=args.save)
    
#-------------------------------------------------------------------------------    
if __name__ == '__main__':
    
    # parse the input arguments
    desc = "write out interpolated, simulation spectra"
    parser = argparse.ArgumentParser(description=desc)
    
    h = "the wavenumber of transition between PT and sims"
    parser.add_argument('--k_transition', default=0.1, type=float, help=h)
    
    h = "whether to plot the results"
    parser.add_argument('--plot', action='store_true', default=False, help=h)
    
    h = "whether to save the results to a data file"
    parser.add_argument('--save', action='store_true', default=False, help=h)
    
    args = parser.parse_args()
    
    main(args)
