"""
 write_spectra.py
 write out power spectra measured from simulations
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 03/18/2014
"""
from pyRSD import rsd
from utils import mpfit

import scipy.interpolate as interp
import numpy as np
import matplotlib.pyplot as plt
import argparse

#-------------------------------------------------------------------------------
# tools
#-------------------------------------------------------------------------------
def transition(k, k_transition, b):
    return 0.5 + 0.5*np.tanh((k-k_transition)/b)

def transition_model(k, p, pt_model):
    return np.exp(p[0]*k*k)*pt_model

def fitting_function(p, fjac=None, k=None, y=None, err=None, pt_model=None):
    m = transition_model(k, p, pt_model)
    status = 0
    return ([status, (y-m)/err])

#-------------------------------------------------------------------------------
# SPT functions
#-------------------------------------------------------------------------------
def P00_spt(model, k, z):
    """
    Return P00(k,z) in SPT
    """
    model.z = z
    model.sigma8_z = model.cosmo.Sigma8_z(z)
    model.f = model.cosmo.f_z(z)
    
    # the necessary integrals 
    I00 = model.I00(k)
    J00 = model.J00(k)

    # evaluate
    P11 = model.normed_power_lin(k)
    P22 = 2*I00
    P13 = 6*k**2*J00*P11
    return P11 + P22 + P13


def P01_spt(model, k, z):
    """
    Return P01(k,z) in SPT
    """
    model.z = z
    model.sigma8_z = model.cosmo.Sigma8_z(z)
    model.f = model.cosmo.f_z(z)
    
    # the necessary integrals 
    I00 = model.I00(k)
    J00 = model.J00(k)

    Plin = model.normed_power_lin(k)
    return 2*model.f*(Plin + 4.*(I00 + 3*k**2*J00*Plin))
    

def P11_mu2_spt(model, k, z):
    """
    Return P11(k,z)[mu^2] in SPT
    """
    model.z = z
    model.sigma8_z = model.cosmo.Sigma8_z(z)
    model.f = model.cosmo.f_z(z)
    
    # the necessary integrals 
    I11 = model.I11(k)
    I22 = model.I22(k)
    I13 = model.I13(k)
    J11 = model.J11(k)
    J10 = model.J10(k)

    Plin = model.normed_power_lin(k)
    return model.f**2 * (Plin + 2*I11 + 4*I22 + I13 + 6*k**2 * (J11 + 2*J10)*Plin)
    
def P11_mu2_spt(model, k, z):
    """
    Return P11(k,z)[mu^2] in SPT
    """
    model.z = z
    model.sigma8_z = model.cosmo.Sigma8_z(z)
    model.f = model.cosmo.f_z(z)

    return model.f**2 * model.I31(k) 

def P11_mu4_spt(model, k, z):
    """
    Return P11(k,z)[mu^4] in SPT
    """
    model.z = z
    model.sigma8_z = model.cosmo.Sigma8_z(z)
    model.f = model.cosmo.f_z(z)
    
    # the necessary integrals 
    I11 = model.I11(k)
    I22 = model.I22(k)
    I13 = model.I13(k)
    J11 = model.J11(k)
    J10 = model.J10(k)

    Plin = model.normed_power_lin(k)
    return model.f**2 * (Plin + 2*I11 + 4*I22 + I13 + 6*k**2 * (J11 + 2*J10)*Plin)
    
def Pdv_spt(model, k, z):
    """
    Return Pdv(k,z) in SPT
    """
    model.z = z
    model.sigma8_z = model.cosmo.Sigma8_z(z)
    model.f = model.cosmo.f_z(z)

    # the necessary integrals 
    I01 = model.I01(k)
    J01 = model.J01(k)

    # evaluate
    P11 = model.normed_power_lin(k)
    P22 = 2*I01
    P13 = 6*k**2*J01*P11
    return (-model.f) * (P11 + P22 + P13)
    
#-------------------------------------------------------------------------------
# fitting functions
#-------------------------------------------------------------------------------
def fit_P00(model, k_transition=0.1, plot=False, save=True, tag=""):
    """
    Fit a smooth function to the measured P00 from simulations, extrapolating
    using SPT at low-k
    """
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

        # initialize the PT model
        pt_model = P00_spt(model, x[inds], float(z))

        # measure the exponential scale
        parinfo = [{'value':1.0, 'fixed':0, 'limited':[0,0], 'limits':[0.,0.]}]
        fa = {'k': x[inds], 'y': y[inds], 'err': err[inds], 'pt_model': pt_model}
        m = mpfit.mpfit(fitting_function, parinfo=parinfo, functkw=fa)
        
        # full k results
        k_final = np.logspace(np.log10(5e-6), np.log10(np.amax(x)), 500)
        pt_full = P00_spt(model, k_final, float(z))
        sim_spline = interp.InterpolatedUnivariateSpline(x, y, w=1/err)
        
        # now transition nicely
        switch = transition(k_final, k_transition, 0.01)
        P00 = (1-switch)*transition_model(k_final, m.params, pt_full) + switch*sim_spline(k_final)
        
        if plot:
            norm = model.normed_power_lin_nw(k_final)
            plt.plot(k_final, P00/norm)
            plt.plot(k_final, pt_full/norm)
            
            norm = model.normed_power_lin_nw(x)
            plt.errorbar(x, y/norm, err/norm, c='k', marker='.', ls='')
            
            plt.xlim(0., 0.4)
            plt.ylim(0.7, 2.0)
            plt.xlabel(r"$\mathrm{k \ (h/Mpc)}$", fontsize=16)
            plt.ylabel(r"$P_{00} / (D^2 P_\mathrm{lin}(z=0))$", fontsize=16)
            plt.title(r"$\mathrm{P_{00} \ at \ z \ = \ %s}$" %z)
            plt.show()
        
        if save: 
            t = "_"+tag if tag else ""
            np.savetxt("./pkmu_P00_mu0_z_%s%s.dat" %(z, t), zip(k_final, P00))

def fit_P01(model, k_transition=0.1, plot=False, save=True, tag=""):
    """
    Fit a smooth function to the measured P01 from simulations, extrapolating
    using SPT at low-k
    """
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

        # initialize the PT model
        pt_model = P01_spt(model, x[inds], float(z))

        # measure the exponential scale
        parinfo = [{'value':1.0, 'fixed':0, 'limited':[0,0], 'limits':[0.,0.]}]
        fa = {'k': x[inds], 'y': y[inds], 'err': err[inds], 'pt_model': pt_model}
        m = mpfit.mpfit(fitting_function, parinfo=parinfo, functkw=fa)
        
        # full k results
        k_final = np.logspace(np.log10(5e-6), np.log10(np.amax(x)), 500)
        pt_full = P01_spt(model, k_final, float(z))
        sim_spline = interp.InterpolatedUnivariateSpline(x, y, w=1/err)
        
        # now transition nicely
        switch = transition(k_final, k_transition, 0.01)
        P01 = (1-switch)*transition_model(k_final, m.params, pt_full) + switch*sim_spline(k_final)

        if plot:
            norm = 2*model.f*model.normed_power_lin_nw(k_final)
            plt.plot(k_final, P01/norm)
            plt.plot(k_final, pt_full/norm)
            
            norm = 2*model.f*model.normed_power_lin_nw(x)
            plt.errorbar(x, y/norm, err/norm, c='k', marker='.', ls='')
            
            plt.xlim(0., 0.4)
            plt.ylim(0.7, 4.0)
            plt.xlabel(r"$\mathrm{k \ (h/Mpc)}$", fontsize=16)
            plt.ylabel(r"$P_{01} / (2 f D^2 P_\mathrm{lin}(z=0))$", fontsize=16)
            plt.title(r"$\mathrm{P_{01} \ at \ z \ = \ %s}$" %z)
            plt.show()

        if save:
            t = "_"+tag if tag else ""
            np.savetxt("./pkmu_P01_mu2_z_%s%s.dat" %(z, t), zip(k_final, P01))


def fit_P11(model, mu, k_transition=0.1, plot=False, save=True, tag=""):
    """
    Fit a smooth function to the measured P11 from simulations, extrapolating
    using SPT at low-k
    """
    zs = ['0.000', '0.509','0.989']
    zlabels = ['007', '005', '004']
    
    if mu == 'mu4':
        mu_lab = r'\mu^4'
        model_callable = P11_mu4_spt
    else:
        mu_lab = r'\mu^2'
        model_callable = P11_mu2_spt
        
    for z, zlabel in zip(zs, zlabels):
        
        # read in the data
        data = np.loadtxt("./pkmu_chi_11_m_m_z%s_1-3_02-13binaaQ" %zlabel)
        
        x = data[:,0]
        if mu == 'mu4':
            y = data[:,-3]*x**2
            err = data[:,-1]*x**2
        else:
            y = data[:,-4]*x**2
            err = data[:,-2]*x**2
    
        # find where k < 0.1
        inds = np.where(x < k_transition)

        # initialize the PT model
        pt_model = model_callable(model, x[inds], float(z))

        # measure the exponential scale
        parinfo = [{'value':1.0, 'fixed':0, 'limited':[0,0], 'limits':[0.,0.]}]
        fa = {'k': x[inds], 'y': y[inds], 'err': err[inds], 'pt_model': pt_model}
        m = mpfit.mpfit(fitting_function, parinfo=parinfo, functkw=fa)

        # full k results
        k_final = np.logspace(np.log10(5e-6), np.log10(np.amax(x)), 500)
        pt_full = model_callable(model, k_final, float(z))
        sim_spline = interp.InterpolatedUnivariateSpline(x, y, w=1/err)

        # now transition nicely
        switch = transition(k_final, k_transition, 0.01)
        P11 = (1-switch)*transition_model(k_final, m.params, pt_full) + switch*sim_spline(k_final)
        
        if plot:
            norm = model.f**2*model.normed_power_lin_nw(k_final)
            plt.plot(k_final, P11/norm)
            plt.plot(k_final, pt_full/norm)
            
            norm = model.f**2*model.normed_power_lin_nw(x)
            plt.errorbar(x, y/norm, err/norm, c='k', marker='.', ls='')
            
            plt.xlim(0., 0.4)
            if mu == 'mu4':
                plt.ylim(0.7, 5.0)
            else:
                plt.ylim(0., 5.0)
            plt.xlabel(r"$\mathrm{k \ (h/Mpc)}$", fontsize=16)
            plt.ylabel(r"$P_{11}[%s] / (f^2 D^2 P_\mathrm{lin}(z=0))$" %mu_lab, fontsize=16)
            plt.title(r"$\mathrm{P_{11}[%s] \ at \ z \ = \ %s}$" %(mu_lab, z))
            plt.show()

        if save:
            t = "_"+tag if tag else ""
            np.savetxt("./pkmu_P11_%s_z_%s%s.dat" %(mu, z, t), zip(k_final, P11))
 

def fit_Pdv(model, k_transition=0.07, plot=False, save=True, tag=""):
    """
    Fit a smooth function to the measured Pdv from simulations, extrapolating
    using SPT at low-k
    """
    zs = ['0.000', '0.509','0.989']
    zlabels = ['007', '005', '004']
    for z, zlabel in zip(zs, zlabels):
        
        # read in the data
        data = np.loadtxt("./pkmu_chi_01_m_m_z%s_1-3_02-13binvvQ" %zlabel)
        
        x = data[:,0]
        y = data[:,-2]*(-x)
        err = data[:,-1]*(x)
        
        # find where k < 0.1
        inds = np.where(x < k_transition)

        # get the PT model
        k_lo = x[inds]
        
        # initialize the PT model
        pt_model = Pdv_spt(model, x[inds], float(z))

        # measure the exponential scale
        parinfo = [{'value':1.0, 'fixed':0, 'limited':[0,0], 'limits':[0.,0.]}]
        fa = {'k': x[inds], 'y': y[inds], 'err': err[inds], 'pt_model': pt_model}
        m = mpfit.mpfit(fitting_function, parinfo=parinfo, functkw=fa)

        # full k results
        k_final = np.logspace(np.log10(5e-6), np.log10(np.amax(x)), 500)
        pt_full = Pdv_spt(model, k_final, float(z))
        sim_spline = interp.UnivariateSpline(x, y, w=1/err)

        # now transition nicely
        switch = transition(k_final, k_transition, 0.01)
        Pdv = (1-switch)*transition_model(k_final, m.params, pt_full) + switch*sim_spline(k_final)
        
        if plot:
            norm = -model.f*model.normed_power_lin_nw(k_final)
            plt.plot(k_final, Pdv/norm)
            plt.plot(k_final, pt_full/norm)
            
            norm = -model.f*model.normed_power_lin_nw(x)
            plt.errorbar(x, y/norm, err/norm, c='k', marker='.', ls='')
            
            plt.xlim(0., 0.4)
            plt.ylim(0.6, 2.0)
            plt.xlabel(r"$\mathrm{k \ (h/Mpc)}$", fontsize=16)
            plt.ylabel(r"$P_{\delta \theta} / (-f H D^2 P_\mathrm{lin}(z=0))$", fontsize=16)
            plt.title(r"$\mathrm{P_{\delta \theta} \ at \ z \ = \ %s}$" %z)
            plt.show()
        
        if save:
            t = "_"+tag if tag else ""
            np.savetxt("./pkmu_Pdv_mu0_z_%s%s.dat" %(z, t), zip(k_final, Pdv))

def main(args):
    """
    The main function
    """
    
    # initialize the DM model
    model = rsd.DarkMatterSpectrum(cosmo_filename="teppei_sims.ini")
        
    # do the fitting  
    for i, w in enumerate(args.which):
        kwargs = {'k_transition':args.k0[i], 'plot':args.plot, 'save':args.save, 'tag':args.tag}
        if w == 'P00':  
            fit_P00(model, **kwargs)
        elif w == 'P01':
            fit_P01(model, **kwargs)
        elif w == 'P11_mu2':
            fit_P11(model, 'mu2', **kwargs)
        elif w == 'P11_mu4':
            fit_P11(model, 'mu4', **kwargs)
        elif w == 'Pdv':
            fit_Pdv(model, **kwargs)
    
#-------------------------------------------------------------------------------    
if __name__ == '__main__':
    
    # parse the input arguments
    desc = "write out interpolated, simulation spectra"
    parser = argparse.ArgumentParser(description=desc)
    
    choices = ['P00', 'P01', 'P11_mu2', 'P11_mu4', 'Pdv']
    h = 'which power spectra to compute results for'
    parser.add_argument('which', nargs='+', choices=choices, help=h)
    
    h = "the wavenumber of transition between PT and sims"
    parser.add_argument('--k0', default=[0.1], type=float, nargs='*', help=h)
    
    h = "a tag to append to the output file names"
    parser.add_argument('--tag', default="", type=str, help=h)
    
    h = "whether to plot the results"
    parser.add_argument('--plot', action='store_true', default=False, help=h)
    
    h = "whether to save the results to a data file"
    parser.add_argument('--save', action='store_true', default=False, help=h)
    
    args = parser.parse_args()
    
    if len(args.k0) == 1:
        args.k0 = [args.k0[0]]*len(args.which)
    if len(args.k0) != len(args.which):
        raise ValueError("mismatch between desired spectra and `k_transition` values")
    
    main(args)
