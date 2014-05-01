/* 
Module to compute various linear transfer functions. Adapted from 
Eisenstein and Hu's tf_fit.c code. 

Eisenstein \& Hu (1997) fitting formulas:

    The first set, TFfit_hmpc(), TFset_parameters(), and TFfit_onek(),
    calculate the transfer function for an arbitrary CDM+baryon universe using
    the fitting formula in Section 3 of the paper.  The second set,
    
	The second set, TFsound_horizon_fit(), TFk_peak(), TFnowiggles(), 
	and TFzerobaryon(), calculate other quantities given in Section 4 of the paper. 

Also, routines available to compute tranfer functions from Bardeen, Bond, Kaiser, 
and Szalay (BBKS) 1986 and Bond and Efstathiou 1984
*/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "../include/transfer.h"

/* ------------------------ FITTING FORMULAE ROUTINES ----------------- */

/* There are two routines here.  TFset_parameters() sets all the scalar
parameters, while TFfit_onek() calculates the transfer function for a 
given wavenumber k.  TFfit_onek() may be called many times after a single
call to TFset_parameters() */

/*----------------------------------------------------------------------------*/
void TF_eh_set_parameters(double omega0hh, double f_baryon, double Tcmb)
/*  
    Set all the scalars quantities for Eisenstein & Hu 1997 fitting formula 

    Input:  omega0hh -- The density of CDM and baryons, in units of critical dens,
		                multiplied by the square of the Hubble constant, in units 
		                of 100 km/s/Mpc
            f_baryon -- The fraction of baryons to CDM 
            Tcmb -- The temperature of the CMB in Kelvin.  Tcmb<=0 forces use
			        of the COBE value of  2.728 K. 
    Output: Nothing, but set many global variables used in TFfit_onek(). 
            You can access them yourself, if you want. 
Note: Units are always Mpc, never h^-1 Mpc. 
*/
{
    double z_drag_b1, z_drag_b2;
    double alpha_c_a1, alpha_c_a2, beta_c_b1, beta_c_b2, alpha_b_G, y;

    if (f_baryon<=0.0 || omega0hh<=0.0) {
	fprintf(stderr, "TF_EH_set_parameters(): Illegal input.\n");
	exit(1);
    }
    omhh = omega0hh;
    obhh = omhh*f_baryon;
    if (Tcmb<=0.0) Tcmb=2.728;	/* COBE FIRAS */
    theta_cmb = Tcmb/2.7;

    z_equality = 2.50e4*omhh/POW4(theta_cmb);  /* Really 1+z */
    k_equality = 0.0746*omhh/SQR(theta_cmb);

    z_drag_b1 = 0.313*pow(omhh,-0.419)*(1+0.607*pow(omhh,0.674));
    z_drag_b2 = 0.238*pow(omhh,0.223);
    z_drag = 1291*pow(omhh,0.251)/(1+0.659*pow(omhh,0.828))*
		(1+z_drag_b1*pow(obhh,z_drag_b2));
    
    R_drag = 31.5*obhh/POW4(theta_cmb)*(1000/(1+z_drag));
    R_equality = 31.5*obhh/POW4(theta_cmb)*(1000/z_equality);

    sound_horizon = 2./3./k_equality*sqrt(6./R_equality)*
	    log((sqrt(1+R_drag)+sqrt(R_drag+R_equality))/(1+sqrt(R_equality)));

    k_silk = 1.6*pow(obhh,0.52)*pow(omhh,0.73)*(1+pow(10.4*omhh,-0.95));

    alpha_c_a1 = pow(46.9*omhh,0.670)*(1+pow(32.1*omhh,-0.532));
    alpha_c_a2 = pow(12.0*omhh,0.424)*(1+pow(45.0*omhh,-0.582));
    alpha_c = pow(alpha_c_a1,-f_baryon)*
		pow(alpha_c_a2,-CUBE(f_baryon));
    
    beta_c_b1 = 0.944/(1+pow(458*omhh,-0.708));
    beta_c_b2 = pow(0.395*omhh, -0.0266);
    beta_c = 1.0/(1+beta_c_b1*(pow(1-f_baryon, beta_c_b2)-1));

    y = z_equality/(1+z_drag);
    alpha_b_G = y*(-6.*sqrt(1+y)+(2.+3.*y)*log((sqrt(1+y)+1)/(sqrt(1+y)-1)));
    alpha_b = 2.07*k_equality*sound_horizon*pow(1+R_drag,-0.75)*alpha_b_G;

    beta_node = 8.41*pow(omhh, 0.435);
    beta_b = 0.5+f_baryon+(3.-2.*f_baryon)*sqrt(pow(17.2*omhh,2.0)+1);

    k_peak = 2.5*3.14159*(1+0.217*omhh)/sound_horizon;
    sound_horizon_fit = 44.5*log(9.83/omhh)/sqrt(1+10.0*pow(obhh,0.75));

    alpha_gamma = 1-0.328*log(431.0*omhh)*f_baryon + 0.38*log(22.3*omhh)*
		SQR(f_baryon);
    
    return;
}

/*----------------------------------------------------------------------------*/
double TF_eh_onek(double k)
/*  Eisenstein + Hu 1997 CDM + baryon transfer function

    Input: k -- Wavenumber at which to calculate transfer function, in Mpc^-1.
    Output: Returns the value of the full transfer function fitting formula.
	        This is the form given in Section 3 of Eisenstein & Hu (1997).


    Notes: Units are Mpc, not h^-1 Mpc. 
*/
{
    double T_c_ln_beta, T_c_ln_nobeta, T_c_C_alpha, T_c_C_noalpha;
    double q, xx, xx_tilde, qsq;
    double T_c_f, T_c, s_tilde, T_b_T0, T_b, f_baryon, T_full;


    k = fabs(k);	/* Just define negative k as positive */
    q = k/13.41/k_equality;
    xx = k*sound_horizon;

    T_c_ln_beta = log(2.718282+1.8*beta_c*q);
    T_c_ln_nobeta = log(2.718282+1.8*q);
    T_c_C_alpha = 14.2/alpha_c + 386.0/(1+69.9*pow(q,1.08));
    T_c_C_noalpha = 14.2 + 386.0/(1+69.9*pow(q,1.08));

    qsq = SQR(q);
    T_c_f = 1.0/(1.0+POW4(xx/5.4));
    T_c = T_c_f*T_c_ln_beta/(T_c_ln_beta+T_c_C_noalpha*qsq) +
	    (1-T_c_f)*T_c_ln_beta/(T_c_ln_beta+T_c_C_alpha*qsq);
    
    s_tilde = sound_horizon*pow(1+CUBE(beta_node/xx),-1./3.);
    xx_tilde = k*s_tilde;

    T_b_T0 = T_c_ln_nobeta/(T_c_ln_nobeta+T_c_C_noalpha*SQR(q));
    T_b = sin(xx_tilde)/(xx_tilde)*(T_b_T0/(1+SQR(xx/5.2))+
		alpha_b/(1+CUBE(beta_b/xx))*exp(-pow(k/k_silk,1.4)));
    
    f_baryon = obhh/omhh;
    T_full = f_baryon*T_b + (1-f_baryon)*T_c;

    return T_full;
}

/* ======================= Approximate forms =========================== */

double TF_eh_sound_horizon_fit(double omega0, double f_baryon, double hubble)
/* 
    The approximate value of the sound horizon, in Mpc/h.

    Input:  omega0 -- CDM density, in units of critical density
	        f_baryon -- Baryon fraction, the ratio of baryon to CDM density.
	        hubble -- Hubble constant, in units of 100 km/s/Mpc
    Output: The approximate value of the sound horizon, in h^-1 Mpc.

Note: If you prefer to have the answer in  units of Mpc, use hubble -> 1
and omega0 -> omega0*hubble^2. 
*/
{
    double omhh, sound_horizon_fit_mpc;
    omhh = omega0*hubble*hubble;
    sound_horizon_fit_mpc = 
	44.5*log(9.83/omhh)/sqrt(1+10.0*pow(omhh*f_baryon,0.75));
    return sound_horizon_fit_mpc*hubble;
}

double TF_eh_k_peak(double omega0, double f_baryon, double hubble)
/* 
    The approximate location of the first baryonic peak, in h/Mpc

    Input:  omega0 -- CDM density, in units of critical density
	        f_baryon -- Baryon fraction, the ratio of baryon to CDM density.
	        hubble -- Hubble constant, in units of 100 km/s/Mpc
    Output: The approximate location of the first baryonic peak, in h Mpc^-1
    
Note: If you prefer to have the answer in  units of Mpc^-1, use hubble -> 1
and omega0 -> omega0*hubble^2. 
*/ 
{
    double omhh, k_peak_mpc;
    omhh = omega0*hubble*hubble;
    k_peak_mpc = 2.5*3.14159*(1+0.217*omhh)/TF_eh_sound_horizon_fit(omhh,f_baryon,1.0);
    return k_peak_mpc/hubble;
}

double TF_eh_nowiggles(double omega0, double f_baryon, double hubble, 
		double Tcmb, double k_hmpc)
/*
    The value of an approximate transfer function that captures the
    non-oscillatory part of a partial baryon transfer function.

    Input:  omega0 -- CDM density, in units of critical density
	        f_baryon -- Baryon fraction, the ratio of baryon to CDM density.
	        hubble -- Hubble constant, in units of 100 km/s/Mpc
	        Tcmb -- Temperature of the CMB in Kelvin; Tcmb<=0 forces use of
			        COBE FIRAS value of 2.728 K
	        k_hmpc -- Wavenumber in units of (h Mpc^-1). 
    Output: The value of an approximate transfer function that captures the
            non-oscillatory part of a partial baryon transfer function.  
            In other words, the baryon oscillations are left out, but the 
            suppression of power below the sound horizon is included. 
            See equations (30) and (31).

Note: If you prefer to use wavenumbers in units of Mpc^-1, use hubble -> 1
and omega0 -> omega0*hubble^2. 
*/ 
{
    double k, omhh, theta_cmb, k_equality, q, xx, alpha_gamma, gamma_eff;
    double q_eff, T_nowiggles_L0, T_nowiggles_C0;

    k = k_hmpc*hubble;	/* Convert to Mpc^-1 */
    omhh = omega0*hubble*hubble;
    if (Tcmb<=0.0) Tcmb=2.728;	/* COBE FIRAS */
    theta_cmb = Tcmb/2.7;

    k_equality = 0.0746*omhh/SQR(theta_cmb);
    q = k/13.41/k_equality;
    xx = k*TF_eh_sound_horizon_fit(omhh, f_baryon, 1.0);

    alpha_gamma = 1-0.328*log(431.0*omhh)*f_baryon + 0.38*log(22.3*omhh)*
		SQR(f_baryon);
    gamma_eff = omhh*(alpha_gamma+(1-alpha_gamma)/(1+POW4(0.43*xx)));
    q_eff = q*omhh/gamma_eff;

    T_nowiggles_L0 = log(2.0*2.718282+1.8*q_eff);
    T_nowiggles_C0 = 14.2 + 731.0/(1+62.5*q_eff);
    return T_nowiggles_L0/(T_nowiggles_L0+T_nowiggles_C0*SQR(q_eff));
}

/* ======================= Zero Baryon Formula =========================== */

double TF_eh_zerobaryon(double omega0, double hubble, double Tcmb, double k_hmpc)
/* 
    The value of the transfer function for a zero-baryon universe.
    
    Input:  omega0 -- CDM density, in units of critical density
	        hubble -- Hubble constant, in units of 100 km/s/Mpc
	        Tcmb -- Temperature of the CMB in Kelvin; Tcmb<=0 forces use of
			        COBE FIRAS value of 2.728 K
	        k_hmpc -- Wavenumber in units of (h Mpc^-1). 
	Output: The value of the transfer function for a zero-baryon universe. 

Note: If you prefer to use wavenumbers in units of Mpc^-1, use hubble -> 1
and omega0 -> omega0*hubble^2. 
*/ 
{
    double k, omhh, theta_cmb, k_equality, q, T_0_L0, T_0_C0;

    k = k_hmpc*hubble;	/* Convert to Mpc^-1 */
    omhh = omega0*hubble*hubble;
    if (Tcmb<=0.0) Tcmb=2.728;	/* COBE FIRAS */
    theta_cmb = Tcmb/2.7;

    k_equality = 0.0746*omhh/SQR(theta_cmb);
    q = k/13.41/k_equality;

    T_0_L0 = log(2.0*2.718282+1.8*q);
    T_0_C0 = 14.2 + 731.0/(1+62.5*q);
    return T_0_L0/(T_0_L0+T_0_C0*q*q);
}

/*----------------------------------------------------------------------------*/
double TF_bbks(double omega_m, double f_baryon, double hubble, double k_hmpc)
/*
    The value of the linear BBKS transfer function from Bardeen et al. 1986. 
    Fitting formula taken from eq 7.70 in Dodelson's Modern Cosmology.
    
    Input:  omega_m -- The matter density, in units of the critical density
            hubble -- Hubble constant, in units of 100 km/s/Mpc
            k_hmpc -- Wavenumber in units of h/Mpc
    Output: The value of the BBKS transfer function

Note: This depends on q = k (in Mpc) / (omega_m*h^2)  
*/
{
    double q, gamma, omega_b;
    
    gamma = omega_m*hubble;
    omega_b = omega_m*f_baryon;
    q = k_hmpc / gamma * exp(omega_b + sqrt(2.*hubble)*f_baryon); 
    
    return log(1. + 2.34*q)/(2.34*q)*pow((1. + 3.89*q + SQR(16.1*q) + CUBE(5.47*q) + POW4(6.71*q)), -0.25);
}

/*----------------------------------------------------------------------------*/
double TF_bond_efs(double omega_m, double f_baryon, double hubble, double k_hmpc)
/*
    The value of the linear Bond and Efstathiou 1984 transfer function.
    
    Input:  omega_m -- The matter density, in units of the critical density
            hubble -- Hubble constant, in units of 100 km/s/Mpc
            k_hmpc -- Wavenumber in units of h/Mpc
    Output: The value of the linear transfer function 
*/
{
    double q, gamma, omega_b;
    
    gamma = omega_m*hubble;
    omega_b = omega_m*f_baryon;
    q = k_hmpc / gamma * exp(omega_b + sqrt(2.*hubble)*f_baryon);

    return pow(1 + pow(6.4*q + pow(3.*q, 1.5) + (1.7*q)*(1.7*q), 1.13), (-1/1.13));
}
/*----------------------------------------------------------------------------*/