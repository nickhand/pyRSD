#ifndef __POWER_H
#define __POWER_H

/* global variables */
double  omega_m,		/* Density of CDM and baryons, in units of critical density */
        omega_b,        /* Density of baryons, in units of critical density */
        omega_l,		/* Density of dark energy in units of critical density */
        omega_r,        /* Density of radiation in units of critical density */
        omega_k,        /* Curvature parameter */
        sigma_8,		/* Mass variance on scale of R = 8 Mpc/h */
        n_spec, 		/* Primordial power spectrum spectral index */
        hubble,         /* Hubble parameter in units of 100 km/s/Mpc */
        Tcmb,           /* Temperature of the CMB*/
        w_lam,          /* Dark energy equation of state */
        f_baryon,       /* Baryon fraction */
        power_norm;      /* normalization of the power spectrum */
        
int  transfer;          /* which transfer function to use */


// the main callable cosmology functions
void set_parameters(double OMEGAM, double OMEGAB, double OMEGAL, double OMEGAR, 
                double SIGMA8, double HUBBLE, double NSPEC, double TCMB, 
                double W_LAM, int TRANSFER);
double E(double a);
void D_plus(double *z, int numz, int normed, double *growth);
void sigma_r(double *r, double z, int numr, int normed, double *sigma);
void linear_power(double *k, double z, int numk, double *power);
void nonlinear_power(double *k, double z, int numk, double *power);
void dlnsdlnm_integral(double *r, int numr, double *output);
double omegal_a(double a);
double omegam_a(double a);


// these are internal helper functions for the main cosmology ones
double normalize_power(void);
double growth_function_integrand(double a, void *params);
double sigma2_integrand(double k, void *params);
double dlnsdlnm_integrand(double k, void *params);
double linear_power_unnorm(double k);


// halofit functions
void halofit(double rk, double rn, double rncur, double rknl, double plin, 
    double om_m, double om_v, double *pnl);
double knl_integrand(double logk);
double neff_integrand(double logk);
double ncur_integrand(double logk);
void wint(double r, double *sig, double *d1, double *d2, double amp, int onlysig);

//  numerical recipes functions used during the nonlinear power calculation
double dlog(double x);
double qromb1(double (*func)(double), double a, double b);
double trapzd1(double (*func)(double), double a, double b, int n);
void error(const char *s);
void polint(double xa[], double ya[], int n, double x, double *y, double *dy);
void free_vector(double *v, long nl, long nh);
double *vector(long nl, long nh);

#define NR_END 1
#define FREE_ARG char*

#endif