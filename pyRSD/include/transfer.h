#ifndef __TRANSFER_H
#define __TRANSFER_H

// Eisenstein + Hu fitting formulas
void TF_eh_set_parameters(double omega0hh, double f_baryon, double Tcmb);
double TF_eh_onek(double k); 
double TF_eh_sound_horizon_fit(double omega0, double f_baryon, double hubble);
double TF_eh_k_peak(double omega0, double f_baryon, double hubble);
double TF_eh_nowiggles(double omega0, double f_baryon, double hubble, double Tcmb, double k_hmpc);
double TF_eh_zerobaryon(double omega0, double hubble, double Tcmb, double k_hmpc);

// BBKS fitting formula
double TF_bbks(double omega_m, double f_baryon, double hubble, double k_hmpc);

// Bond and Efstathiou fitting formula
double TF_bond_efs(double omega_m, double f_baryon, double hubble, double k_hmpc);

/* Global variables */
double  omhh,		        /* Omega_matter*h^2 */
	    obhh,		        /* Omega_baryon*h^2 */
	    theta_cmb,	        /* Tcmb in units of 2.7 K */
	    z_equality,	        /* Redshift of matter-radiation equality, really 1+z */
	    k_equality,	        /* Scale of equality, in Mpc^-1 */
	    z_drag,		        /* Redshift of drag epoch */
	    R_drag,	        	/* Photon-baryon ratio at drag epoch */
	    R_equality,	        /* Photon-baryon ratio at equality epoch */
	    sound_horizon,	    /* Sound horizon at drag epoch, in Mpc */
	    k_silk,		        /* Silk damping scale, in Mpc^-1 */
	    alpha_c,	        /* CDM suppression */
	    beta_c,	        	/* CDM log shift */
	    alpha_b,        	/* Baryon suppression */
	    beta_b,	        	/* Baryon envelope shift */
	    beta_node,	        /* Sound horizon shift */
	    k_peak,		        /* Fit to wavenumber of first peak, in Mpc^-1 */
	    sound_horizon_fit,	/* Fit to sound horizon, in Mpc */
	    alpha_gamma;	    /* Gamma suppression in approximate TF */

/* Convenience from Numerical Recipes in C, 2nd edition */
static double sqrarg;
#define SQR(a) ((sqrarg=(a)) == 0.0 ? 0.0 : sqrarg*sqrarg)
static double cubearg;
#define CUBE(a) ((cubearg=(a)) == 0.0 ? 0.0 : cubearg*cubearg*cubearg)
static double pow4arg;
#define POW4(a) ((pow4arg=(a)) == 0.0 ? 0.0 : pow4arg*pow4arg*pow4arg*pow4arg)

#endif