/* ====================================================== *
 * MK 1/2005						  *
 * Nonlinear density (3-d) and convergence (2-d) power    *
 * spectra for LambdaCDM models.			  *
 * Based on Peacock&Dodds (1996) and Smith et al. (2002). *
 * The latter nonlinear model incorporates an improved    *
 * and faster version of halofit.f.			  *
 * ====================================================== */

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <gsl/gsl_integration.h>
#include "transfer.h"
#include "power.h"

/*----------------------------------------------------------------------------*/

void set_parameters(double OMEGAM, double OMEGAB, double OMEGAL, double OMEGAR, 
                    double SIGMA8, double HUBBLE, double NSPEC, double TCMB, 
                    double W_LAM, int TRANSFER)
/* 
    Set the parameters needed for power spectrum calculations 
    
    Input:  OMEGAM -- Density of CDM and baryons, in units of critical density 
            OMEGAB -- Density of baryons, in units of critical density 
            OMEGAV -- Density of dark energy in units of critical density
            OMEGAR -- Density of radiation, in units of critical density
            SIGMA8 -- Mass variance on scale of R = 8 Mpc/h 
            HUBBLE -- Hubble parameter in units of 100 km/s/Mpc 
            NSPEC -- Primordial power spectrum spectral index 
            TCMB -- Temperature of the CMB,
            W_LAM -- Dark energy equation of state
            TRANSFER -- Defines which transfer function to use
*/
{
    omega_m  = OMEGAM;
    omega_b  = OMEGAB;
    omega_l  = OMEGAL;
    omega_r  = OMEGAR;
    omega_k  = 1. - omega_m - omega_l - omega_r;
    sigma_8  = SIGMA8;
    n_spec   = NSPEC; 
    hubble   = HUBBLE;
    Tcmb     = TCMB;
    w_lam    = W_LAM;
    f_baryon = omega_b/omega_m; 
    transfer = TRANSFER;     
    
    if ((transfer == 0) || (transfer == 1) || (transfer == 2)) {
        TF_eh_set_parameters(omega_m*hubble*hubble, f_baryon, Tcmb);
    }
    power_norm = normalize_power();
}
/*----------------------------------------------------------------------------*/
double E(double a) {
/*
    The dimensionaless Hubble parameter, H(z) / H0
    
    Input: a -- Scale factor 
*/
    if (a == 0) {
        return 0.;
    } else {
        return sqrt(omega_m/(a*a*a) + omega_r/(a*a*a*a) + omega_k/(a*a) + omega_l/pow(a, 3.*(1+w_lam)));
    }
}

/*----------------------------------------------------------------------------*/
double normalize_power(void)
/*
    Normalize the power spectrum to the value of sigma_8 at z = 0
*/
{
    double sigma0, rnorm = 8.;
    sigma_r(&rnorm, 0., 1, 0, &sigma0);
    return sigma_8*sigma_8 / (sigma0*sigma0);
}

/*----------------------------------------------------------------------------*/
double growth_function_integrand(double a, void *params)
/*
    Integrand used internally in D_plus() to compute the growth function
*/
{
    double integral, Ea;
        
    if (a == 0.) {
        integral = 0.;
    } else {
        Ea = E(a);
        integral = 1./(Ea*Ea*Ea*a*a*a);   
    }
    return integral;
}
/*----------------------------------------------------------------------------*/
void D_plus(double *z, int numz, int normed, double *growth)
/*
    The linear growth function, normalized to unity at z = 0, if normed=1
    
    Input:  z -- the array of redshifts to compute the function at
            numz -- the number of redshifts to compute at
            normed -- whether to normalize to unity at z = 0
            growth -- the growth function values to return
*/
{
    int i;
    double result, error, Ea, norm;
    
    // set up the gsl function and integration workspace
    gsl_integration_cquad_workspace * work = gsl_integration_cquad_workspace_alloc(1000);
    gsl_function F;
    F.function = &growth_function_integrand;
    F.params = NULL;

    // compute the normalization
    if (normed) {
        double this_z = 0.;
        D_plus(&this_z, 1, 0, &norm);
    } else {
        norm = 1.;
    }

    
    // integrate for every redshift provided    
    for (i=0; i < numz; i++) {
        gsl_integration_cquad(&F, 0., 1./(1.+z[i]), 0., 1e-5, work, &result, &error, NULL);
        
        Ea = E(1./(1.+z[i]));
        growth[i] = 5./2.*omega_m*Ea*result/norm;
    }
    // free the integration workspace
    gsl_integration_cquad_workspace_free(work);
}
/*----------------------------------------------------------------------------*/
double sigma2_integrand(double k, void *params)
/*
    Integrand of the sigma squared integral, used internally by the 
    sigma_r() function. It is equal to k^2 W(kr)^2 T(k)^2 k^n_spec, 
    where the transfer function used is specified by the global 
    parameter ``transfer``.
*/
{
    double r  = *((double*)params);
    double W  = 3.*(sin(k*r)-k*r*cos(k*r))/(k*k*k*r*r*r);
    double Pk;
       
    Pk = linear_power_unnorm(k);
    return (k*k*W*W)*Pk/(2.*M_PI*M_PI);
}
/*----------------------------------------------------------------------------*/
void sigma_r(double *r, double z, int numr, int normed, double *sigma)
/*
    The average mass fluctuation within a sphere of radius r, using the specified
    transfer function for the linear power spectrum.
    
    Input: r -- radii to compute the statistic at in Mpc/h
           z -- the redshift to compute at
           numr -- the number of radii for which to compute sigma
           normed -- whether to normalize to sigma_8 at z = 0
           sigma -- the output statistic         
*/
{
    int i;
    double result, error;
    double norm, growth;
    
    // set up the gsl function and integration workspace
    gsl_integration_cquad_workspace * work = gsl_integration_cquad_workspace_alloc(1000);
    gsl_function F;
    F.function = &sigma2_integrand;
    
    // compute the normed growth function for this redshift 
    D_plus(&z, 1, 1, &growth);
    
    // compute the normalization at r = 8 Mpc/h
    if (normed) {
        norm = power_norm;
    } else {
        norm = 1.;
    }
    
    // integrate for every radius provided    
    for (i=0; i < numr; i++) {
        F.params = &r[i];
        gsl_integration_cquad(&F, 1e-3/r[i], 100./r[i], 0., 1e-5, work, &result, &error, NULL);
        sigma[i] = growth*norm*sqrt(result);
    }
    
    // free the integration workspace
    gsl_integration_cquad_workspace_free(work);
}
/*----------------------------------------------------------------------------*/
void linear_power(double *k, double z, int numk, double *power)
/*
    The linear matter power spectrum, using the transfer function specified
    by the ``transfer`` global parameter.
    
    Input: k -- wavenumber to compute power spectrum at in h/Mpc
           z -- the redshift to compute at
           numk -- the number of wavenumber for which to compute power
           power -- the output power spectrum
*/
{
    int i;
    double growth, Pk;
        
    // compute the growth function 
    D_plus(&z, 1, 1, &growth);
        
    for (i=0; i < numk; i++){
        
        Pk = linear_power_unnorm(k[i]);
        power[i] = power_norm*(growth*growth)*Pk;
    }
}
/*----------------------------------------------------------------------------*/
double dlnsdlnm_integrand(double k, void *params)
/*
    Integrand for the dln(sigma)/dln(mass) integral, for use internally
    by the dlnsdlnm_integral() function.
*/
{

    double r  = *((double*)params);
    double Pk, dW2dm; 
    
    Pk = linear_power_unnorm(k);
    dW2dm = (sin(k*r)-k*r*cos(k*r))*(sin(k*r)*(1.-3./(k*k*r*r))+3*cos(k*r)/(k*r));
    return dW2dm*Pk/(k*k);
}
/*----------------------------------------------------------------------------*/
void dlnsdlnm_integral(double *r, int numr, double *output)
/*
    Helper function for computing the integral in dln(sigma)/dln(mass). It is
    used to computed halo mass functions and is given by \int dW^2/dM P(k) / k^2 dk.
    Note that it is computed at z = 0.
    
    Input: r -- radii to compute the statistic at in Mpc/h
           numr -- the number of radii for which to compute sigma
           output -- the output statistic
*/
{
    int i;
    double result, error;
    
    // set up the gsl function and integration workspace
    gsl_integration_cquad_workspace * work = gsl_integration_cquad_workspace_alloc(1000);
    gsl_function F;
    F.function = &dlnsdlnm_integrand;
    
    // integrate for every radius provided    
    for (i=0; i < numr; i++) {
        F.params = &r[i];
        gsl_integration_cquad(&F, 1e-3/r[i], 100./r[i], 0., 1e-5, work, &result, &error, NULL);
        output[i] = result*power_norm;
    }
    
    // free the integration workspace
    gsl_integration_cquad_workspace_free(work);
}
/*----------------------------------------------------------------------------*/
double linear_power_unnorm(double k)
/*
    Helper function to compute the unnormalize linear power for one k (in h/Mpc),
    given by T(k)^2 * k^n_spec
*/
{ 
    double Tk;
    
    if (transfer == 0) {
        Tk = TF_eh_onek(k*hubble);
    } else if (transfer == 1) {
        Tk = TF_eh_nowiggles(omega_m, f_baryon, hubble, Tcmb, k);
    } else if (transfer == 2) {
        Tk = TF_eh_zerobaryon(omega_m, hubble, Tcmb, k);
    } else if (transfer == 3) {
        Tk = TF_bbks(omega_m, f_baryon, hubble, k);
    } else if (transfer == 4) {
        Tk = TF_bond_efs(omega_m, f_baryon, hubble, k);
    }
    return Tk*Tk*pow(k, n_spec);
}

/*----------------------------------------------------------------------------*/
double omegal_a(double a) 
/*
    The dark energy density omega_l as a function of scale factor.
*/
{   double Ea = E(a);
    return omega_l / (Ea*Ea);
}

/*----------------------------------------------------------------------------*/
double omegam_a(double a)
/*
    The matter density omega_m as a function of scale factor.
*/
{
    double Ea = E(a);
    return omega_m / (Ea*Ea*a*a*a);
}
/*----------------------------------------------------------------------------*/
void nonlinear_power(double *k, double z, int numk, double *power)
/*
    The nonlinear matter power spectrum, using the Halofit prescription.
    
    Input: k -- wavenumber to compute power spectrum at in h/Mpc
           z -- the redshift to compute at
           numk -- the number of wavenumber for which to compute power
           power -- the output power spectrum
*/
{
    const double k_nonlin_max = 1.e6;   
    const int itermax  = 20;
    const double logstep = 5.0;
       
    double a;
    double sig, rknl, rneff, rncur, d1, d2;
    double diff, logr1, logr2, rmid;
    double logr1start, logr2start, logrmid, logrmidtmp;
    double om_m, om_v;
    double amp, Pk ;
    double Delta_NL, Delta_L;
    int iter, i, golinear;
    
    a = 1./(1.+z);
    om_m = omegam_a(a);
    om_v = omegal_a(a);
    golinear = 0;
        
    // the redshift growth
    D_plus(&z, 1, 1, &amp);
    
    /*  calculate nonlinear wavenumber (rknl), effective spectral index (rneff) and 
        curvature (rncur) of the power spectrum at the desired redshift, using method 
        described in Smith et al (2002). */
    logr1 = -2.0;
    logr2 =  3.5;
    
    iterstart:

    logr1start = logr1;
    logr2start = logr2;

    iter = 0;
    do {
        logrmid = 0.5*(logr2+logr1);
        rmid    = pow(10, logrmid);
        wint(rmid, &sig, 0x0, 0x0, amp, 1);

        diff = sig - 1.0;

        if (diff > 0.001)
            logr1 = dlog(rmid);
        if(diff < -0.001)
            logr2 = dlog(rmid);
    } 
    while (fabs(diff) >= 0.001 && ++iter < itermax);

    if (iter >= itermax) {
        logrmidtmp = 0.5*(logr2start+logr1start);
        if (logrmid < logrmidtmp) {
            logr1 = logr1start-logstep;
            logr2 = logrmid;
        } else if (logrmid >= logrmidtmp) {
            logr1 = logrmid;
            logr2 = logr2start+logstep;
        }
    
        /* non-linear scale far beyond maximum scale: set flag golinear */
        if (1/pow(10, logr2) > k_nonlin_max) {
            golinear = 1;
            goto after_wint;
        } else {
            goto iterstart;
        }
    }
    
    /* spectral index & curvature at non-linear scale */
    wint(rmid, &sig, &d1, &d2, amp, 0);
    rknl  = 1./rmid;
    rneff = -3-d1;
    rncur = -d2;
        
    after_wint:

    for (i=0; i < numk; i++){
        
        // compute linear power at z = 0 and then add it the growth factors
        linear_power(&k[i], 0., 1, &Pk);
        Delta_L = (Pk*amp*amp)*(k[i]*k[i]*k[i])/(2*M_PI*M_PI);

        if (golinear == 0) {
            halofit(k[i], rneff, rncur, rknl, Delta_L, om_m, om_v, &Delta_NL);
        } else {
            Delta_NL = Delta_L;
        }
        power[i] = (2*M_PI*M_PI)*Delta_NL/(k[i]*k[i]*k[i]);
    }
}
    
/*----------------------------------------------------------------------------*/
void halofit(double rk, double rn, double rncur, double rknl, double plin, 
      double om_m, double om_v, double *pnl)
/*
    Halo model nonlinear fitting formula as described in Appendix C of 
    Smith et al. 2003.
    
    The halofit in Smith et al. 2003 predicts a smaller power
    than latest N-body simulations at small scales.
    Update the following fitting parameters of gam,a,b,c,xmu,xnu,
    alpha & beta from the simulations in Takahashi et al. 2012.
    The improved halofit accurately provide the power spectra for WMAP
    cosmological models with constant w.
*/
{
    double gam,a,b,c,xmu,xnu,alpha,beta,f1,f2,f3;
    double y, ysqr;
    double f1a,f2a,f3a,f1b,f2b,f3b,frac,pq,ph;
   
    gam = 0.1971 - 0.0843*rn + 0.8460*rncur;
    a   = 1.5222 + 2.8553*rn + 2.3706*rn*rn + 0.9903*rn*rn*rn + 
            0.2250*rn*rn*rn*rn - 0.6038*rncur + 0.1749*om_v*(1.+w_lam);
    a = pow(10, a);
    b = pow(10, (-0.5642 + 0.5864*rn + 0.5716*rn*rn - 1.5474*rncur + 0.2279*om_v*(1.+w_lam)));
    c = pow(10, (0.3698 + 2.0404*rn + 0.8161*rn*rn + 0.5869*rncur));
    xmu = 0.;
    xnu = pow(10, (5.2105 + 3.6902*rn));
    alpha = abs(6.0835 + 1.3373*rn - 0.1959*rn*rn - 5.5274*rncur);  
    beta = 2.0379 - 0.7354*rn + 0.3157*rn*rn + 1.2490*rn*rn*rn + 0.3980*rn*rn*rn*rn - 0.1682*rncur;

    if (fabs(1-om_m) > 0.01) { 
        f1a  = pow(om_m, -0.0732);
        f2a  = pow(om_m, -0.1423);
        f3a  = pow(om_m, 0.0725);
        f1b  = pow(om_m, -0.0307);
        f2b  = pow(om_m, -0.0585);
        f3b  = pow(om_m, 0.0743);       
        frac = om_v/(1.-om_m);
        f1   = frac*f1b + (1-frac)*f1a;
        f2   = frac*f2b + (1-frac)*f2a;
        f3   = frac*f3b + (1-frac)*f3a;
    } else {
        f1 = 1.;
        f2 = 1.;
        f3 = 1.;
    }
    
    y = rk/rknl;
    ysqr = y*y;
    ph = a*pow(y, f1*3)/(1 + b*pow(y,f2) + pow(f3*c*y, 3-gam));
    ph = ph/(1 + xmu/y + xnu/ysqr);
    pq = plin*pow(1 + plin, beta)/(1 + plin*alpha)*exp(-y/4.0 - ysqr/8.0);
    *pnl = pq + ph;

    assert(finite(*pnl));
}
/*----------------------------------------------------------------------------*/
/* ============================================================ *
 * Global variables needed in integrand functions for wint	
 * ============================================================ */

double rglob;

/* ============================================================ *
 * Calculates k_NL, n_eff, n_cur				
 * ============================================================ */ 
double knl_integrand(double logk)
{
   double krsqr, k;
   double D2_L;
   
   k = exp(logk);
   krsqr = SQR(k*rglob);
   D2_L = (power_norm*linear_power_unnorm(k)*k*k*k)/(2.*M_PI*M_PI);
   return D2_L*exp(-krsqr);
}

double neff_integrand(double logk)
{
   double krsqr, k;
   double D2_L; 
   
   k = exp(logk);
   krsqr = SQR(k*rglob);
   D2_L = (power_norm*linear_power_unnorm(k)*k*k*k)/(2.*M_PI*M_PI);
   return D2_L*2.*krsqr*exp(-krsqr);
}

double ncur_integrand(double logk)
{
   double krsqr, k;
   double D2_L;
   
   k = exp(logk);
   krsqr = SQR(k*rglob);
   D2_L = (power_norm*linear_power_unnorm(k)*k*k*k)/(2.*M_PI*M_PI);
   return D2_L*4.*krsqr*(1.-krsqr)*exp(-krsqr);
}

/*-----------------------------------------------------------------------------*/
void wint(double r, double *sig, double *d1, double *d2, double amp, int onlysig)
/*
    The subroutine wint, finds the effective spectral quantities
    rknl, rneff & rncur. This it does by calculating the radius of 
    the Gaussian filter at which the variance is unity = rknl.
    rneff is defined as the first derivative of the variance, calculated 
    at the nonlinear wavenumber and similarly the rncur is the second
    derivative at the nonlinear wavenumber.
*/
{
   const double kmin = 1.e-2;
   double kmax, logkmin, logkmax, s1, s2, s3;

   /* choose upper integration limit to where filter function dropped
    substantially */
   kmax  = sqrt(5.*log(10.))/r;
   if (kmax < 8000.) kmax = 8000.;

   logkmin = log(kmin);
   logkmax = log(kmax);
   rglob = r;

   if (onlysig==1) {
      s1   = qromb1(knl_integrand, logkmin, logkmax);
      *sig = amp*sqrt(s1);
   } else s1 = SQR(1/amp);   /* sigma = 1 */

   if (onlysig==0) {
      s2  = qromb1(neff_integrand, logkmin, logkmax);
      s3  = qromb1(ncur_integrand, logkmin, logkmax);
      *d1 = -s2/s1;
      *d2 = -SQR(*d1) - s3/s1;
   }
}

/*----------------------------------------------------------------------------*/
double dlog(double x)
{
   return log(x)/log(10.0);
}

/*----------------------------------------------------------------------------*/
/* Numerical Recipes Functions */
/*----------------------------------------------------------------------------*/
#define EPS 1.0e-6
#define JMAX 40
#define JMAXP (JMAX+1)
#define K 5

double qromb1(double (*func)(double), double a, double b)
{
	double ss,dss;
	double s[JMAXP],h[JMAXP+1];
	int j;

	h[1]=1.0;
	for (j=1;j<=JMAX;j++) {
		s[j]=trapzd1(func,a,b,j);
		if (j >= K) {
			polint(&h[j-K],&s[j-K],K,0.0,&ss,&dss);
			if (fabs(dss) <= EPS*fabs(ss)) return ss;
		}
		h[j+1]=0.25*h[j];
	}
	error("Too many steps in routine qromb");
	return 0.0;
}
#undef EPS
#undef JMAX
#undef JMAXP
#undef K

/*----------------------------------------------------------------------------*/
#define FUNC(x) ((*func)(x))

double trapzd1(double (*func)(double), double a, double b, int n)
{
	double x,tnm,sum,del;
	static double s;
	int it,j;

	if (n == 1) {
		return (s=0.5*(b-a)*(FUNC(a)+FUNC(b)));
	} else {
		for (it=1,j=1;j<n-1;j++) it <<= 1;
		tnm=it;
		del=(b-a)/tnm;
		x=a+0.5*del;
		for (sum=0.0,j=1;j<=it;j++,x+=del) {
		   sum += FUNC(x);
		}
		s=0.5*(s+(b-a)*sum/tnm);
		return s;
	}
}
#undef FUNC

/*----------------------------------------------------------------------------*/
void polint(double xa[], double ya[], int n, double x, double *y, double *dy)
{
	int i,m,ns=1;
	double den,dif,dift,ho,hp,w;
	double *c,*d;

	dif=fabs(x-xa[1]);
	c=vector(1,n);
	d=vector(1,n);
	for (i=1;i<=n;i++) {
		if ( (dift=fabs(x-xa[i])) < dif) {
			ns=i;
			dif=dift;
		}
		c[i]=ya[i];
		d[i]=ya[i];
	}
	*y=ya[ns--];
	for (m=1;m<n;m++) {
		for (i=1;i<=n-m;i++) {
			ho=xa[i]-x;
			hp=xa[i+m]-x;
			w=c[i+1]-d[i];
			if ( (den=ho-hp) == 0.0)
			  error("Error in routine polint");
			den=w/den;
			d[i]=hp*den;
			c[i]=ho*den;
		}
		*y += (*dy=(2*ns < (n-m) ? c[ns+1] : d[ns--]));
	}
	free_vector(d,1,n);
	free_vector(c,1,n);
}

/*----------------------------------------------------------------------------*/
void error(const char *s)
{
   fprintf(stderr, "error: ");
   fprintf(stderr, "%s", s);
   fprintf(stderr,"\n");
   exit(1);
}

/*----------------------------------------------------------------------------*/
double *vector(long nl, long nh)
/* allocate a double vector with subscript range v[nl..nh] */
{
	double *v;

	v=(double *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(double)));
	if (!v) error("allocation failure in vector()");
	return v-nl+NR_END;
}

/*----------------------------------------------------------------------------*/
void free_vector(double *v, long nl, long nh)
/* free a double vector allocated with vector() */
{
	free((FREE_ARG) (v+nl-NR_END));
}


