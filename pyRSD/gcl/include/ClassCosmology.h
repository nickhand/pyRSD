/*------------------------------------------------------------------------------
Description:
    class ClassCosmology : encapsulation of class calls


Author List:
    Stephane Plaszczynski (plaszczy@lal.in2p3.fr)
    Nick Hand (nhand@berkeley.edu)

History (add to end):
    creation:   ven. nov. 4 11:02:20 CET 2011 
    modified: nov 21 14
-----------------------------------------------------------------------------*/
#ifndef CLASSCOSMOLOGY_H
#define CLASSCOSMOLOGY_H


#include "ClassParams.h"
#include "Engine.h"
#include "Common.h"

/*----------------------------------------------------------------------------*/
/* class to serve as the engine for CLASS */
/*----------------------------------------------------------------------------*/
class ClassCosmology : public Engine
{
public:
  
    ClassCosmology();
    // construct from a ClassParams object
    ClassCosmology(const ClassParams& pars, const std::string& precision_file = "");
    // construct directly from a parameter file
    ClassCosmology(const std::string& param_file, const std::string& precision_file = "");
  
    void Initialize(const ClassParams& pars, const std::string& precision_file);
    
    // destructor
    ~ClassCosmology();

    // the functions to return the various spectra 
    int GetCls(const std::vector<unsigned>& lVec, parray& cltt, parray& clte, parray& clee, parray& clbb);
    int GetLensing(const std::vector<unsigned>& lVec, parray& clphiphi, parray& cltphi, parray& clephi);

    // linear/nonlinear matter power spectra
    int GetPklin(double z, const parray& k, parray& Pk);
    int GetPknl(double z, const parray& k, parray& Pk);

    // get the total density transfer function
    int GetTk(double z, const parray& k, parray& Tk);

    // parameter accessors

    inline int l_max_scalars() const { return _lmax; }
    

    
    /*------------------------------------------------------------------------*/
    /* present day quantities */
    /*------------------------------------------------------------------------*/
    // the present-day Hubble constant in units of km/s/Mpc
    inline double H0() const { return ba.H0*Constants::c_light/(Constants::km/Constants::second); }
    // the dimensionless Hubble constant
    inline double h() const { return ba.h; }
    // CMB temperature today in Kelvin
    inline double Tcmb() const { return ba.T_cmb; }
    // present baryon density parameter (\Omega_b)
    inline double Omega0_b() const { return ba.Omega0_b; }
    // present cold dark matter density fraction (\Omega_cdm)
    inline double Omega0_cdm() const { return ba.Omega0_cdm; }
    // present ultra-relativistic neutrino density fraction (\Omega_{\nu, ur})
    inline double Omega0_ur() const { return ba.Omega0_ur; }
    // present non-relativistic density fraction (\Omega_b + \Omega_cdm + \Omega_{\nu nr}) 
    inline double Omega0_m() const { return Omega0_m_; }
    // relativistic density fraction (\Omega_{\gamma} + \Omega_{\nu r})
    inline double Omega0_r() const { return Omega0_r_; }
    // photon density parameter today
    inline double Omega0_g() const { return ba.Omega0_g; }
    // cosmological constant density fraction
    inline double Omega0_lambda() const { return ba.Omega0_lambda; }
    // curvature density fraction
    inline double Omega0_k() const { return ba.Omega0_k; }
    // current fluid equation of state parameter
    inline double w0_fld() const { return ba.w0_fld; }
    // fluid equation of state parameter derivative
    inline double wa_fld() const { return ba.wa_fld; }
    
    // the spectral index of the primordial power spectrum
    inline double n_s() const { return pm.n_s; }
    // pivot scale in Mpc-1 
    inline double k_pivot() const { return pm.k_pivot; }
    // scalar amplitude = curvature power spectrum at pivot scale
    inline double A_s() const { return pm.A_s; }
    // convenience function returns log (1e10*A_s)
    inline double ln_1e10_A_s() const { return log(1e10*A_s()); }
    // convenience function to return sigma8 at z = 0
    inline double sigma8() const { return sp.sigma8; }
    // maximum k value computed in h/Mpc
    inline double k_max() const { return exp(sp.ln_k[sp.ln_k_size-1])/h(); }
    
    // baryon drag redshift
    inline double z_drag() const { return th.z_d; }
    // comoving sound horizon at baryon drag
    inline double rs_drag() const { return th.rs_d; } 
    // reionization optical depth
    inline double tau_reio() const { return th.tau_reio; }
    // reionization redshift
    inline double z_reio() const { return th.z_reio; }
    
    /*------------------------------------------------------------------------*/
    /* background quantities as a function of z */
    /*------------------------------------------------------------------------*/
    // the growth rate f(z) (unitless)
    inline double f_z(double z) const { return BackgroundValue(z, ba.index_bg_f); }
    
    // Hubble constant H(z) (km/s/Mpc)
    double H_z(double z) const { return (z != 0.) ? BackgroundValue(z, ba.index_bg_H)*Constants::c_light/(Constants::km/Constants::second) : H0(); }
        
    // angular diameter distance (Mpc)
    double Da_z(double z) const { return BackgroundValue(z, ba.index_bg_ang_distance); }
    
    // growth function D(z) / D(0) (normalized to unity at z = 0)
    double D_z(double z) const { return (z != 0.) ? BackgroundValue(z, ba.index_bg_D) / BackgroundValue(0., ba.index_bg_D) : 1.; }
    
    // sigma8 (z) as derived from the scalar amplitude
    // may not be equal to the desired sigma8
    double Sigma8_z(double z) const;

    // print content of file_content
    void PrintFC() const;
    
protected:
  
    // the main class structures
    // make them mutable such that they can be changed for different
    // redshifts, etc
    mutable struct file_content fc;
    mutable struct precision pr;        /* for precision parameters */
    mutable struct background ba;       /* for cosmological background */
    mutable struct thermo th;           /* for thermodynamics */
    mutable struct perturbs pt;         /* for source functions */
    mutable struct transfers tr;        /* for transfer functions */
    mutable struct primordial pm;       /* for primordial spectra */
    mutable struct spectra sp;          /* for output spectra */
    mutable struct nonlinear nl;        /* for non-linear spectra */
    mutable struct lensing le;          /* for lensed spectra */
    mutable struct output op;           /* for output files */

    ErrorMsg _errmsg;                   /* for error messages */
    double * cl;

    // helpers
    bool dofree;
    int FreeStructs();

    // call once per model
    int Run();

    int class_main(struct file_content *pfc,
    	           struct precision * ppr,
    	           struct background * pba,
    	           struct thermo * pth,
    	           struct perturbs * ppt,
    	           struct transfers * ptr,
    	           struct primordial * ppm,
    	           struct spectra * psp,
    	           struct nonlinear * pnl,
    	           struct lensing * ple,
    	           struct output * pop,
    	           ErrorMsg errmsg);
    
    // parameter names
    std::vector<std::string> parNames;
    
    // nonlinear vs linear P(k)
    enum Pktype {Pk_linear=0, Pk_nonlinear};
    int GetPk(double z, const parray& k, parray& Pk, Pktype method = Pk_linear);
    double GetCl(Engine::cltype t, const long &l); 
        
    // functions for returning cosmological quantities
    double BackgroundValue(double z, int index) const;
    
    // helper function to find correctfile name
    const std::string FindFilename(const std::string& file_name);
    
    // omega0_m and omega0_r are constants
    double Omega0_m_, Omega0_r_;
    
};


;
#endif

