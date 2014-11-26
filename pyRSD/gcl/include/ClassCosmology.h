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

// CLASS
#include "class.h"

#include "Engine.h"
#include "Common.h"

// the std c++ libraries
#include <string>
#include <utility>
#include <stdexcept>
#include <map>
#include <iostream>

using namespace std;
using namespace Constants;

// general utility to convert safely numerical types to string
template<typename T> string str(const T &x);

//specializations
template<> string str(const float &x);
template<> string str(const double &x);
template<> string str(const bool &x); // translate bool to "yes" or "no"
template<> string str(const string &x);
string str(const char* x);

/*----------------------------------------------------------------------------*/
/* class to encapsulate CLASS parameters from any type (numerical or string)  */
/*----------------------------------------------------------------------------*/
class ClassParams {

public:
    
    typedef map<string, string> param_vector;
    typedef param_vector::iterator iterator;
    typedef param_vector::const_iterator const_iterator;
    
    ClassParams(){};
    ClassParams(const ClassParams& o) : pars(o.pars){};
    ClassParams(const string& param_file){
        
        // for error messages
        ErrorMsg _errmsg; 
        
        // initialize an empty file_content to read the params file
        struct file_content fc_input;
        fc_input.size = 0;
        fc_input.filename=new char[1];
        
        // read the param file
        if (parser_read_file(const_cast<char*>(param_file.c_str()), &fc_input, _errmsg) == _FAILURE_){
            throw invalid_argument(_errmsg);
        }
        
        // set the pars
        for (int i=0; i < fc_input.size; i++) {
            this->Update(fc_input.name[i], fc_input.value[i]);
        }
        
        // free the input
        parser_free(&fc_input);
  }

  // use this to add a CLASS variable
  template<typename T> 
  unsigned Update(const string& key, const T& val){ pars[key] = str(val); return pars.size(); }
  
  void Print() {
     for (const_iterator iter = pars.begin(); iter != pars.end(); iter++)       
         cout << iter->first << " = " << iter->second << endl;
  }
  
  // accesors
  inline unsigned size() const {return pars.size();}
  inline const string& value(const string& key) const {return pars.at(key);}
  
  // iterate over the pars variable
  const_iterator begin() const { return pars.begin(); }
  const_iterator end() const { return pars.end(); }
  
  // overload the [] operator to return const reference
  const string& operator[](const string& key) const { return this->value(key); }

private:
    param_vector pars;
};

/*----------------------------------------------------------------------------*/
/* class to serve as the engine for CLASS */
/*----------------------------------------------------------------------------*/
class ClassCosmology : public Engine
{
public:
  
    ClassCosmology();
    // construct from a ClassParams object
    ClassCosmology(const ClassParams& pars, const string& precision_file = "");
    // construct directly from a parameter file
    ClassCosmology(const string& param_file, const string& precision_file = "");
  
    void Initialize(const ClassParams& pars, const string& precision_file);
    
    // destructor
    ~ClassCosmology();

    // the functions to return the various spectra 
    int GetCls(const vector<unsigned>& lVec, // input 
	            parray& cltt, 
	            parray& clte, 
	            parray& clee, 
	            parray& clbb);

  
    int GetLensing(const vector<unsigned>& lVec, //input 
	                parray& clphiphi, 
	                parray& cltphi, 
	                parray& clephi);

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
    inline double H0() const { return ba.H0*c_light/(km/second); }
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
    inline double Omega0_lambda() const { return ba.Omega0_g; }
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
    inline double ln_1e10_A_s() const { return log(1e10*pm.A_s); }
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
    inline double f_z(double z) { return BackgroundValue(z, ba.index_bg_f); }
    
    // Hubble constant H(z) (km/s/Mpc)
    double H_z(double z) { return (z != 0.) ? BackgroundValue(z, ba.index_bg_H)*c_light/(km/second) : H0(); }
        
    // angular diameter distance (Mpc)
    double Da_z(double z) { return BackgroundValue(z, ba.index_bg_ang_distance); }
    
    // growth function D(z) / D(0) (normalized to unity at z = 0)
    double D_z(double z) { return (z != 0.) ? BackgroundValue(z, ba.index_bg_D) / BackgroundValue(0., ba.index_bg_D) : 1.; }
    
    // sigma8 (z) as derived from the scalar amplitude
    // may not be equal to the desired sigma8
    double Sigma8_z(double z);

    // print content of file_content
    void PrintFC();
    
protected:
  
    // the main class structures
    struct file_content fc;
    struct precision pr;        /* for precision parameters */
    struct background ba;       /* for cosmological background */
    struct thermo th;           /* for thermodynamics */
    struct perturbs pt;         /* for source functions */
    struct transfers tr;        /* for transfer functions */
    struct primordial pm;       /* for primordial spectra */
    struct spectra sp;          /* for output spectra */
    struct nonlinear nl;        /* for non-linear spectra */
    struct lensing le;          /* for lensed spectra */
    struct output op;           /* for output files */

    ErrorMsg _errmsg;            /* for error messages */
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
    vector<string> parNames;
    
    // nonlinear vs linear P(k)
    enum Pktype {Pk_linear=0, Pk_nonlinear};
    int GetPk(double z, const parray& k, parray& Pk, Pktype method = Pk_linear);
    double GetCl(Engine::cltype t, const long &l); 
        
    // functions for returning cosmological quantities
    double BackgroundValue(double z, int index);
    
    // helper function to find correctfile name
    const string FindFilename(const string& file_name);
    
    // omega0_m and omega0_r are constants
    double Omega0_m_, Omega0_r_;
    
};


;
#endif

