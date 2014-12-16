#ifndef ZELDOVICH_PS_H
#define ZELDOVICH_PS_H

#include "Cosmology.h"
#include "PowerSpectrum.h"
#include "FFTLog.h"

/*------------------------------------------------------------------------------
    ZeldovichPS
    ---------
    Power spectra in the Zel'dovich approximation for P00 and P01
------------------------------------------------------------------------------*/

// base class for Zel'dovich PS
class ZeldovichPS  {
public:
       
    ZeldovichPS(const PowerSpectrum& P_L);
    virtual ~ZeldovichPS();
    
    virtual double Evaluate(double k) const; 
    double operator()(double k) const { return Evaluate(k); }
    
    parray EvaluateMany(const parray& k) const;
    parray operator()(const parray& k) const { return EvaluateMany(k); }
        
    // get references to various attributes
    const PowerSpectrum& GetLinearPS() const { return P_L; }
    const double& GetRedshift() const { return z; }
    const double& GetSigma8() const { return sigma8; }
    const Cosmology& GetCosmology() const { return P_L.GetCosmology(); }
 
    // set the redshift and sigma8
    // convenience tracking so we don't recompute XX, YY, sigma_sq if sigma8, z change
    void SetRedshift(double z); 
    void SetSigma8(double sigma8);
     
protected:

    // power spectrum reference
    const PowerSpectrum& P_L;
    
    // keep track of redshift, sigma8 for easy scaling
    double z, sigma8;
    double nc, dlogr, logrc;
    
    // the integrals needed for the FFTLog integral
    double sigma_sq;
    parray r, XX, YY; 
    
    double fftlog_compute(double k, const double factor = 1) const;
    virtual void Fprim(parray&, const parray&, double) const;
    virtual void Fsec(parray&, const parray&, double, double) const;
    
};


class ZeldovichP00 : public ZeldovichPS {
public:
    
    ZeldovichP00(const PowerSpectrum& P_L);
    ZeldovichP00(const ZeldovichPS& ZelPS);
    
    double Evaluate(double k) const; 

private:
    
    void Fprim(parray& a, const parray& r, double k) const ;
    void Fsec(parray& a, const parray& r, double k, double n) const;    
};

class ZeldovichP01 : public ZeldovichPS {
public:
    
    ZeldovichP01(const PowerSpectrum& P_L);
    ZeldovichP01(const ZeldovichPS& ZelPS);
    
    double Evaluate(double k) const; 
    

private:
        
    void Fprim(parray& a, const parray& r, double k) const ;
    void Fsec(parray& a, const parray& r, double k, double n) const;    
};

#endif // ZELDOVICH_PS_H