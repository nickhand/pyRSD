#ifndef ZELDOVICH_PS_H
#define ZELDOVICH_PS_H

#include "Cosmology.h"
#include "PowerSpectrum.h"
#include "FortranFFTLog.h"
#include "LinearPS.h"

/*------------------------------------------------------------------------------
    ZeldovichPS
    ---------
    Power spectra in the Zel'dovich approximation for P00 and P01
------------------------------------------------------------------------------*/

// base class for Zel'dovich PS
class ZeldovichPS  {
public:
       
    // constructors
    ZeldovichPS(const Cosmology& C, double z, bool approx_lowk=false);
    ZeldovichPS(const Cosmology& C, bool approx_lowk, double sigma8_z, double k0_low,
                  double sigma_sq, const parray& X0, const parray& XX, const parray& YY);
    virtual ~ZeldovichPS();
    
    // evaluate
    virtual double Evaluate(double k) const; 
    double operator()(double k) const { return Evaluate(k); }
    
    parray EvaluateMany(const parray& k) const;
    parray operator()(const parray& k) const { return EvaluateMany(k); }
        
    // functions for doing low-k approximation
    void SetLowKApprox(bool approx_lowk_=true) { approx_lowk=approx_lowk_; }
    void SetLowKTransition(double k0) { k0_low = k0; }  
    virtual double LowKApprox(double k) const { return Evaluate(k); }  
        
    // get references to various attributes
    const Cosmology& GetCosmology() const { return C; }
    const double& GetSigma8AtZ() const { return sigma8_z; }
    const bool& GetApproxLowKFlag() const { return approx_lowk; }
    const double& GetK0Low() const { return k0_low; }
    parray GetXZel() const { return XX; }
    parray GetYZel() const { return YY; }
    parray GetX0Zel() const { return X0; }
    const double& GetSigmaSq() const { return sigma_sq; }
             
    // set sigma8(z)
    // convenience tracking so we don't recompute XX, YY, sigma_sq if sigma8, z change
    void SetSigma8AtZ(double sigma8_z);
         
protected:
    
    // the cosmology
    const Cosmology& C;
    
    // keep track of redshift, sigma8 for easy scaling
    double sigma8_z;
    
    // linear power spectrum
    LinearPS Plin;
    
    // low k approximation 
    bool approx_lowk;
    double k0_low;
    
    
    double nc, dlogr, logrc;
    
    // the integrals needed for the FFTLog integral
    double sigma_sq;
    parray r, X0, XX, YY; 
    
    void InitializeR();
    double fftlog_compute(double k, const double factor = 1) const;
    virtual void Fprim(parray&, const parray&, double) const;
    virtual void Fsec(parray&, const parray&, double, double) const;
    
};


class ZeldovichP00 : public ZeldovichPS {
public:
    
    ZeldovichP00(const Cosmology& C, double z, bool approx_lowk=false);
    ZeldovichP00(const ZeldovichPS& ZelPS);
    
    double Evaluate(double k) const; 
    double LowKApprox(double k) const;

private:
    
    void Fprim(parray& a, const parray& r, double k) const;
    void Fsec(parray& a, const parray& r, double k, double n) const;   
};

class ZeldovichP01 : public ZeldovichPS {
public:
    
    ZeldovichP01(const Cosmology& C, double z, bool approx_lowk=false);
    ZeldovichP01(const ZeldovichPS& ZelPS);
    
    double Evaluate(double k) const; 
    double LowKApprox(double k) const;

private:
        
    void Fprim(parray& a, const parray& r, double k) const;
    void Fsec(parray& a, const parray& r, double k, double n) const;
};

class ZeldovichP11 : public ZeldovichPS {
public:
    
    ZeldovichP11(const Cosmology& C, double z, bool approx_lowk=false);
    ZeldovichP11(const ZeldovichPS& ZelPS);
    
    double Evaluate(double k) const;
    double LowKApprox(double k) const; 
    

private:
        
    void Fprim(parray& a, const parray& r, double k) const;
    void Fsec(parray& a, const parray& r, double k, double n) const;
};

#endif // ZELDOVICH_PS_H