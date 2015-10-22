#ifndef ZELDOVICH_PS_H
#define ZELDOVICH_PS_H

#include "Cosmology.h"
#include "PowerSpectrum.h"
#include "MyFFTLog.h"

/*------------------------------------------------------------------------------
    ZeldovichPS
    ---------
    Power spectra in the Zel'dovich approximation for P00 and P01
------------------------------------------------------------------------------*/

// base class for Zel'dovich PS
class ZeldovichPS  {
public:
       
    // constructors
    ZeldovichPS(const Cosmology& C, double z);
    ZeldovichPS(const Cosmology& C, double sigma8_z, double sigma_sq, const parray& X, const parray& Y);
    virtual ~ZeldovichPS();
    
    // evaluate
    virtual double Evaluate(double k) const; 
    double operator()(double k) const { return Evaluate(k); }
    
    parray EvaluateMany(const parray& k) const;
    parray operator()(const parray& k) const { return EvaluateMany(k); }
        
    // get references to various attributes
    const Cosmology& GetCosmology() const { return C; }
    const double& GetSigma8AtZ() const { return sigma8_z; }
    parray GetXZel() const { return XX; }
    parray GetYZel() const { return YY; }
    const double& GetSigmaSq() const { return sigma_sq; }
             
    // set the redshift and sigma8
    // convenience tracking so we don't recompute XX, YY, sigma_sq if sigma8, z change
    void SetSigma8AtZ(double sigma8_z);
         
protected:
    
    // the cosmology
    const Cosmology& C;
    
    // keep track of redshift, sigma8 for easy scaling
    double sigma8_z;
    
    // the integrals needed for the FFTLog integral
    double sigma_sq;
    parray r, XX, YY; 
    
    void InitializeR();
    double fftlog_compute(double k, const double factor = 1) const;
    virtual void Fprim(dcomplex[], const double[], double) const;
    virtual void Fsec(dcomplex[], const double[], double, double) const;
    
};


class ZeldovichP00 : public ZeldovichPS {
public:
    
    ZeldovichP00(const Cosmology& C, double z);
    ZeldovichP00(const ZeldovichPS& ZelPS);
    
    double Evaluate(double k) const; 

private:
    
    void Fprim(dcomplex a[], const double r[], double k) const;
    void Fsec(dcomplex a[], const double r[], double k, double n) const;   
};

class ZeldovichP01 : public ZeldovichPS {
public:
    
    ZeldovichP01(const Cosmology& C, double z);
    ZeldovichP01(const ZeldovichPS& ZelPS);
    
    double Evaluate(double k) const; 
    

private:
        
    void Fprim(dcomplex a[], const double r[], double k) const;
    void Fsec(dcomplex a[], const double r[], double k, double n) const;
};

#endif // ZELDOVICH_PS_H