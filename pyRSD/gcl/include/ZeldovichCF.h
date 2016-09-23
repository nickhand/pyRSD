#ifndef ZELDOVICH_CF_H
#define ZELDOVICH_CF_H

#include "ZeldovichPS.h"
#include "LinearPS.h"

/*------------------------------------------------------------------------------
    ZeldovichCF
    ---------
    Zel'dovich correlation function which Fourier transforms ZeldovichP00
------------------------------------------------------------------------------*/

class ZeldovichCF  {
public:
       
public:
    ZeldovichCF(const Cosmology& C, double z, double kmin = 1e-5, double kmax = 10.0);
    ZeldovichCF(const ZeldovichPS& ZelPS, double kmin = 1e-5, double kmax = 10.0);

    double Evaluate(double r, double smoothing=0.);
    double operator()(double r, double smoothing=0.) { return Evaluate(r, smoothing); }

    parray EvaluateMany(const parray& r, double smoothing=0.);
    parray operator()(const parray& r, double smoothing=0.) { return EvaluateMany(r, smoothing); }

    void SetSigma8AtZ(double sigma8_z);

private:
    
    ZeldovichP00 Pzel;
    double sigma8_z; 
    
    double kmin, kmax;
    parray k_grid, Pzel_grid;
    
};


#endif // ZELDOVICH_CF_H