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
    ZeldovichCF(const Cosmology& C, double z, double kmin = 1e-5, double kmax = 1.0);

    double Evaluate(double r, double smoothing=0.) const;
    double operator()(double r, double smoothing=0.) const { return Evaluate(r, smoothing); }

    parray EvaluateMany(const parray& r, double smoothing=0.) const;
    parray operator()(const parray& r, double smoothing=0.) const { return EvaluateMany(r, smoothing); }


private:
    
    const Cosmology& C;
    double z;
    const ZeldovichP00 Pzel;
    const LinearPS Plin;
    double sigma8_z; 
    
    double kmin, kmax;
    parray k_grid, Pzel_grid;
    
};


#endif // ZELDOVICH_CF_H