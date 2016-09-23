#include "ZeldovichCF.h"
#include "CorrelationFunction.h"

using namespace Common;

static const int NUM_PTS = 1024;


ZeldovichCF::ZeldovichCF(const Cosmology& C_, double z_, double kmin_, double kmax_) 
                          : Pzel(C_, z_, true), sigma8_z(Pzel.GetSigma8AtZ()), kmin(kmin_), kmax(kmax_)
{    
    // initialize the k, Pzel grid arrays
    k_grid = parray::logspace(kmin, kmax, NUM_PTS);
    Pzel_grid = Pzel(k_grid);
}

ZeldovichCF::ZeldovichCF(const ZeldovichPS& ZelPS, double kmin_, double kmax_) 
                          : Pzel(ZelPS), sigma8_z(Pzel.GetSigma8AtZ()), kmin(kmin_), kmax(kmax_)
{    
    // initialize the k, Pzel grid arrays
    k_grid = parray::logspace(kmin, kmax, NUM_PTS);
    Pzel.SetLowKApprox();
    Pzel_grid = Pzel(k_grid);
}


double ZeldovichCF::Evaluate(double r, double smoothing) 
{
    parray rarr(1);
    rarr[0] = r;
    
    if (sigma8_z != Pzel.GetSigma8AtZ()) {
        Pzel.SetSigma8AtZ(sigma8_z);
        Pzel_grid = Pzel(k_grid);
    }
    
    return pk_to_xi(0, k_grid, Pzel_grid, rarr, smoothing)[0];
}

parray ZeldovichCF::EvaluateMany(const parray& r, double smoothing) 
{
    
    if (sigma8_z != Pzel.GetSigma8AtZ()) {
        Pzel.SetSigma8AtZ(sigma8_z);
        Pzel_grid = Pzel(k_grid);
    }
    
    return  pk_to_xi(0, k_grid, Pzel_grid, r, smoothing);
}

void ZeldovichCF::SetSigma8AtZ(double new_sigma8_z) 
{     
    // store the sigma8_z
    sigma8_z = new_sigma8_z;
}
