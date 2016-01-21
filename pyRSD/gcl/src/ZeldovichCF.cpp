#include "ZeldovichCF.h"
#include "CorrelationFunction.h"

using namespace Common;

static const int NUM_PTS = 1024;


ZeldovichCF::ZeldovichCF(const Cosmology& C_, double z_, double kmin_, double kmax_) 
                          : C(C_), z(z_), Pzel(C_, z_, true), Plin(C_, z_),
                            sigma8_z(C_.Sigma8_z(z_)), kmin(kmin_), kmax(kmax_)
{    
    // initialize the k, Pzel grid arrays
    k_grid = parray::logspace(kmin, kmax, NUM_PTS);
    Pzel_grid = Pzel(k_grid);
}

double ZeldovichCF::Evaluate(double r, double smoothing) const 
{
    parray rarr(1);
    rarr[0] = r;
    return pk_to_xi(0, k_grid, Pzel_grid, rarr, smoothing)[0];
}

parray ZeldovichCF::EvaluateMany(const parray& r, double smoothing) const 
{
    return  pk_to_xi(0, k_grid, Pzel_grid, r, smoothing);
}

void ZeldovichCF::SetSigma8AtZ(double new_sigma8_z) 
{     
    
    // store the sigma8_z
    sigma8_z = new_sigma8_z;
    
    // set the Plin
    Plin.SetSigma8AtZ(sigma8_z);
    
    // set the Pzel
    Pzel.SetSigma8AtZ(sigma8_z);
    Pzel_grid = Pzel(k_grid);
}
