#include "Kaiser.h"
#include "CorrelationFunction.h"
#include "Cosmology.h"
#include "LinearPS.h"

using namespace Common;

Kaiser::Kaiser(const PowerSpectrum& P_, double f_, double b1_)
    : P(P_), f(f_), b1(b1_)
{
}

double Kaiser::P_s(double k, double mu) const {
    return pow2(1 + f*mu*mu) * P(k);
}

double Kaiser::PowerPolePrefactor(int ell) const {
    
    double prefactor(0.); 
    double beta = f/b1;
    if (ell == 0)
        prefactor = 1. + (2./3.)*beta + (1./5.)*pow2(beta);
    else if(ell == 2)
        prefactor = (4./3.)*beta + (4./7.)*pow2(beta);
    else if(ell == 4)
        prefactor = (8./35.)*pow2(beta);

    return prefactor*pow2(b1);
  
}

double Kaiser::P_ell(int ell, double k) const {
    return PowerPolePrefactor(ell) * P(k);
}

parray Kaiser::P_ell(int ell, const parray& k) const {
    return PowerPolePrefactor(ell) * P(k);
}

double Kaiser::Xi_ell(int ell, double r, int Nk, double kmin, double kmax) const {
  
    parray r_(1); r_[0] = r; 
    parray k = parray::logspace(kmin, kmax, Nk);
    parray xi = ComputeXiLM(ell, 2, k, P(k), r_, 0.);
    return pow(-1, ell/2)*PowerPolePrefactor(ell)*xi[0];
}

parray Kaiser::Xi_ell(int ell, const parray& r, int Nk, double kmin, double kmax) const {
    
    parray k = parray::logspace(kmin, kmax, Nk);
    parray xi = ComputeXiLM(ell, 2, k, P(k), r, 0.);
    return pow(-1, ell/2)*PowerPolePrefactor(ell)*xi;
}

