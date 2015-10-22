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
    
    double xi;
    ComputeXiLM(ell, 2, P, 1, &r, &xi, Nk, kmin, kmax);
    return pow(-1, ell/2)*PowerPolePrefactor(ell)*xi;
}

parray Kaiser::Xi_ell(int ell, const parray& r, int Nk, double kmin, double kmax) const {
    int Nr = (int) r.size();
    parray xi(Nr);
    ComputeXiLM(ell, 2, P, Nr, &r[0], &xi[0], Nk, kmin, kmax);
    return pow(-1, ell/2)*PowerPolePrefactor(ell)*xi;
}

