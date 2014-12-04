#include "Kaiser.h"
#include "CorrelationFunction.h"
#include "Cosmology.h"
#include "LinearPS.h"

using namespace Common;

Kaiser::Kaiser(const Cosmology& C, double z) : P(LinearPS(C, z)), f(C.f_z(z)) {}


Kaiser::Kaiser(const PowerSpectrum& P_) : P(P_), f(P.GetCosmology().f_z(P.GetRedshift())) {}

double Kaiser::P_s(double k, double mu) const {
    return pow2(1 + f*mu*mu) * P(k);
}

double Kaiser::PMultipole(int ell, double k) const {
    double prefactor;
    if(ell == 0)
        prefactor = 1. + (2./3.)*f + (1./5.)*f*f;
    else if(ell == 2)
        prefactor = (4./3.)*f + (4./7.)*f*f;
    else if(ell == 4)
        prefactor = (8./35.)*f*f;
    else
        prefactor = 0.;

    return prefactor * P(k);
}

parray Kaiser::PMultipole(int ell, const parray& k) const {
    int N = (int) k.size();
    parray pell(N);
    #pragma omp parallel for
    for(int j = 0; j < N; j++)
        pell[j] = PMultipole(ell, k[j]);
    return pell;
}

double Kaiser::XiMultipole(int ell, double r, int Nk, double kmin, double kmax) const {
    double xi;
    ComputeXiLM(ell, 2, P, 1, &r, &xi, Nk, kmin, kmax);
    return xi;
}

parray Kaiser::XiMultipole(int ell, const parray& r, int Nk, double kmin, double kmax) const {
    int Nr = (int) r.size();
    parray xi(Nr);
    ComputeXiLM(ell, 2, P, Nr, &r[0], &xi[0], Nk, kmin, kmax);
    return xi;
}

