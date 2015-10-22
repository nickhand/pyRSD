#include "CorrelationFunction.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <functional>

#include "PowerSpectrum.h"
#include "Quadrature.h"
#include "SpecialFunctions.h"
#include "Spline.h"

using std::bind;
using std::cref;
using namespace std::placeholders;
using namespace Common;

parray SmoothedXiMultipole(Spline P, int l, const parray& r, int Nk, double kmin, double kmax, double smoothing)
{
    
    int Nr = (int) r.size();
    parray xi(Nr);
    int m = 2;
    
    assert(Nk > 0 && (Nk % 2) == 0);
    const double dk = (kmax - kmin)/Nk;

    /* Choose appropriate spherical Bessel function */
    double (*sj)(double x);
    if (l == 0)      sj = SphericalBesselJ0;
    else if (l == 1) sj = SphericalBesselJ1;
    else if (l == 2) sj = SphericalBesselJ2;
    else if (l == 3) sj = SphericalBesselJ3;
    else if (l == 4) sj = SphericalBesselJ4;
    else if (l == 6) sj = SphericalBesselJ6;
    else if (l == 8) sj = SphericalBesselJ8;
    else {
        error("SmoothedXiMultipole: l = %d not supported\n", l);
    }

    parray k = parray::linspace(kmin, kmax, Nk+1);
    parray mult(Nk+1);

    #pragma omp parallel for
    for(int j = 0; j <= Nk; j++) {
        /* Multiplicative factor for Simpson's rule: either 1, 2, or 4 */
        mult[j] = 2 + 2*(j % 2) - (j == 0) - (j == Nk);
        /* All other purely k-dependent factors */
        mult[j] *= exp(-pow2(k[j]*smoothing)) * P(k[j]) * pow(k[j], m) * (dk/3) / (2*M_PI*M_PI);
    }

    /* Integrate $P(k) k^m j_l(kr) dk$ over the interval $[kmin,kmax]$ using Simpson's rule */
    #pragma omp parallel for
    for(int i = 0; i < Nr; i++) {
        xi[i] = 0;
        for(int j = 0; j <= Nk; j++)
            xi[i] += mult[j] * sj(k[j]*r[i]);
    }
    return xi;
}


parray ComputeDiscreteXiLM(int l, int m, const parray& k, const parray& pk, const parray& r, double smoothing)
{
    // array sizes
    int Nk(pk.size()), Nr(r.size());
    parray xi(Nr);
    
    // choose appropriate spherical Bessel function
    double (*sj)(double x);
    if (l == 0)      sj = SphericalBesselJ0;
    else if (l == 1) sj = SphericalBesselJ1;
    else if (l == 2) sj = SphericalBesselJ2;
    else if (l == 3) sj = SphericalBesselJ3;
    else if (l == 4) sj = SphericalBesselJ4;
    else if (l == 6) sj = SphericalBesselJ6;
    else if (l == 8) sj = SphericalBesselJ8;
    else {
        error("ComputeDiscreteXiLM: l = %d not supported\n", l);
        return xi;
    }

    // integrate $P(k) k^m j_l(kr) dk$ over the interval $[kmin,kmax]$ using Simpson's rule */
    #pragma omp parallel for
    for(int i = 0; i < Nr; i++) {
        
        // the integrand for this r
        parray integrand(Nk);
        for(int j = 0; j < Nk; j++) 
            integrand[j] = exp(-pow2(k[j]*smoothing)) * pk[j] * pow(k[j], m) / (2*M_PI*M_PI) * sj(k[j]*r[i]);
        
        // integrate using simpson's rule
        xi[i] = SimpsIntegrate(k, integrand);
    }
    return xi;
}


void ComputeXiLM(int l, int m, const PowerSpectrum& P,
                 int Nr, const double r[], double xi[],
                 int Nk, double kmin, double kmax)
{
    assert(Nk > 0 && (Nk % 2) == 0);
    const double dk = (kmax - kmin)/Nk;

    /* Choose appropriate spherical Bessel function */
    double (*sj)(double x);
    if (l == 0)      sj = SphericalBesselJ0;
    else if (l == 1) sj = SphericalBesselJ1;
    else if (l == 2) sj = SphericalBesselJ2;
    else if (l == 3) sj = SphericalBesselJ3;
    else if (l == 4) sj = SphericalBesselJ4;
    else if (l == 6) sj = SphericalBesselJ6;
    else if (l == 8) sj = SphericalBesselJ8;
    else {
        error("ComputeXiLM: l = %d not supported\n", l);
        return;
    }

    parray k = parray::linspace(kmin, kmax, Nk+1);
    parray mult(Nk+1);

    #pragma omp parallel for
    for(int j = 0; j <= Nk; j++) {
        /* Multiplicative factor for Simpson's rule: either 1, 2, or 4 */
        mult[j] = 2 + 2*(j % 2) - (j == 0) - (j == Nk);
        /* All other purely k-dependent factors */
        mult[j] *= P(k[j]) * pow(k[j], m) * (dk/3) / (2*M_PI*M_PI);
    }

    /* Integrate $P(k) k^m j_l(kr) dk$ over the interval $[kmin,kmax]$ using Simpson's rule */
    #pragma omp parallel for
    for(int i = 0; i < Nr; i++) {
        xi[i] = 0;
        for(int j = 0; j <= Nk; j++)
            xi[i] += mult[j] * sj(k[j]*r[i]);
    }
}


CorrelationFunction::CorrelationFunction(const PowerSpectrum& P_, double kmin_, double kmax_)
    : P(P_), kmin(kmin_), kmax(kmax_)
{ }

static double f(const PowerSpectrum& P, double r, double k) {
    return k*sin(k*r)*P(k);
}

double CorrelationFunction::Evaluate(double r) const {
    return 1/(2*M_PI*M_PI*r) * Integrate<ExpSub>(bind(f, cref(P), r, _1), kmin, kmax);
}

parray CorrelationFunction::EvaluateMany(const parray& r) const {
    int Nr = (int) r.size();
    parray xi(Nr);
    #pragma omp parallel for
    for(int i = 0; i < Nr; i++)
        xi[i] = Evaluate(r[i]);

    return xi;
}
