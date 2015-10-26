#include "CorrelationFunction.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <functional>

#include "PowerSpectrum.h"
#include "Quadrature.h"
#include "DiscreteQuad.h"
#include "SpecialFunctions.h"
#include "Spline.h"
#include "MyFFTLog.h"

using std::bind;
using std::cref;
using namespace std::placeholders;
using namespace Common;



void ComputeXiLM_fftlog(int l, int m, int N, const double k[], const double pk[], 
                          double r[], double xi[], double smoothing) 
{
    // complex arrays for the FFT
    dcomplex* a = new dcomplex[N];
    dcomplex* b = new dcomplex[N];
    
    // the fftlog magic
    for(int i = 0; i < N; i++)
        a[i] = pow(k[i], m - 0.5) * pk[i] * exp(-pow2(k[i]*smoothing));
    fht(N, &k[0], a, r, b, l + 0.5);
    for(int i = 0; i < N; i++)
        xi[i] = std::real(pow(2*M_PI*r[i], -1.5) * b[i]);
    
    // delete arrays
    delete[] b;
    delete[] a;
}


parray ComputeXiLM(int l, int m, const parray& k_, const parray& pk_, const parray& r, double smoothing) {
    
    int N(k_.size());
    double r_[N];
    double xi_[N]; 
    
    // force log-spacing using a spline
    parray k = parray::logspace(k_.min(), k_.max(), k_.size());
    auto pk_spline = CubicSpline(k_, pk_);
    auto pk = pk_spline(k);
    
    // call fftlog on the double[] arrays
    ComputeXiLM_fftlog(l, m, N, &k[0], &pk[0], r_, xi_, smoothing);
    
    // return the result at desired domain values using a spline
    auto spline = CubicSpline(N, r_, xi_);
    return spline(r);
}

parray pk_to_xi(int ell, const parray& k, const parray& pk, const parray& r, double smoothing) 
{
    return pow(-1, ell/2)*ComputeXiLM(ell, 2, k, pk, r, smoothing);
}

parray xi_to_pk(int ell, const parray& r, const parray& xi, const parray& k, double smoothing)
{
    static const double TwoPiCubed = 8*M_PI*M_PI*M_PI;
    return pow(-1, ell/2) * TwoPiCubed * ComputeXiLM(ell, 2, r, xi, k, smoothing);
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


CorrelationFunction::CorrelationFunction(const PowerSpectrum& P_, double kmin_, double kmax_)
    : P(P_), kmin(kmin_), kmax(kmax_)
{
     
}

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
