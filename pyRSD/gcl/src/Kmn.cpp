#include <functional>
#include "Quadrature.h"
#include "Kmn.h"
#include "PowerSpectrum.h"


using std::bind;
using std::cref;
using namespace std::placeholders;
using namespace Common;

/* Limits of integration for second-order power spectrum */
const double QMIN = 1e-5;
const double QMAX = 1e5;

Kmn::Kmn(const PowerSpectrum& P_L_, double epsrel_) : P_L(P_L_), epsrel(epsrel_) {}


/*----------------------------------------------------------------------------*/
/* The kernels */
/*----------------------------------------------------------------------------*/

// the 2nd order PT density kernel
static double F2(double u, double v) {
    double u2 = u*u, v2 = v*v;
    return (8 + 6*v2 + u2*(6 - 20*v2))/(7*pow2(u2 - v2));
}

// the 2nd order PT velocity kernel
static double G2(double u, double v) {
    double u2 = u*u, v2 = v*v;
    return -((2*(-8 + u2 + v2 + 6*u2*v2)) / (7*pow2(u2 - v2)));
}

// the biasing tidal term kernel
static double S2(double u, double v) {
    double u2 = u*u, v2 = v*v;
    return (2*(6 + u2*u2 - 6*v2 + v2*v2 + u2*(-6 + 4*v2)))/(3*pow2(u2 - v2));
}

// h03 from Zvonimir's notation (overall factor of k^{-2} not included)
static double h03(double u, double v) {
    double u2 = u*u, v2 = v*v;
    return 2*(-1 + u2)*(-1 + v2) / pow2(u2 - v2);
}

// h04 from Zvonimir's notation (overall factor of k^{-2} not included)
static double h04(double u, double v) {
    double u2 = u*u, v2 = v*v;
    return 2*(1 + v2 + u2*(1-3*v2)) / pow2(u2 - v2);
}

// k00(\vec{k}, \vec{q})
static double k00(double u, double v) {
    return F2(u, v);
}

// k00s(\vec{k}, \vec{q})
static double k00s(double u, double v) {
    return F2(u, v)*S2(u, v);
}

// k01(\vec{k}, \vec{q})
static double k01(double, double) {
    return 1;
}

// k01s(\vec{k}, \vec{q})
static double k01s(double u, double v) {
    return pow2(S2(u, v));
}

static double k02s(double u, double v) {
    return S2(u, v);
}

static double k10(double u, double v) {
    return G2(u, v);
}

static double k10s(double u, double v) {
    return G2(u, v)*S2(u, v);
}

static double k11(double u, double v) {
    return (2 - 2*u*v)/pow2(u - v);
}

static double k11s(double u, double v) {
    return (2 - 2*u*v)/pow2(u - v) * S2(u, v);
}
    
static double k20_a(double u, double v) {
    return h03(u, v);
}

static double k20s_a(double u, double v) {
    return S2(u, v)*h03(u, v);
}
    
static double k20_b(double u, double v) {
    return h04(u, v);
}

static double k20s_b(double u, double v) {
    return S2(u, v)*h04(u, v);
}

// the Kmn integrand
static double KmnIntegrand(double (*f)(double, double), const PowerSpectrum& P_L, double k, double logu, double v) {    
    double u = exp(logu);
    double q = (k/2)*(u - v);
    double r = (k/2)*(u + v);
    return u * q * r * P_L(q) * P_L(r) * f(u, v);
    
}

template<class KmnKernel>
static double ComputeIntegral(KmnKernel f, const PowerSpectrum& P_L, double k, double epsrel = 1e-3, bool symmetric=true, double qmax = QMAX) {
    if(k <= 0) return 0;
    

    double umin = 1, umax = 2*qmax/k;
    double vmin = 0, vmax = 1.;
    double a[] = { log(umin), vmin };
    double b[] = { log(umax), vmax };
    double V = k / (4*M_PI*M_PI);
    double ans;
    
    if (symmetric)
        ans =  V * Integrate<2>(bind(KmnIntegrand, f, cref(P_L), k, _1, _2), a, b, epsrel, 0.);
    else {
        a[1] = -1;
        ans = 0.5 * V * Integrate<2>(bind(KmnIntegrand, f, cref(P_L), k, _1, _2), a, b, epsrel, 0.);
    }
    return ans;
    
}

parray Kmn::EvaluateMany(const parray& k, int m, int n,  bool tidal, int part) const {
    int size = (int)k.size();
    parray toret(size);
    #pragma omp parallel for
    for(int i = 0; i < size; i++) 
        toret[i] = Evaluate(k[i], m, n, tidal, part);
    return toret;
}

double Kmn::Evaluate(double k, int m, int n, bool tidal, int part) const {

    switch(2*(3*m + int(tidal)) + n) {
        case 0:
            return ComputeIntegral(k00, P_L, k, epsrel);
        case 1:
            return ComputeIntegral(k01, P_L, k, epsrel);
        case 2:
            return ComputeIntegral(k00s, P_L, k, epsrel);
        case 3:
            return ComputeIntegral(k01s, P_L, k, epsrel);
        case 4:
            return ComputeIntegral(k02s, P_L, k, epsrel);
        case 6:
            return ComputeIntegral(k10, P_L, k, epsrel);
        case 7:
            return ComputeIntegral(k11, P_L, k, epsrel, false, QMAX);            
        case 8:
            return ComputeIntegral(k10s, P_L, k, epsrel);
        case 9:
            return ComputeIntegral(k11s, P_L, k, epsrel, false, QMAX);
        case 12:
            if (part == 0) 
                return ComputeIntegral(k20_a, P_L, k, epsrel);
            else
                return ComputeIntegral(k20_b, P_L, k, epsrel);
        case 14:
            if (part == 0) 
                return ComputeIntegral(k20s_a, P_L, k, epsrel);
            else
                return ComputeIntegral(k20s_b, P_L, k, epsrel);
        default:
            error("Kmn: invalid indices, m = %d, n = %d\n", m, n);
            return 0;
    }
}




























