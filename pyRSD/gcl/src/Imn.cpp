#include <functional>

#include "Imn.h"
#include "PowerSpectrum.h"
#include "Quadrature.h"
#include "omp.h"
using std::bind;
using std::cref;
using namespace std::placeholders;


/* Limits of integration for second-order power spectrum */
const double QMIN = 1e-5;
const double QMAX = 1e5;

// Constructor
Imn::Imn(const PowerSpectrum& P_L_, double epsrel_)
    : P_L(P_L_), epsrel(epsrel_) {}


/*----------------------------------------------------------------------------*/
/* The kernels */
/*----------------------------------------------------------------------------*/

// f00(\vec{k}, \vec{q})
static double f00(double r, double x) {
    double s = (7*x + 3*r - 10*r*x*x) / (14*r*(1 + r*r - 2*r*x));
    return s*s;
}

// f01(\vec{k}, \vec{q})
static double f01(double r, double x) {
    return (7*x + 3*r - 10*r*x*x)*(7*x - r - 6*r*x*x) / pow2(14*r*(1+r*r-2*r*x));
}

// f02(\vec{k}, \vec{q})
static double f02(double r, double x) {
    return (x*x - 1)*(7*x + 3*r - 10*r*x*x) / (14*r*pow2(1 + r*r - 2*r*x));
}

// f03(\vec{k}, \vec{q})
static double f03(double r, double x) {
    return (1 - x*x)*(3*r*x - 1) / (r*r*(1 + r*r - 2*r*x));
}

// f10(\vec{k}, \vec{q})
static double f10(double r, double x) {
    return x*(7*x + 3*r - 10*r*x*x) / (14*r*r*(1 + r*r - 2*r*x));
}

// f11(\vec{k}, \vec{q})
static double f11(double r, double x) {
    double s = (7*x -r - 6*r*x*x) / (14*r*(1 + r*r - 2*r*x));
    return s*s;
}

// f12(\vec{k}, \vec{q})
static double f12(double r, double x) {
    return (x*x - 1)*(7*x - r - 6*r*x*x) / (14*r*pow2(1 + r*r - 2*r*x));
}

// f13(\vec{k}, \vec{q})
static double f13(double r, double x) {
    return (4*r*x + 3*x*x - 6*r*pow3(x) - 1) / (2*r*r*(1 + r*r - 2*r*x));
}

// f20(\vec{k}, \vec{q})
static double f20(double r, double x) {
    return (2*x + r - 3*r*x*x)*(7*x + 3*r - 10*r*x*x) / (14*r*r*pow2(1 + r*r - 2*r*x));
}

// f21(\vec{k}, \vec{q})
static double f21(double r, double x) {
    return (2*x + r - 3*r*x*x)*(7*x - r - 6*r*x*x) / (14*r*r*pow2(1 + r*r - 2*r*x));
}

// f22(\vec{k}, \vec{q})
static double f22(double r, double x) {
    return x*(7*x - r - 6*r*x*x) / (14*r*r*(1 + r*r - 2*r*x));
}

// f23(\vec{k}, \vec{q})
static double f23(double r, double x) {
    return 3*pow2((1 - x*x) / (1 + r*r - 2*r*x));
}

// f30(\vec{k}, \vec{q})
static double f30(double r, double x) {
    return (1 - 3*x*x - 3*r*x + 5*r*x*x*x) / (r*r*(1 + r*r - 2*r*x));
}

// f31(\vec{k}, \vec{q})
static double f31(double r, double x) {
    return (1 - 2*r*x)*(1 - x*x) / (2*r*r*(1 + r*r - 2*r*x));
}

// f32(\vec{k}, \vec{q})
static double f32(double r, double x) {
    return (1 - x*x)*(2 - 12*r*x - 3*r*r + 15*r*r*x*x) / (r*r*pow2(1 + r*r - 2*r*x));
}

// f33(\vec{k}, \vec{q})
static double f33(double r, double x) {
    return (-4 + 12*x*x + 24*r*x - 40*r*pow3(x) + 3*r*r - 30*r*r*x*x + 35*r*r*pow4(x)) / (r*r*pow2(1 + r*r - 2*r*x));
}
static double F2(double k, double q, double r) {
    q = fmax(q, QMIN); r = fmax(r, QMIN);
    double k2 = k*k, q2 = q*q, r2 = r*r;
    return (2*k2*k2 - 5*pow2(q2 - r2) + 3*k2*(q2 + r2))/(28*q2*r2);
}

static double ImnIntegrand(const PowerSpectrum& P_L, double k, double logu, double v) {
    double u = exp(logu);
    double q = (k/2)*(u - v);
    double r = (k/2)*(u + v);
    return u * q * r * P_L(q) * P_L(r) * pow2(F2(k, q, r));
}

//template<class ImnKernel>
static double ComputeIntegral(const PowerSpectrum& P_L, double k, double epsrel = 1e-5, double qmax = QMAX) {
    if(k <= 0) return 0;

    double umin = 1, umax = 2*qmax/k;
    double vmin = 0, vmax = 1;
    double a[] = { log(umin), vmin };
    double b[] = { log(umax), vmax };
    double V = 1. / (2*M_PI*M_PI);
    return V * Integrate<2>(bind(ImnIntegrand, cref(P_L), k, _1, _2), a, b, epsrel, epsrel*P_L(k)/V);
}

parray Imn::EvaluateMany(int m, int n, const parray& k) const {
    int size = (int)k.size();
    parray toret(size);
    #pragma omp parallel for
    for(int i = 0; i < size; i++) {
        toret[i] = Evaluate(m, n, k[i]);
        debug("i = %d\n", i);
        debug("        num of threads = %d\n", omp_get_num_threads());
    }
    return toret;
}

double Imn::Evaluate(int m, int n, double k) const {
    switch(4*m + n) {
        case 0:
            return ComputeIntegral(P_L, k, epsrel);
        // case 1:
        //     return ComputeIntegral(f01, P_L, k, epsrel);
        // case 2:
        //     return ComputeIntegral(f02, P_L, k, epsrel);
        // case 3:
        //     return ComputeIntegral(f03, P_L, k, epsrel);
        // case 4:
        //     return ComputeIntegral(f10, P_L, k, epsrel);
        // case 5:
        //     return ComputeIntegral(f11, P_L, k, epsrel);
        // case 6:
        //     return ComputeIntegral(f12, P_L, k, epsrel);
        // case 7:
        //     return ComputeIntegral(f13, P_L, k, epsrel);            
        // case 8:
        //     return ComputeIntegral(f20, P_L, k, epsrel);
        // case 9:
        //     return ComputeIntegral(f21, P_L, k, epsrel);
        // case 10:
        //     return ComputeIntegral(f22, P_L, k, epsrel);
        // case 11:
        //     return ComputeIntegral(f23, P_L, k, epsrel);
        // case 12:
        //     return ComputeIntegral(f30, P_L, k, epsrel);
        // case 13:
        //     return ComputeIntegral(f31, P_L, k, epsrel);
        // case 14:
        //     return ComputeIntegral(f32, P_L, k, epsrel);
        // case 15:
        //     return ComputeIntegral(f33, P_L, k, epsrel);
        default:
            warning("Imn: invalid indices, m = %d, n = %d\n", m, n);
            return 0;
    }
}




























