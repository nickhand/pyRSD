#include <functional>
#include "Imn.h"
#include "PowerSpectrum.h"
#include "Quadrature.h"

using std::bind;
using std::cref;
using namespace std::placeholders;
using namespace Common;

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
static double f00(double u, double v) {
    return (4*pow2(4 + 3*v*v + u*u*(3 - 10*v*v)) / (49*pow4(u*u - v*v)));
}

// f01(\vec{k}, \vec{q})
static double f01(double u, double v) {
    double u2 = u*u, v2 = v*v;
    return (4*(-8 + v2 + u2*(1 + 6*v2))*(-4 - 3*v2 + u2*(-3 + 10*v2))) / (49*pow4(u2 - v2));
}

// f02(\vec{k}, \vec{q})
static double f02(double u, double v) {
    double u2 = u*u, v2 = v*v;
    return -((8*(-1 + u2)*(-1 + v2)*(-4 - 3*v2 + u2*(-3 + 10*v2))) / (7*pow4(u2 - v2)));
}

// f03(\vec{k}, \vec{q})
static double f03(double u, double v) {
    return (8*(-1 + u*u)*(-1 + 3*u*v)*(-1 + v*v)) / (pow4(u - v)*pow2(u + v));
}

// f10(\vec{k}, \vec{q})
static double f10(double u, double v) {
    return (4*(-1 + u*v)*(-4 - 3*v*v + u*u*(-3 + 10*v*v))) / (7*pow4(u - v)*pow2(u + v));
}

// f11(\vec{k}, \vec{q})
static double f11(double u, double v) {
    double u2 = u*u, v2 = v*v;
    return (4*pow2(-8 + u2 + v2 + 6*u2*v2)) / (49*pow4(u2 - v2));
}

// f12(\vec{k}, \vec{q})
static double f12(double u, double v) {
    double u2 = u*u, v2 = v*v;
    return -((8*(-1 + u2)*(-1 + v2)*(-8 + u2 + v2 + 6*u2*v2)) / (7*pow4(u2 - v2)));
}

// f13(\vec{k}, \vec{q})
static double f13(double u, double v) {
    double u2 = u*u, v2 = v*v;
    return (8*(v2 + u2*(1 - 2*v2) + pow3(u)*v*(-2 + 3*v2) + u*(v - 2*pow3(v)))) / (pow4(u -v)*pow2(u + v));
}

// f20(\vec{k}, \vec{q})
static double f20(double u, double v) {
    double u2 = u*u, v2 = v*v;
    return (8*(-1 - v2 + u2*(-1 + 3*v2))*(-4 - 3*v2 + u2*(-3 + 10*v2))) / (7*pow4(u2 - v2));
}

// f21(\vec{k}, \vec{q})
static double f21(double u, double v) {
    double u2 = u*u, v2 = v*v;
    return (8*(-1 - v2 + u2*(-1 + 3*v2))*(-8 + v2 + u2*(1 + 6*v2)))/(7*pow4(u2 - v2));
}

// f22(\vec{k}, \vec{q})
static double f22(double u, double v) {
    return (4*(-1 + u*v)*(-8 + v*v + u*u*(1 + 6*v*v))) / (7*pow4(u - v)*pow2(u + v));
}

// f23(\vec{k}, \vec{q})
static double f23(double u, double v) {
    double u2 = u*u, v2 = v*v;
    return (48*pow2(-1 + u2)*pow2(-1 + v2)) / pow4(u2 - v2);
}

// f30(\vec{k}, \vec{q})
static double f30(double u, double v) {
    return -((8*(1 + v*v + u*u*(1 - 3*v*v) + pow3(u)*v*(-3 + 5*v*v) + u*(v - 3*pow3(v)))) / (pow4(u - v)*pow2(u + v)));
}

// f31(\vec{k}, \vec{q})
static double f31(double u, double v) {
    return -((8*u*(-1 + u*u)*v*(-1 + v*v)) / (pow4(u - v)*pow2(u + v)));
}

// f32(\vec{k}, \vec{q})
static double f32(double u, double v) {
    double u2 = u*u, v2 = v*v;
    return -((16*(-1 + u2)*(-1 + v2)*(-1 - 3*v2 + 3*u2*(-1 + 5*v2))) / pow4(u2 - v2));
}

//f33(\vec{k}, \vec{q})
static double f33(double u, double v) {
    return (16*(3 + 2*v*v + 3*pow4(v) + u*u*(2 + 12*v*v - 30*pow4(v)) + pow4(u)*(3 - 30*v*v + 35*pow4(v))))/pow4(u*u - v*v);
}

static double ImnIntegrand(double (*f)(double, double), const PowerSpectrum& P_L, double k, double logu, double v) {
    double u = exp(logu);
    double q = (k/2)*(u - v);
    double r = (k/2)*(u + v);
    return u * q * r * P_L(q) * P_L(r) * f(u, v);
}

template<class ImnKernel>
static double ComputeIntegral(ImnKernel f, const PowerSpectrum& P_L, double k, double epsrel = 1e-5, bool symmetric=true, double qmax = QMAX) {
    if(k <= 0) return 0;

    double umin = 1, umax = 2*qmax/k;
    double vmin = 0., vmax = 1.;
    double a[] = { log(umin), vmin };
    double b[] = { log(umax), vmax };
    double V = k / (8*M_PI*M_PI);
    double posv = V * Integrate<2>(bind(ImnIntegrand, f, cref(P_L), k, _1, _2), a, b, epsrel, epsrel*P_L(k)/V);
    
    if (symmetric) {
        return 2*posv;
    }
    else {
        a[1] = -1.;
        b[1] = 0.;
        double negv = V * Integrate<2>(bind(ImnIntegrand, f, cref(P_L), k, _1, _2), a, b, epsrel, epsrel*P_L(k)/V);
        return posv+negv;
    }
}

parray Imn::EvaluateMany(const parray& k, int m, int n) const {
    int size = (int)k.size();
    parray toret(size);
    #pragma omp parallel for
    for(int i = 0; i < size; i++)
        toret[i] = Evaluate(k[i], m, n);
    return toret;
}

double Imn::Evaluate(double k, int m, int n) const {
    switch(4*m + n) {
        case 0:
            return ComputeIntegral(f00, P_L, k, epsrel);
        case 1:
            return ComputeIntegral(f01, P_L, k, epsrel);
        case 2:
            return ComputeIntegral(f02, P_L, k, epsrel);
        case 3:
            return ComputeIntegral(f03, P_L, k, epsrel, false);
        case 4:
            return ComputeIntegral(f10, P_L, k, epsrel, false);
        case 5:
            return ComputeIntegral(f11, P_L, k, epsrel);
        case 6:
            return ComputeIntegral(f12, P_L, k, epsrel);
        case 7:
            return ComputeIntegral(f13, P_L, k, epsrel, false);            
        case 8:
            return ComputeIntegral(f20, P_L, k, epsrel);
        case 9:
            return ComputeIntegral(f21, P_L, k, epsrel);
        case 10:
            return ComputeIntegral(f22, P_L, k, epsrel, false);
        case 11:
            return ComputeIntegral(f23, P_L, k, epsrel);
        case 12:
            return ComputeIntegral(f30, P_L, k, epsrel, false);
        case 13:
            return ComputeIntegral(f31, P_L, k, epsrel, false);
        case 14:
            return ComputeIntegral(f32, P_L, k, epsrel);
        case 15:
            return ComputeIntegral(f33, P_L, k, epsrel);
        default:
            error("Imn: invalid indices, m = %d, n = %d\n", m, n);
            return 0;
    }
}




























