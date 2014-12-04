#include <functional>
#include "ImnOneLoop.h"
#include "OneLoopPS.h"
#include "Quadrature.h"

using std::bind;
using std::cref;
using namespace std::placeholders;
using namespace Common;

/* Limits of integration for second-order power spectrum */
const double QMIN = 1e-5;
const double QMAX = 1e1;

// Constructor
ImnOneLoop::ImnOneLoop(const OneLoopPS& P_1_, double epsrel_)
    : P_1(P_1_), epsrel(epsrel_), equal(true), P_2(P_1_), P_L(P_1.GetLinearPS())
{
    // store the reference to the linear PS too
    //P_L = P_1.GetLinearPS();
}

ImnOneLoop::ImnOneLoop(const OneLoopPS& P_1_, const OneLoopPS& P_2_, double epsrel_)
    : P_1(P_1_), epsrel(epsrel_), equal(false), P_2(P_2_), P_L(P_1.GetLinearPS()) 
{
    // store the reference to the linear PS too
    // P_L = P_1.GetLinearPS();
}

/*----------------------------------------------------------------------------*/
/* The kernels */
/*----------------------------------------------------------------------------*/


// h01(\vec{k}, \vec{q})
static double h01(double u, double v) {
    return -((2*(-1 + u*u)*(-1 + v*v))/pow4(u - v));
}

// h02(\vec{k}, \vec{q})
static double h02(double u, double v) {
    return (6 - 8*u*v - 2*v*v + u*u*(-2 + 6*v*v))/pow4(u - v);
}

// h03(\vec{k}, \vec{q})
static double h03(double u, double v) {
    double u2 = u*u, v2 = v*v;
    return (2*(-1 + u2)*(-1 + v2))/pow2(u2 - v2);
}

// h04(\vec{k}, \vec{q})
static double h04(double u, double v) {
    double u2 = u*u, v2 = v*v;
    return (2*(1 + v2 + u2*(1 - 3*v2))) / pow2(u2 - v2);
}

// f23(\vec{k}, \vec{q})
static double f23(double u, double v) {
    double u2 = u*u, v2 = v*v;
    return (48*pow2(-1 + u2)*pow2(-1 + v2)) / pow4(u2 - v2);
}

// f32(\vec{k}, \vec{q})
static double f32(double u, double v) {
    double u2 = u*u, v2 = v*v;
    return -((16*(-1 + u2)*(-1 + v2)*(-1 - 3*v2 + 3*u2*(-1 + 5*v2))) / pow4(u2 - v2));
}

// f33(\vec{k}, \vec{q})
static double f33(double u, double v) {
    return (16*(3 + 2*v*v + 3*pow4(v) + u*u*(2 + 12*v*v - 30*pow4(v)) + pow4(u)*(3 - 30*v*v + 35*pow4(v))))/pow4(u*u - v*v);
}


static double ImnIntegrand(double (*f)(double, double), const PowerSpectrum& P_1, const PowerSpectrum& P_2, double k, double logu, double v) {
    double u = exp(logu);
    double q = (k/2)*(u - v);
    double r = (k/2)*(u + v);
    return u * q * r * P_1(q) * P_2(r) * f(u, v);
}

template<class ImnKernel>
static double ComputeIntegral(ImnKernel f, const PowerSpectrum& P_1, const PowerSpectrum& P_2, double k, double epsrel = 1e-5, double qmax = QMAX) {
    if(k <= 0) return 0;

    double umin = 1, umax = 2*qmax/k;
    double vmin = -1., vmax = 1.;
    double a[] = { log(umin), vmin };
    double b[] = { log(umax), vmax };
    double V = k / (8*M_PI*M_PI);
    
    return V * Integrate<2>(bind(ImnIntegrand, f, cref(P_1), cref(P_2), k, _1, _2), a, b, epsrel, epsrel);
}


double ImnOneLoop::EvaluateLinear(double k, int m, int n) const {
    switch(2*m + n) {
        case 1:
            return ComputeIntegral(h01, P_L, P_L, k, epsrel);
        case 2:
            return ComputeIntegral(h02, P_L, P_L, k, epsrel);
        case 3:
            return ComputeIntegral(h03, P_L, P_L, k, epsrel);
        case 4:
            return ComputeIntegral(h04, P_L, P_L, k, epsrel);
        case 7:
            return ComputeIntegral(f23, P_L, P_L, k, epsrel);
        case 8:
            return ComputeIntegral(f23, P_L, P_L, k, epsrel);
        case 9:
            return ComputeIntegral(f33, P_L, P_L, k, epsrel);            
        default:
            error("ImnOneLoop: invalid indices, m = %d, n = %d\n", m, n);
            return 0;
    }
}

parray ImnOneLoop::EvaluateLinear(const parray& k, int m, int n) const {
    int size = (int)k.size();
    parray toret(size);
    #pragma omp parallel for
    for(int i = 0; i < size; i++) 
        toret[i] = EvaluateLinear(k[i], m, n);
    return toret;
}

double ImnOneLoop::EvaluateCross(double k, int m, int n) const {
    double part1; 
    switch(2*m + n) {
        case 1:
            part1 = ComputeIntegral(h01, P_L, P_2, k, epsrel);
            if (equal)
                return 2*part1;
            else
                return part1 + ComputeIntegral(h01, P_1, P_L, k, epsrel);
        case 2:
            part1 = ComputeIntegral(h02, P_L, P_2, k, epsrel);
            if (equal)
                return 2*part1;
            else
                return part1 + ComputeIntegral(h02, P_1, P_L, k, epsrel);
        case 3:
            part1 = ComputeIntegral(h03, P_L, P_2, k, epsrel);
            if (equal)
                return 2*part1;
            else
                return part1 + ComputeIntegral(h03, P_1, P_L, k, epsrel);
        case 4:
            part1 = ComputeIntegral(h04, P_L, P_2, k, epsrel);
            if (equal)
                return 2*part1;
            else
                return part1 + ComputeIntegral(h04, P_1, P_L, k, epsrel);
        case 7:
            part1 = ComputeIntegral(f23, P_L, P_2, k, epsrel);
            if (equal)
                return 2*part1;
            else
                return part1 + ComputeIntegral(f23, P_1, P_L, k, epsrel);
        case 8:
            part1 = ComputeIntegral(f32, P_L, P_2, k, epsrel);
            if (equal)
                return 2*part1;
            else
                return part1 + ComputeIntegral(f32, P_1, P_L, k, epsrel);
        case 9:
            part1 = ComputeIntegral(f33, P_L, P_2, k, epsrel);
            if (equal)
                return 2*part1;
            else
                return part1 + ComputeIntegral(f33, P_1, P_L, k, epsrel);            
        default:
            error("ImnOneLoop: invalid indices, m = %d, n = %d\n", m, n);
            return 0;
    }
}

parray ImnOneLoop::EvaluateCross(const parray& k, int m, int n) const {
    int size = (int)k.size();
    parray toret(size);
    #pragma omp parallel for
    for(int i = 0; i < size; i++) 
        toret[i] = EvaluateCross(k[i], m, n);
    return toret;
}

double ImnOneLoop::EvaluateOneLoop(double k, int m, int n) const {
    switch(2*m + n) {
        case 1:
            return ComputeIntegral(h01, P_1, P_2, k, epsrel);
        case 2:
            return ComputeIntegral(h02, P_1, P_2, k, epsrel);
        case 3:
            return ComputeIntegral(h03, P_1, P_2, k, epsrel);
        case 4:
            return ComputeIntegral(h04, P_1, P_2, k, epsrel);
        case 7:
            return ComputeIntegral(f23, P_1, P_2, k, epsrel);
        case 8:
            return ComputeIntegral(f23, P_1, P_2, k, epsrel);
        case 9:
            return ComputeIntegral(f33, P_1, P_2, k, epsrel);            
        default:
            error("ImnOneLoop: invalid indices, m = %d, n = %d\n", m, n);
            return 0;
    }
}

parray ImnOneLoop::EvaluateOneLoop(const parray& k, int m, int n) const {
    int size = (int)k.size();
    parray toret(size);
    #pragma omp parallel for
    for(int i = 0; i < size; i++) 
        toret[i] = EvaluateOneLoop(k[i], m, n);
    return toret;
}




























