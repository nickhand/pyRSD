#include <functional>
#include <iostream>
#include "Jmn.h"
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
Jmn::Jmn(const PowerSpectrum& P_L_, double epsrel_)
        : P_L(P_L_), epsrel(epsrel_) {}


/*----------------------------------------------------------------------------*/
/* The kernels */
/*----------------------------------------------------------------------------*/

// g00(q/k)
static double g00(double r) {
    if (r < 60) {
        return (1./3024.)*(12./(r*r) - 158. + 100.*r*r - 42.*pow4(r) + 3./pow3(r)*pow3(r*r-1.)*(7.*r*r+2.)*log((r+1.)/fabs(r-1.)));
    }
    else {
        double r12 = pow2(pow6(r));
        double r10 = pow2(pow5(r));
        return (-2./3024)*(70. + 125.*r*r - 354.*pow4(r) + 263.*pow6(r) + 400.*pow8(r) - 1008.*r10 + 5124.*r12)/(105.*r12);
    }
}

// g01(q/k)
static double g01(double r) {
    if (r < 60) {
        return (1./3024.)*(24./(r*r) - 202. + 56.*r*r - 30.*pow4(r) + 3./pow3(r)*pow3(r*r-1.)*(5.*r*r+4.)*log((r+1.)/fabs(r-1.)));
    }
    else {
        double r12 = pow2(pow6(r));
        double r10 = pow2(pow5(r));
        return (-2./3024)*(140. - 65.*r*r - 168.*pow4(r) + 229.*pow6(r) + 656.*pow8(r) - 3312.*r10 + 10500.*r12)/(105.*r12);
        }
}

// g10(q/k)
static double g10(double r) {
    if (r < 60) {
        return (1./1008.)*(-38. + 48.*r*r - 18.*pow4(r) + 9./r*pow3(r*r-1.)*log((r+1.)/fabs(r-1.)));
    }
    else {
        double r10 = pow2(pow5(r));
        return (8./1008)*(-28. - 60.*r*r - 156.*pow4(r) - 572.*pow6(r) - 5148.*pow8(r) + 1001.*r10)/(5005.*r10);
    }
}

// g11(q/k)
static double g11(double r) {
    if (r < 60) {
        return (1./1008.)*(12./(r*r) - 82. + 4.*r*r - 6.*pow4(r) + 3./(r*r*r)*pow3(r*r-1.)*(r*r+2.)*log((r+1.)/fabs(r-1.)));
    }
    else {
        double r12 = pow2(pow6(r));
        double r10 = pow2(pow5(r));
        return (-2./1008)*(70. - 85.*r*r + 6.*pow4(r) + 65.*pow6(r) + 304.*pow8(r) - 1872.*r10 + 5292.*r12)/(105.*r12);
    }
}

// g02(q/k)
static double g02(double r) {
    if (r < 60) {
        return (1./224.)*(2./(r*r)*(r*r+1.)*(3.*pow4(r) - 14.*r*r + 3.) - 3./(r*r*r)*pow4(r*r-1.)*log((r+1.)/fabs(r-1.)));
    }
    else {
        double r12 = pow2(pow6(r));
        double r10 = pow2(pow5(r));
        return (-2./224)*(35. - 95.*r*r + 93.*pow4(r) - 17.*pow6(r) + 128.*pow8(r) - 1152.*r10 + 2688.*r12)/(105.*r12);
    }
}

// g20(q/k)
static double g20(double r) {
    if (r < 60) {
        return (1./672.)*(2./(r*r)*(9. - 109.*r*r + 63.*pow4(r) - 27.*pow6(r)) + 9./(r*r*r)*pow3(r*r-1.)*(3*r*r+1.)*log((r+1.)/fabs(r-1.)));
    }
    else {
        double r12 = pow2(pow6(r));
        double r10 = pow2(pow5(r));
        return (-2./672)*(35. + 45.*r*r - 147.*pow4(r) + 115.*pow6(r) + 192.*pow8(r) - 576.*r10 + 2576.*r12)/(35.*r12);
    }
}

static double JmnIntegrand(double (*g)(double), const PowerSpectrum& P_L, double k, double logq) {
    double q = exp(logq), r = q/k;
    if (r == 1.) r = 1.-1e-10;
    return q * P_L(q) * g(r);
}

template<class JmnKernel>
static double ComputeIntegral(JmnKernel g, const PowerSpectrum& P_L, double k, double epsrel = 1e-5, double qmax = QMAX) {
    if(k <= 0) return 0;

    double q0 = log(QMIN), q1 = log(qmax);
    double V = 1 / (2*M_PI*M_PI);
    return V * Integrate<1>(bind(JmnIntegrand, g, cref(P_L), k, _1), &q0, &q1, epsrel, epsrel*P_L(k)/V);
    
}

parray Jmn::EvaluateMany(const parray& k, int m, int n) const {
    int size = (int)k.size();
    parray toret(size);
    #pragma omp parallel for
    for(int i = 0; i < size; i++) 
        toret[i] = Evaluate(k[i], m, n);
    return toret;
}

double Jmn::Evaluate(double k, int m, int n) const {
    switch(3*m + n) {
        case 0:
            return ComputeIntegral(g00, P_L, k, epsrel);
        case 1:
            return ComputeIntegral(g01, P_L, k, epsrel);
        case 2:
            return ComputeIntegral(g02, P_L, k, epsrel);
        case 3:
            return ComputeIntegral(g10, P_L, k, epsrel);
        case 4:
            return ComputeIntegral(g11, P_L, k, epsrel);
        case 6:
            return ComputeIntegral(g20, P_L, k, epsrel);
        default:
            error("Jmn: invalid indices, m = %d, n = %d\n", m, n);
            return 0;
    }
}




























