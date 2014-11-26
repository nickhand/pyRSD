#include "GSL_Quadrature.h"
#include "Kmn.h"
#include "PowerSpectrum.h"


using std::bind;
using std::cref;
using namespace std::placeholders;


/* Limits of integration for second-order power spectrum */
const double QMIN = 1e-5;
const double QMAX = 1e5;

Kmn::Kmn(const PowerSpectrum& P_L_, double epsrel_) : P_L(P_L_), epsrel(epsrel_) {}


/*----------------------------------------------------------------------------*/
/* The kernels */
/*----------------------------------------------------------------------------*/

// the 2nd order PT density kernel
static double F2(double r, double x) {
    return (7*x + 3*r - 10*r*x*x)/(14*r*(1 + r*r - 2*r*x));
}

// the 2nd order PT velocity kernel
static double G2(double r, double x) {
    return (7*x - r - 6*r*x*x)/(14*r*(1 + r*r - 2*r*x));
}

// the biasing tidal term kernel
static double S2(double r, double x) {
    return (r-x)*(r-x)/(1. + r*r - 2.*r*x) - 1./3.;
}

// k00(\vec{k}, \vec{q})
static double k00(double r, double x) {
    return F2(r, x);
}

// k00s(\vec{k}, \vec{q})
static double k00s(double r, double x) {
    return F2(r, x)*S2(r, x);
}

// k01(\vec{k}, \vec{q})
static double k01(double r, double x) {
    return 1;
}

// k01s(\vec{k}, \vec{q})
static double k01s(double r, double x) {
    return pow2(S2(r, x));
}

static double k02s(double r, double x) {
    return S2(r, x);
}

static double k10(double r, double x) {
    return G2(r, x);
}

static double k10s(double r, double x) {
    return G2(r, x)*S2(r, x);
}

static double k11(double r, double x) {
    return x/r;
}

static double k11s(double r, double x) {
    return x/r*S2(r, x);
}
    
static double k20_a(double r, double x) {
    return -0.5*(1 - x*x) / (1 + r*r - 2*r*x); // this is h03
}

static double k20s_a(double r, double x) {
    return S2(r, x) * (-0.5*(1 - x*x) / (1 + r*r - 2*r*x)); // S2*h03
}
    
static double k20_b(double r, double x) {
    return (0.5 - 1.5*x*x + x/r) / (1 + r*r - 2*r*x); // this is h04
}

static double k20s_b(double r, double x) {
    return S2(r, x)*((0.5 - 1.5*x*x + x/r) / (1 + r*r - 2*r*x)); 
}

// the Kmn integrand
static double KmnIntegrand(double (*f)(double, double), const PowerSpectrum& P_L, double k, double logq, double x) {
    double q = exp(logq);
    double r = q/k, d = 1 + r*r - 2*r*x;
    return q * q * q * P_L(q) * P_L(k*sqrt(d)) * f(r, x);
    
}

template<class KmnKernel>
static double ComputeIntegral(KmnKernel f, const PowerSpectrum& P_L, double k, double epsrel = 1e-5, double qmax = QMAX) {
    if(k <= 0) return 0;
    

    double a[] = { log(QMIN), -1. };
    double b[] = { log(QMAX), 1. };    
    double V = 1 / (4*M_PI*M_PI);
    
    // use a double wrapper of CQUAD to do the doube integral
    return V * DoubleIntegrateCQUAD(bind(KmnIntegrand, f, cref(P_L), k, _1, _2), a, b, epsrel, 0.);      
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
            return ComputeIntegral(k11, P_L, k, epsrel);            
        case 8:
            return ComputeIntegral(k10s, P_L, k, epsrel);
        case 9:
            return ComputeIntegral(k11s, P_L, k, epsrel);
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
            warning("Kmn: invalid indices, m = %d, n = %d\n", m, n);
            return 0;
    }
}




























