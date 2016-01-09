#include <cassert>
#include <cstdio>
#include <ctime>
#include <functional>

#include "Quadrature.h"
#include "PowerSpectrum.h"
#include "SpecialFunctions.h"

using std::bind;
using std::cref;
using namespace std::placeholders;
using namespace Common;

PowerSpectrum::PowerSpectrum() {}

PowerSpectrum::~PowerSpectrum() {}

parray PowerSpectrum::EvaluateMany(const parray& k) const {
    int n = (int)k.size();
    parray pk(n);
    #pragma omp parallel for
    for(int i = 0; i < n; i++)
        pk[i] = Evaluate(k[i]);
    return pk;
}

/* Top-hat window function */
static double W(double x) {
    if(x < 1e-5)
        return 1 - (1./30.)*x*x;
    else
        return 3/pow3(x) * (sin(x) - x*cos(x));
}

static double f(const PowerSpectrum& P, double R, double k) {
    return k*k/(2*M_PI*M_PI) * P(k) * pow2(W(k*R));
}
double PowerSpectrum::Sigma(double R) const {
    double sigma2 = Integrate<ExpSub>(bind(f, cref(*this), R, _1), 1e-5, 1e2, 1e-5, 1e-12);
    return sqrt(sigma2);
}

parray PowerSpectrum::Sigma(const parray& R) const {
    int n = (int)R.size();
    parray sig(n);
    #pragma omp parallel for
    for(int i = 0; i < n; i++)
        sig[i] = Sigma(R[i]);
    return sig;
}

static double g(const PowerSpectrum& P, double k) {
    return P(k);
}
double PowerSpectrum::VelocityDispersion() const {
    return 1/(6*M_PI*M_PI) * Integrate<ExpSub>(bind(g, cref(*this), _1), 1e-5, 1e2, 1e-5, 1e-12);
}

double PowerSpectrum::VelocityDispersion(double k, double factor ) const {
    return 1/(6*M_PI*M_PI) * Integrate<ExpSub>(bind(g, cref(*this), _1), 1e-5, factor*k, 1e-5, 1e-12);
}

parray PowerSpectrum::VelocityDispersion(const parray& k, double factor) const {
    int n = (int)k.size();
    parray sigmasq(n);
    #pragma omp parallel for
    for(int i = 0; i < n; i++)
        sigmasq[i] = VelocityDispersion(k[i], factor);
    return sigmasq;
}

double PowerSpectrum::NonlinearScale() const {
    return 1/sqrt(VelocityDispersion());
}

void PowerSpectrum::Save(const char* filename, double kmin, double kmax, int Nk, bool logscale) {
    FILE* fp = fopen(filename, "w+");
    if(!fp) {
        fprintf(stderr, "PowerSpectrum::Save(): could not write to %s\n", filename);
        return;
    }

    time_t t = time(0);
    fprintf(fp, "# Power spectrum saved at %s\n", ctime(&t));
    fprintf(fp, "# k -- P(k)\n");
    for(int i = 0; i < Nk; i++) {
        double k = logscale ? kmin*exp(i*log(kmax/kmin)/(Nk-1)) : kmin + i*(kmax - kmin)/(Nk-1);
        double p = Evaluate(k);
        fprintf(fp, "%e %e\n", k, p);
    }
}

static double X_integrand(const PowerSpectrum& P, double k, double q) {
    return P(q)*(-2*SphericalBesselJ1(k*q)/(k*q));
}

double PowerSpectrum::X_Zel(double k) const {
    return 1/(2*M_PI*M_PI) * Integrate(bind(X_integrand, cref(*this), k, _1), 1e-5, 1e2, 1e-4, 1e-10);
}

parray PowerSpectrum::X_Zel(const parray& k) const {
    int n = (int)k.size();
    parray out(n);
    #pragma omp parallel for
    for(int i = 0; i < n; i++)
        out[i] = X_Zel(k[i]);
    return out;
}

static double Y_integrand(const PowerSpectrum& P, double k, double q) {
    double x = k*q;
    return P(q)*(-2*SphericalBesselJ0(x) + 6*SphericalBesselJ1(x)/x);
}

double PowerSpectrum::Y_Zel(double k) const {
    return 1/(2*M_PI*M_PI) * Integrate(bind(Y_integrand, cref(*this), k, _1), 1e-5, 1e2, 1e-4, 1e-10);
}

parray PowerSpectrum::Y_Zel(const parray& k) const {
    int n = (int)k.size();
    parray out(n);
    #pragma omp parallel for
    for(int i = 0; i < n; i++)
        out[i] = Y_Zel(k[i]);
    return out;
}

static double Q3_integrand(const PowerSpectrum& P, double q) {
    double Pq = P(q);
    return Pq*Pq/q/q;
}

double PowerSpectrum::Q3_Zel(double k) const {
    return 1/(10*M_PI*M_PI) * pow4(k) * Integrate(bind(Q3_integrand, cref(*this), _1), 1e-5, 1e2, 1e-4, 1e-10);
}

parray PowerSpectrum::Q3_Zel(const parray& k) const {
    int n = (int)k.size();
    parray out(n);
    #pragma omp parallel for
    for(int i = 0; i < n; i++)
        out[i] = Q3_Zel(k[i]);
    return out;
}

// sigma3_squared kernel I_R(k)
static double I_R(double r) {
    double r2 = pow2(r), r4 = pow4(r);
    return (5./(512.*r4*r)) * (-4*r*(1+r2)*(3-14*r2+3*r4) - 3*pow4(r2-1)*log(pow2(r-1)/pow2(r+1)));
}

static double sigma3_sq_integrand(const PowerSpectrum& P, double k, double r) {
    if (r == 1.) r = 1.-1e-10;
    return r*r * P(k*r) * I_R(r);
}

double PowerSpectrum::sigma3_squared(double k) const {
    
    if(k <= 0) return 0;
    return 1 / (2*M_PI*M_PI) * pow3(k) * Integrate(bind(sigma3_sq_integrand, cref(*this), k, _1), 1e-5, 1e2, 1e-4, 1e-10);
}

parray PowerSpectrum::sigma3_squared(const parray& k) const {
    int n = (int)k.size();
    parray out(n);
    #pragma omp parallel for
    for(int i = 0; i < n; i++)
        out[i] = sigma3_squared(k[i]);
    return out;
}



