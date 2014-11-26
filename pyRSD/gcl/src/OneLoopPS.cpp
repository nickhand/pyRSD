#include <cmath>
#include <functional>
#include "OneLoopPS.h"
#include "Quadrature.h"

using std::bind;
using std::cref;
using namespace std::placeholders;

double KMIN = 1e-5;
double KMAX = 1e1;
int NUM_PTS = 1000;

OneLoopPS::OneLoopPS(const PowerSpectrum& P_L_, double epsrel) : P_L(P_L_), I(P_L, epsrel), J(P_L, epsrel) {}

double OneLoopPS::EvaluateFull(double k) const {
    if (k > KMAX)
        return 0.;  
    else
        return Evaluate(k) + P_L(k);
}

double OneLoopPS::Evaluate(double k) const {
    if (k < KMIN || k > KMAX)
        return 0.;
    else
        return oneloop_spline(k);
}

parray OneLoopPS::EvaluateFull(const parray& k) const {
    int n = (int)k.size();
    parray pk(n);
    #pragma omp parallel for
    for(int i = 0; i < n; i++)
        pk[i] = EvaluateFull(k[i]);
    return pk;
}

OneLoopPdd::OneLoopPdd(const PowerSpectrum& P_L, double epsrel) : OneLoopPS(P_L, epsrel) 
{ 
    OneLoopPdd::InitializeSpline(); 
}

void OneLoopPdd::InitializeSpline() {
        
    // spline seeded at these k values
    parray k_spline = parray::logspace(KMIN, KMAX, NUM_PTS);
    
    // need I00 and J00
    parray I00 = I(k_spline, 0, 0);
    parray J00 = J(k_spline, 0, 0);
    
    
    
    // linear and 1loop values at k_spline
    parray P_lin = P_L(k_spline);
    parray P_1loop = 2*I00 + 6*k_spline*k_spline * J00 * P_lin;
    
    oneloop_spline = CubicSpline(k_spline, P_1loop);
            
}

OneLoopPdv::OneLoopPdv(const PowerSpectrum& P_L, double epsrel) : OneLoopPS(P_L, epsrel) 
{ 
    OneLoopPdv::InitializeSpline(); 
}

void OneLoopPdv::InitializeSpline() {
    
    // spline seeded at these k values
    parray k_spline = parray::logspace(KMIN, KMAX, NUM_PTS);
    
    // need I01 and J01
    parray I01 = I(k_spline, 0, 1);
    parray J01 = J(k_spline, 0, 1);
    
    // linear and 1loop values at k_spline
    parray P_lin = P_L(k_spline);
    parray P_1loop = 2*I01 + 6*k_spline*k_spline * J01 * P_lin;
    
    oneloop_spline = CubicSpline(k_spline, P_1loop);
            
}

OneLoopPvv::OneLoopPvv(const PowerSpectrum& P_L, double epsrel) : OneLoopPS(P_L, epsrel) 
{ 
    OneLoopPvv::InitializeSpline(); 
}

void OneLoopPvv::InitializeSpline() {
    
    // spline seeded at these k values
    parray k_spline = parray::logspace(KMIN, KMAX, NUM_PTS);
    
    // need I11 and J11
    parray I11 = I(k_spline, 1, 1);
    parray J11 = J(k_spline, 1, 1);
    
    // linear and 1loop values at k_spline
    parray P_lin = P_L(k_spline);
    parray P_1loop = 2*I11 + 6*k_spline*k_spline * J11 * P_lin;
    
    oneloop_spline = CubicSpline(k_spline, P_1loop);
            
}

OneLoopP22Bar::OneLoopP22Bar(const PowerSpectrum& P_L, double epsrel) : OneLoopPS(P_L, epsrel) 
{ 
    OneLoopP22Bar::InitializeSpline(); 
}

double OneLoopP22Bar::EvaluateFull(double k) const {
    if (k > KMAX)
        return 0.;  
    else
        return Evaluate(k);
}

parray OneLoopP22Bar::EvaluateFull(const parray& k) const {
    int n = (int)k.size();
    parray pk(n);
    #pragma omp parallel for
    for(int i = 0; i < n; i++)
        pk[i] = EvaluateFull(k[i]);
    return pk;
}

void OneLoopP22Bar::InitializeSpline() {
    
    // spline seeded at these k values
    parray k_spline = parray::logspace(KMIN, KMAX, NUM_PTS);
    
    // I23 is the mu^0 part
    parray I23 = I(k_spline, 2, 3);
    
    // I32 is the mu^2 part
    parray I32 = I(k_spline, 3, 2);
    
    // I33 is the mu^4 part
    parray I33 = I(k_spline, 3, 3);
    
    // linear and 1loop values at k_spline
    parray P_1loop = I23 + (2./3)*I32 + (1./5)*I33;
    oneloop_spline = CubicSpline(k_spline, P_1loop);
            
}

static double f(const OneLoopPS& P, double q) {
    return P.EvaluateFull(q)/(q*q);
}

double OneLoopP22Bar::VelocityKurtosis() const {
    return 0.25/(2*M_PI*M_PI) * Integrate<ExpSub>(bind(f, cref(*this), _1), 1e-5, KMAX, 1e-4, 1e-10);
}
