#ifndef CORRELATION_FUNCTION_H
#define CORRELATION_FUNCTION_H

#include "Common.h"
#include "parray.h"

/* Functions to compute the quantity
      \xi_l^m(r) = \int_0^\infty \frac{dk}{2\pi^2} k^m j_l(kr) P(k) 
*/
// use Simpson's rule.
parray ComputeDiscreteXiLM(int l, int m, const parray& k, const parray& pk, 
                            const parray& r, double smoothing=0.);

// use FFTLog -- k, Pk will be interpolated onto a log-spaced grid
parray ComputeXiLM(int l, int m, const parray& k, const parray& pk, 
                      const parray& r, double smoothing=0.);
  
// use FFTlog, given log-spaced input k, Pk -- fills r, xi 
void ComputeXiLM_fftlog(int l, int m, int N, const double k[], const double pk[], 
                          double r[], double xi[], double smoothing=0.);

/*  Compute the correlation function xi(r) from a power spectrum P(k), this is 
    just Xi_0^2 in the notation above*/
parray pk_to_xi(const parray& k, const parray& pk, const parray& r, double smoothing=0.5);

/* Compute the power spectrum P(k) from a correlation function xi(r), sampled
 * at logarithmically spaced points r[i]. */
parray xi_to_pk(const parray& r, const parray& xi, const parray& k, double smoothing=0.005);


//-------------------------------------------------------------------------------------
class CorrelationFunction {
public:
    CorrelationFunction(const PowerSpectrum& P, double kmin = 1e-4, double kmax = 1e1);

    double Evaluate(double r) const;
    double operator()(double r) const { return Evaluate(r); }

    parray EvaluateMany(const parray& r) const;
    parray operator()(const parray& r) const { return EvaluateMany(r); }

    const PowerSpectrum& GetPowerSpectrum() const { return P; }
    double GetKmin() const { return kmin; }
    void SetKmin(double kmin_) { kmin=kmin_; }
    double GetKmax() const { return kmax; }
    void SetKmax(double kmax_) { kmax=kmax_; }

protected:
    const PowerSpectrum& P;
    double kmin, kmax;
};

#endif // CORRELATION_FUNCTION_H
