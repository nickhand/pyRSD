#ifndef CORRELATION_FUNCTION_H
#define CORRELATION_FUNCTION_H

#include "Common.h"
#include "parray.h"

/* Compute the quantity
 *   \xi_l^m(r) = \int_0^\infty \frac{dk}{2\pi^2} k^m j_l(kr) P(k)
 * using Simpson's rule.  The parameters Nk, kmin, and kmax determine the
 * accuracy of the integration.  Note that Nk must be even. */
void ComputeXiLM(int l, int m, const PowerSpectrum& P,
                 int Nr, const double r[], double xi[],
                 int Nk = 32768, double kmin = 0., double kmax = 100.);


parray SmoothedXiMultipole(Spline P, int ell, const parray& r, int Nk=32768, double kmin=0, double kmax=100., double smoothing=0.);

class CorrelationFunction {
public:
    CorrelationFunction(const PowerSpectrum& P, double kmin = 1e-4, double kmax = 1e1);

    double Evaluate(double r) const;
    double operator()(double r) const { return Evaluate(r); }

    parray EvaluateMany(const parray& r) const;
    parray operator()(const parray& r) const { return EvaluateMany(r); }

    const PowerSpectrum& GetPowerSpectrum() const { return P; }

protected:
    const PowerSpectrum& P;
    double kmin, kmax;
};

#endif // CORRELATION_FUNCTION_H
