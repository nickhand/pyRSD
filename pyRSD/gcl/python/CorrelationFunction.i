%{
#include "CorrelationFunction.h"
%}


void ComputeXiLM(int l, int m, const PowerSpectrum& P,
                 int Nr, const double r[], double xi[],
                 int Nk = 32768, double kmin = 0., double kmax = 100.);

parray SmoothedXiMultipole(Spline P, int ell, const parray& r, int Nk=32768, double kmin=0, double kmax=100., double smoothing=0.);
                 
class CorrelationFunction {
public:

    CorrelationFunction(const PowerSpectrum& P, double kmin = 1e-4, double kmax = 1e1);

    // calls Evaluate(r)
    double operator()(double r) const;

    // calls EvaluateMany(r)
    parray operator()(const parray& r) const;

    const PowerSpectrum& GetPowerSpectrum() const;
};

