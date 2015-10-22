%{
#include "CorrelationFunction.h"
%}



%apply (double* INPLACE_ARRAY1, int DIM1) {(double xi[], int xisize)}

parray ComputeDiscreteXiLM(int l, int m, const parray& k, const parray& pk, const parray& r, double smoothing=0.);
                  
void ComputeXiLM(int l, int m, const PowerSpectrum& P,
                 int Nr, const double r[], double xi[],
                 int Nk = 32768, double kmin = 0., double kmax = 100.);

/*  Wrapper for cos_doubles that massages the types */
//%inline %{
//    /*  takes as input two numpy arrays */
//    void compute_discete_xilm(int l, int m, const parray& k, const parray& pk, const parray& r, double xi[], int xisize, double smoothing=0.) {
//        /*  calls the original funcion, providing only the size of the first */
//        ComputeDiscreteXiLM(l, m, k, pk, r, xi, smoothing);
//    }
//%}


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

