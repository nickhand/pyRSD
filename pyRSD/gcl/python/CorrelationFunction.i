%{
#include "CorrelationFunction.h"
%}

%nodefaultctor IntegrationMethods;
%nodefaultdtor IntegrationMethods;
struct IntegrationMethods {
    enum Type {FFTLOG=0, SIMPS, TRAPZ}; 
};
    

%apply (double* INPLACE_ARRAY1, int DIM1) {(double r[], int rsize)}
%apply (double* INPLACE_ARRAY1, int DIM1) {(double xi[], int xisize)}


parray ComputeXiLM(int l, int m, const parray& k, const parray& pk, const parray& r,
                    double smoothing=0., IntegrationMethods::Type method=IntegrationMethods::FFTLOG);

void ComputeXiLM_fftlog(int l, int m, const parray& k, const parray& pk, 
                          double r[], double xi[], double q=0, double smoothing=0.);

parray pk_to_xi(int ell, const parray& k, const parray& pk, const parray& r, 
                double smoothing=0.5, IntegrationMethods::Type method=IntegrationMethods::FFTLOG);
parray xi_to_pk(int ell, const parray& r, const parray& xi, const parray& k, 
                double smoothing=0.005, IntegrationMethods::Type method=IntegrationMethods::FFTLOG);


/*  Wrapper for ComputeXiLM_fftlog to massage types */
%inline %{
    /*  takes as input two numpy arrays */
    void compute_xilm_fftlog(int l, int m, const parray& k, const parray& pk,
                              double r[], int rsize, double xi[], int xisize, double q=0, double smoothing=0.)
    {
        /*  calls the original funcion, providing only the size of the first */
        ComputeXiLM_fftlog(l, m, k, pk, r, xi, q, smoothing);
   }
%}

class CorrelationFunction {
public:

    CorrelationFunction(const PowerSpectrum& P, double kmin = 1e-4, double kmax = 1e1);

    // calls Evaluate(r)
    double operator()(double r) const;

    // calls EvaluateMany(r)
    parray operator()(const parray& r) const;

    const PowerSpectrum& GetPowerSpectrum() const;
    double GetKmin() const;
    void SetKmin(double kmin_);
    double GetKmax() const;
    void SetKmax(double kmax_);
};

