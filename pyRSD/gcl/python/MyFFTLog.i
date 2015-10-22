%{
#include "MyFFTLog.h"
%}

%apply (double* INPLACE_ARRAY1, int DIM1) {(double r[], int rsize)}
%apply (double* INPLACE_ARRAY1, int DIM1) {(double xi[], int xisize)}

/* Compute the correlation function xi(r) from a power spectrum P(k), sampled
 * at logarithmically spaced points k[j]. */
void pk2xi(int N, const parray& k, const parray& pk, double r[], double xi[]);

/* Compute the power spectrum P(k) from a correlation function xi(r), sampled
 * at logarithmically spaced points r[i]. */
void xi2pk(int N, const parray& r, const parray& xi, parray& k, parray& pk);

void fftlog_ComputeXiLM(int l, int m, int N, const parray& k, const parray& pk, parray& r, parray& xi);

/*  Wrapper for cos_doubles that massages the types */
%inline %{
    /*  takes as input two numpy arrays */
    void pk2xi_func(const parray& k, const parray& pk, double r[], int rsize, double xi[], int xisize) {
        /*  calls the original funcion, providing only the size of the first */
        pk2xi(rsize, k, pk, r, xi);
    }
%}