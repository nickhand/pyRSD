#ifndef KAISER_H
#define KAISER_H

#include "Common.h"
#include "parray.h"

/* Formulas for Kaiser's linear theory redshift-space power spectrum, and
 * Hamilton's corresponding formulas for the redshift-space correlation
 * function. */
class Kaiser {
public:
    Kaiser(const PowerSpectrum& P_L);

    /* Redshift-space power spectrum */
    double P_s(double k, double mu) const;

    /* Multipoles of redshift-space power spectrum */
    double PMultipole(int ell, double k) const;
    parray PMultipole(int ell, const parray& k) const;

    /* Multipoles of redshift-space correlation function */
    double XiMultipole(int ell, double r, int Nk = 32768, double kmin = 0, double kmax = 100) const;
    parray XiMultipole(int ell, const parray& r, int Nk = 32768, double kmin = 0, double kmax = 100) const;


protected:
    const PowerSpectrum& P;
    double f;
};

#endif // KAISER_H
