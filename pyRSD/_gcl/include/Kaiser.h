#ifndef KAISER_H
#define KAISER_H

#include "Common.h"
#include "parray.h"

/* Formulas for Kaiser's linear theory redshift-space power spectrum, and
 * Hamilton's corresponding formulas for the redshift-space correlation
 * function. */
class Kaiser {
public:
    
    Kaiser(const PowerSpectrum& P_L, double f, double b1=1.);
    
    /* redshift-space power spectrum */
    double P_s(double k, double mu) const;
    double operator()(double k, double mu) const { return P_s(k, mu); }

    /* Multipoles of redshift-space power spectrum */
    double P_ell(int ell, double k) const;
    parray P_ell(int ell, const parray& k) const;

    /* Multipoles of redshift-space correlation function */
    double Xi_ell(int ell, double r, int Nk = 32768, double kmin = 0, double kmax = 100) const;
    parray Xi_ell(int ell, const parray& r, int Nk = 32768, double kmin = 0, double kmax = 100) const;

    // setter/getters
    void SetGrowthRate(double f_) { f = f_; }
    double GetGrowthRate() const { return f; }
    void SetLinearBias(double b1_) { b1 = b1_; }
    double GetLinearBias() const { return b1; }
    const PowerSpectrum& GetPowerSpectrum() const { return P; }
    

protected:
    const PowerSpectrum& P;
    double f;
    double b1; 
    
private:
    double PowerPolePrefactor(int ell) const;
    
};

#endif // KAISER_H
