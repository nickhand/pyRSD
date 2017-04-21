%{
#include "Kaiser.h"
%}

class Kaiser {
public:

    Kaiser(const PowerSpectrum& P_L, double f, double b1=1.0);

    /* Redshift-space power spectrum */
    double P_s(double k, double mu) const;
    double operator()(const double k, const double mu) const;

    /* Multipoles of redshift-space power spectrum */
    double P_ell(int ell, double k) const;
    parray P_ell(int ell, const parray& k) const;

    /* Multipoles of redshift-space correlation function */
    double Xi_ell(int ell, double r, int Nk = 32768, double kmin = 1e-5, double kmax = 100) const;
    parray Xi_ell(int ell, const parray& r, int Nk = 32768, double kmin = 1e-5, double kmax = 100) const;
    
    void SetGrowthRate(double f_);
    double GetGrowthRate() const;
    void SetLinearBias(double b1_);
    double GetLinearBias() const;
    const PowerSpectrum& GetPowerSpectrum() const;
};

