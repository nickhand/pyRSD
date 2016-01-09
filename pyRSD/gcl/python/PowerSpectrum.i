%{
#include "PowerSpectrum.h"
%}

class PowerSpectrum {
public:
    PowerSpectrum();
    virtual ~PowerSpectrum();

    // translated to __call__ -> calls Evaluate(K)
    double operator()(const double k) const;
    
    // translated to __call__ -> calls EvaluateMany(K)
    parray operator()(const parray& k) const;
     
    const Cosmology& GetCosmology() const = 0;
    void Save(const char* filename, double kmin = 1e-3, double kmax = 1, int Nk = 1001, bool log = false);
    
    // mass variance sigma(R)
    double Sigma(double R) const;
    parray Sigma(const parray& R) const;
    
    // 1D velocity dispersion
    double VelocityDispersion() const;
    
    // 1 / 1D velocity disp
    double NonlinearScale() const;
    
    // sigma_v as a function of k
    parray VelocityDispersion(const parray& k, double factor = 0.5) const;
    double VelocityDispersion(const double k, double factor = 0.5) const;
    
    // Zeldovich approximation integrals
    parray X_Zel(const parray& k) const;
    double X_Zel(const double k) const;
    
    parray Y_Zel(const parray& k) const;
    double Y_Zel(const double k) const;
    
    parray Q3_Zel(const parray& k) const;
    double Q3_Zel(const double k) const;
    
    double sigma3_squared(double k) const;
    parray sigma3_squared(const parray& k) const;
};

