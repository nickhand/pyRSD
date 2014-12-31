%{
#include "Jmn.h"
%}

class Jmn {
public: 

    Jmn(const PowerSpectrum& P_L, double epsrel = 1e-4);

    // translated to __call__ -> calls Evaluate(K)
    double operator()(const double k, int m, int n) const;
    
    // translated to __call__ -> calls EvaluateMany(K)
    parray operator()(const parray& k, int m, int n) const;
    
    const LinearPS& GetLinearPS() const;
    const double& GetEpsrel() const;

};
