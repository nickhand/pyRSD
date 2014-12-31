%{
#include "Kmn.h"
%}

class Kmn {
public: 

    Kmn(const PowerSpectrum& P_L, double epsrel = 1e-3);

    // translated to __call__ -> calls Evaluate(K)
    double operator()(const double k, int m, int n, bool tidal=false, int part=0) const;
    
    // translated to __call__ -> calls EvaluateMany(K)
    parray operator()(const parray& k, int m, int n, bool tidal=false, int part=0) const;
    
    const LinearPS& GetLinearPS() const;
    const double& GetEpsrel() const;
};

