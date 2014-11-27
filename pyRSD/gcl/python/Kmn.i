%{
#include "Kmn.h"
%}

class Kmn {
public: 

    Kmn(const PowerSpectrum& P_L, double epsrel = 1e-5);

    double Evaluate(double k, int m, int n) const;
    parray EvaluateMany(const parray& k, int m, int n) const;
    
    %extend {
        double __call__(double k, int m, int n) const { return $self->Evaluate(k, m, n); }
        parray __call__(const parray& k, int m, int n) const { return $self->EvaluateMany(k, m, n); }
    };

};

