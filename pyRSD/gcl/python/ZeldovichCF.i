%{
#include "ZeldovichCF.h"
%}

%feature("kwargs");

class ZeldovichCF {
public:

    ZeldovichCF(const Cosmology& C, double z, double kmin = 1e-5, double kmax = 1.0);

    double Evaluate(double r, double smoothing=0.) const;
    double operator()(double r, double smoothing=0.) const;

    parray EvaluateMany(const parray& r, double smoothing=0.) const;
    parray operator()(const parray& r, double smoothing=0.) const;
};

