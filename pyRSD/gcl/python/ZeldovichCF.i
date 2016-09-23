%{
#include "ZeldovichCF.h"
%}

%feature("kwargs");

class ZeldovichCF {
public:

    ZeldovichCF(const Cosmology& C, double z, double kmin = 1e-5, double kmax = 10.0);
    ZeldovichCF(const ZeldovichPS& ZelPS, double kmin = 1e-5, double kmax = 10.0);
    
    double Evaluate(double r, double smoothing=0.);
    double operator()(double r, double smoothing=0.);

    parray EvaluateMany(const parray& r, double smoothing=0.);
    parray operator()(const parray& r, double smoothing=0.);
    
    void SetSigma8AtZ(double sigma8_z);
};

