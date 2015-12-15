%{
#include "LinearPS.h"
%}

class LinearPS : public PowerSpectrum {
public:
    LinearPS(const Cosmology& C, double z = 0);

    const double& GetSigma8AtZ() const;
    const Cosmology& GetCosmology() const;
    const double& GetRedshift() const;
    
    void SetSigma8AtZ(double sigma8_z);
};

