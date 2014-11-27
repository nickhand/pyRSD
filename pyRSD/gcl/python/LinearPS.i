%{
#include "LinearPS.h"
%}

class LinearPS : public PowerSpectrum {
public:
    LinearPS(Cosmology& C, double z = 0);

    const Cosmology& GetCosmology() const;
};
