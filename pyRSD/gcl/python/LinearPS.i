%{
#include "LinearPS.h"
%}

class LinearPS : public PowerSpectrum {
public:
    LinearPS(const Cosmology& C, double z = 0);

    const Cosmology& GetCosmology() const;
    const double& GetRedshift() const;
};

%extend LinearPS {
%pythoncode {
    def __reduce__(self):
        args = self.GetCosmology(), self.GetRedshift()
        return self.__class__, args
}
}