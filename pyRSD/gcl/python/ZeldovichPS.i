%{
#include "ZeldovichPS.h"
%}

class ZeldovichPS {
public:
    
    ZeldovichPS(const PowerSpectrum& P_L);
    ZeldovichPS(const PowerSpectrum& P_L, double z, double sigma8, double sigmasq, const parray& X, const parray& Y);
    virtual ~ZeldovichPS();
    
    // translated to __call__ -> calls Evaluate(K)
    double operator()(const double k) const;
    
    // translated to __call__ -> calls EvaluateMany(K)
    parray operator()(const parray& k) const;
    
    const LinearPS& GetLinearPS() const;
    const double& GetRedshift() const;
    const double& GetSigma8() const;
    const Cosmology& GetCosmology() const;
    parray GetXZel();
    parray GetYZel();
    const double& GetSigmaSq();
    
    void SetRedshift(double z); 
    void SetSigma8(double sigma8);
    
};

%extend ZeldovichPS {
%pythoncode {
    def __reduce__(self):
        args = self.GetLinearPS(), self.GetRedshift(), self.GetSigma8(), \
               self.GetSigmaSq(), self.GetXZel(), self.GetYZel()
        return self.__class__, args
}
}

class ZeldovichP00 : public ZeldovichPS {

public:
    
    ZeldovichP00(const PowerSpectrum& P_L);
    ZeldovichP00(const ZeldovichPS& ZelPS);

};


class ZeldovichP01 : public ZeldovichPS {

public:
    
    ZeldovichP01(const PowerSpectrum& P_L);
    ZeldovichP01(const ZeldovichPS& ZelPS);
    
};
