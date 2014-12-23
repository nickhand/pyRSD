%{
#include "OneLoopPS.h"
%}

class OneLoopPS : public PowerSpectrum {
public:
    
    OneLoopPS(const PowerSpectrum& P_L, double epsrel = 1e-4);
    
    // returns the full 1-loop power spectrum (1-loop term + linear term)
    double EvaluateFull(double k) const;
    parray EvaluateFull(const parray& k) const;
    
    const LinearPS& GetLinearPS() const;
    const double& GetRedshift() const;
    const Cosmology& GetCosmology() const;
    const double& GetEpsrel() const;
};

%extend OneLoopPS {
%pythoncode {
    def __reduce__(self):
        args = self.GetLinearPS(), self.GetEpsrel()
        return self.__class__, args
}
}


class OneLoopPdd : public OneLoopPS {

public:
    
    OneLoopPdd(const PowerSpectrum& P_L, double epsrel = 1e-4);
};

class OneLoopPdv : public OneLoopPS {

public:
    
    OneLoopPdv(const PowerSpectrum& P_L, double epsrel = 1e-4);
};

class OneLoopPvv : public OneLoopPS {

public:
    
    OneLoopPvv(const PowerSpectrum& P_L, double epsrel = 1e-4);
};


class OneLoopP22Bar : public OneLoopPS {

public:
    
    OneLoopP22Bar(const PowerSpectrum& P_L, double epsrel = 1e-4); 
    
    // returns the full power spectrum
    double EvaluateFull(double k) const;
    parray EvaluateFull(const parray& k) const;
    
    // velocity kurtosis
    double VelocityKurtosis() const;
};