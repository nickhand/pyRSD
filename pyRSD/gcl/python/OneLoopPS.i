%{
#include "OneLoopPS.h"
%}

class OneLoopPS : public PowerSpectrum {
public:
    
    OneLoopPS(const PowerSpectrum& P_L, double epsrel = 1e-4);

    // returns full spectrum (linear + 1-loop terms)
    virtual double EvaluateFull(double k) const;
    virtual parray EvaluateFull(const parray& k) const;
    virtual double Evaluate(double k) const; 
    
    %extend {
        double __call__(double k) const { return $self->Evaluate(k); }
        parray __call__(const parray& k) const { return $self->EvaluateMany(k); }
    };
        
    const PowerSpectrum& GetLinearPS() const;
    const double& GetRedshift() const;
    const Cosmology& GetCosmology() const;
};

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
    
    double EvaluateFull(double k) const;
    parray EvaluateFull(const parray& k) const;
    double VelocityKurtosis() const;
};