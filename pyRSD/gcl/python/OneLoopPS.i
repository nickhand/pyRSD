%{
#include "OneLoopPS.h"
%}

class OneLoopPS : public PowerSpectrum {
public:
    
    OneLoopPS(const PowerSpectrum& P_L, double epsrel = 1e-4);
    OneLoopPS(const PowerSpectrum& P_L, double epsrel, const parray& oneloop_power);
    
    // returns the full 1-loop power spectrum (1-loop term + linear term)
    double EvaluateFull(double k) const;
    parray EvaluateFull(const parray& k) const;
    
    const LinearPS& GetLinearPS() const;
    const double& GetRedshift() const;
    const Cosmology& GetCosmology() const;
    const double& GetEpsrel() const;
    parray GetOneLoopPower() const; 

};


class OneLoopPdd : public OneLoopPS {

public:
    
    OneLoopPdd(const PowerSpectrum& P_L, double epsrel = 1e-4);
    OneLoopPdd(const PowerSpectrum& P_L, double epsrel, const parray& oneloop_power);
};

class OneLoopPdv : public OneLoopPS {

public:
    
    OneLoopPdv(const PowerSpectrum& P_L, double epsrel = 1e-4);
    OneLoopPdv(const PowerSpectrum& P_L, double epsrel, const parray& oneloop_power);
    
};

class OneLoopPvv : public OneLoopPS {

public:
    
    OneLoopPvv(const PowerSpectrum& P_L, double epsrel = 1e-4);
    OneLoopPvv(const PowerSpectrum& P_L, double epsrel, const parray& oneloop_power);
};


class OneLoopP22Bar : public OneLoopPS {

public:
    
    OneLoopP22Bar(const PowerSpectrum& P_L, double epsrel = 1e-4); 
    OneLoopP22Bar(const PowerSpectrum& P_L, double epsrel, const parray& oneloop_power);
    
    // returns the full power spectrum
    double EvaluateFull(double k) const;
    parray EvaluateFull(const parray& k) const;
    
    // velocity kurtosis
    double VelocityKurtosis() const;
};