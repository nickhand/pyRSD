#ifndef ONELOOP_PS_H
#define ONELOOP_PS_H

#include "Cosmology.h"
#include "Imn.h"
#include "Jmn.h"
#include "PowerSpectrum.h"

/*------------------------------------------------------------------------------
    OneLoopPS
    ---------
    Standard Perturbation Theory (SPT_ 1-loop power spectrum, for density-density
    density-velocity, and velocity velocity spectra

    Computes spectra from KMIN to KMAX and returns the value at a given k, 
    using a Spline
------------------------------------------------------------------------------*/

// base class for 1-loop PS
class OneLoopPS : public PowerSpectrum {
public:
    
    OneLoopPS(const PowerSpectrum& P_L, double epsrel = 1e-4);
    OneLoopPS(const PowerSpectrum& P_L, double epsrel, const parray& oneloop_power);
    
    // returns full spectrum (linear + 1-loop terms)
    virtual double EvaluateFull(double k) const;
    virtual parray EvaluateFull(const parray& k) const;
    double Evaluate(double k) const; 
        
    const PowerSpectrum& GetLinearPS() const { return P_L; }
    const double& GetRedshift() const { return P_L.GetRedshift(); }
    const Cosmology& GetCosmology() const { return P_L.GetCosmology(); }
    const double& GetEpsrel() const { return epsrel; }
    parray GetOneLoopPower() const { return oneloop_power; }
    
    
protected:
    
    const PowerSpectrum& P_L;
    double epsrel;
    parray oneloop_power; 
    Spline oneloop_spline; 
    
};

// density-density 1-loop power spectrum
class OneLoopPdd : public OneLoopPS {

public:
    
    OneLoopPdd(const PowerSpectrum& P_L, double epsrel = 1e-4);
    OneLoopPdd(const PowerSpectrum& P_L, double epsrel, const parray& oneloop_power);

private:
    
    // this sets the 1-loop spline to 2*I00 + 6*k^2 * J00 * P_lin(k)
    void InitializeSpline();
};

// density-velocity 1-loop power spectrum
class OneLoopPdv : public OneLoopPS {

public:
    
    OneLoopPdv(const PowerSpectrum& P_L, double epsrel = 1e-4);
    OneLoopPdv(const PowerSpectrum& P_L, double epsrel, const parray& oneloop_power); 
    
private:
    
    // this sets the 1-loop spline to 2*I01 + 6*k^2 * J01 * P_lin(k)
    void InitializeSpline();

};

// velocity-velocity 1-loop power spectrum
class OneLoopPvv : public OneLoopPS {

public:
    
    OneLoopPvv(const PowerSpectrum& P_L, double epsrel = 1e-4); 
    OneLoopPvv(const PowerSpectrum& P_L, double epsrel, const parray& oneloop_power);

private:
    
    // this sets the 1-loop spline to 2*I11 + 6*k^2 * J11 * P_lin(k)
    void InitializeSpline();

};

// \bar{P22} 1-loop power spectrum, which is basically < vpar^2 | vpar^2 > 
class OneLoopP22Bar : public OneLoopPS {

public:
    
    OneLoopP22Bar(const PowerSpectrum& P_L, double epsrel = 1e-4); 
    OneLoopP22Bar(const PowerSpectrum& P_L, double epsrel, const parray& oneloop_power);
    
    // dont need P_L term here
    double EvaluateFull(double k) const;
    parray EvaluateFull(const parray& k) const;
    
    /* Calculate the 1-D velocity kurtosis $\sigma_v^4 = \frac{1}{4} \int_0^\infty P22bar(q)/q^2 ~dq$ */
    double VelocityKurtosis() const;

private:
    
    // this sets the 1-loop spline to I23 + (2/3)*I32 + (1/5)*I33
    void InitializeSpline();

};

#endif // ONELOOP_PS_H
