#ifndef IMN_ONELOOP_H
#define IMN_ONELOOP_H
// 
//  ImnOneLoop.h
//  
//  author: Nick Hand
//  contact: nhand@berkeley.edu
//  creation date: 11/26/2014 
// 

#include "Common.h"
#include "parray.h"

/*------------------------------------------------------------------------------
    Peturbation theory integrals from Vlah et al 2012
    
    I_nm = \int d^3q / (2\pi^3) f_nm (\vec(k), \vec{q}) P_X(q) P_X(|\vec{k} - \vec{q}|),
    where P_X can be of linear order or 1-loop over
------------------------------------------------------------------------------*/

class ImnOneLoop {
public: 
    
    // both 1-loop spectra are the same
    ImnOneLoop(const OneLoopPS& P_1, double epsrel = 1e-4);
    ImnOneLoop(const OneLoopPS& P_1, const OneLoopPS& P_2, double epsrel = 1e-4);
    
    /* Evaluate the linear-linear term for a given kernel (scales as D(z)^4) */
    double EvaluateLinear(double k, int m, int n) const;
    parray EvaluateLinear(const parray& k, int m, int n) const;
    
    /* Evaluate the sum of 1loop-linear and linear-1loop terms for a given kernel (scales as D(z)^6) */
    double EvaluateCross(double k, int m, int n) const;
    parray EvaluateCross(const parray& k, int m, int n) const;
    
    /* Evaluate the 1loop - 1loop term for a given kernel (scales as D(z)^8) */
    double EvaluateOneLoop(double k, int m, int n) const;
    parray EvaluateOneLoop(const parray& k, int m, int n) const;
    
    // accessors
    const OneLoopPS& GetOneLoopPS1() const { return P_1; }
    const OneLoopPS& GetOneLoopPS2() const { return P_2; }
    const double& GetEpsrel() const { return epsrel; }
    const bool& GetEqual() const { return equal; }
    
protected:
    const OneLoopPS& P_1;
    double epsrel;
    bool equal;
    const OneLoopPS& P_2;
    const PowerSpectrum& P_L;
};

#endif // IMN_ONELOOP_H