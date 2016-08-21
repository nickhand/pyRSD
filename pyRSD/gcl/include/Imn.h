#ifndef IMN_H
#define IMN_H
// 
//  Imn.h
//  
//  author: Nick Hand
//  contact: nhand@berkeley.edu
//  creation date: 11/23/2014 
// 

#include "Common.h"
#include "parray.h"

/*------------------------------------------------------------------------------
    Peturbation theory integrals from Vlah et al 2012
    I_nm = \int d^3q / (2\pi^3) f_nm (\vec(k), \vec{q}) P_L(q) P_L(|\vec{k} - \vec{q}|)
------------------------------------------------------------------------------*/

class Imn {
public: 
    
    Imn(const PowerSpectrum& P_L, double epsrel = 1e-3);
    
    /* Evaluate integral at a single k */
    double Evaluate(double k, int m, int n) const;
    double operator()(double k, int m, int n) const { return Evaluate(k, m, n); }
    
    /* Evaluate integral at many k values (parallelized for speed) */
    parray EvaluateMany(const parray& k, int m, int n) const;
    parray operator()(const parray& k, int m, int n) const { return EvaluateMany(k, m, n); }
    
    // accessors
    const PowerSpectrum& GetLinearPS() const { return P_L; }
    const double& GetEpsrel() const { return epsrel; }
    
protected:
    const PowerSpectrum& P_L;
    double epsrel;
   
};

#endif // IMN_H