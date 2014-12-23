#ifndef JMN_H
#define JMN_H
// 
//  Jmn.h
//  gcl
//  
//  author: Nick Hand
//  contact: nhand@berkeley.edu
//  creation date: 11/25/2014 
// 

#include "Common.h"
#include "parray.h"

/*------------------------------------------------------------------------------
    Peturbation theory integrals from Vlah et al 2012
    I_nm = \int d^3q / (2\pi^3) g_nm (k/q) P_L(q) /q^2
------------------------------------------------------------------------------*/

class Jmn {
public: 
    
    Jmn(const PowerSpectrum& P_L, double epsrel = 1e-5);
    
    /* Evaluate integral at a single k */
    double Evaluate(double k, int m, int n) const;
    double operator()(double k, int m, int n) const { return Evaluate(k, m, n); }
    
    /* Evaluate integral at many k values (parallelized for speed) */
    parray EvaluateMany(const parray& k, int m, int n) const;
    parray operator()(const parray& k, int m, int n) const { return EvaluateMany(k, m, n); }
    
    const PowerSpectrum& GetLinearPS() const { return P_L; }
    const double& GetEpsrel() const { return epsrel; }
    
private:
    const PowerSpectrum& P_L;
    double epsrel;
   
};

#endif // JMN_H