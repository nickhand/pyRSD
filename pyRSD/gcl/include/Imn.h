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
    
    Imn(const PowerSpectrum& P_L, double epsrel = 1e-5);
    
    /* Evaluate integral at a single k */
    double Evaluate(int m, int n, double k) const;
    double operator()(int m, int n, double k) const { return Evaluate(m, n, k); }
    
    /* Evaluate integral at many k values (parallelized for speed) */
    parray EvaluateMany(int m, int n, const parray& k) const;
    parray operator()(int m, int n, const parray& k) const { return EvaluateMany(m, n, k); }
    
private:
    const PowerSpectrum& P_L;
    double epsrel;
   
};

#endif // IMN_H