#ifndef KMN_H
#define KMN_H
// 
//  Kmn.h
//  
//  author: Nick Hand
//  contact: nhand@berkeley.edu
//  creation date: 11/23/2014 
// 

#include "Common.h"
#include "parray.h"

/*------------------------------------------------------------------------------
    Peturbation theory integrals from Vlah et al 2013 for halo biasing model
------------------------------------------------------------------------------*/

class Kmn  {
public: 
    
    Kmn(const PowerSpectrum& P_L, double epsrel = 1e-3);
    
    /* Evaluate integral at a single k */
    double Evaluate(double k, int m, int n, bool tidal=false, int part=0) const;
    double operator()(double k, int m, int n, bool tidal=false, int part=0) const { return Evaluate(k, m, n, tidal, part); }
    
    /* Evaluate integral at many k values (parallelized for speed) */
    parray EvaluateMany(const parray& k, int m, int n, bool tidal=false, int part=0) const;
    parray operator()(const parray& k, int m, int n, bool tidal=false, int part=0) const { return EvaluateMany(k, m, n, tidal, part); }
 
    const PowerSpectrum& GetLinearPS() const { return P_L; }
    const double& GetEpsrel() const { return epsrel; }
    
protected:
    const PowerSpectrum& P_L;
    double epsrel;  
};


#endif // KMN_H