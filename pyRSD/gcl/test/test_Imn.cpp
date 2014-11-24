// 
//  test_Imn.cpp
//  testing of the Imn class
//  
//  author: Nick Hand
//  contact: nhand@berkeley.edu
//  creation date: 11/23/2014 
// 

#include "Imn.h"
#include "linearPS.h"
#include "pstring.h"
#include "Timer.h"

#include <iostream>
using namespace std;

int main(int argc, char** argv){
    

    double m = double(pstring(argv[1]));
    double n = double(pstring(argv[2]));
    
    // initialize the cosmology
    Cosmology cosmo("explanatory.ini", Cosmology::CLASS);
    
    // initialize the no-wiggle power spectrum
    double z = 0.;
    LinearPS linPS(cosmo, z);
 
    int Nout = 1000;
    double kminout = 1e-5;
    double kmaxout = 10.;
    
    // compute Imn at each k
    parray k = parray::logspace(kminout, kmaxout, Nout);
    
    // do the computation
    Imn I(linPS);
    
    Timer T;
    parray out = I(int(m), int(n), k);
    info("Elapsed time: %d seconds\n", T.WallTimeElapsed());
    
    for (int i = 0; i < Nout; i++){
        info("%.4e %.4e\n", k[i], out[i]);
    }
    return 0;
}
