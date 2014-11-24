// 
//  test_linearPS.cpp
//  test the LinearPS class
//  
//  author: Nick Hand
//  contact: nhand@berkeley.edu
//  creation date: 11/21/2014 
// 

#include "LinearPS.h"
#include <iostream>
using namespace std;

int main(int argc, char** argv){
    
    
    // initialize the cosmology
    Cosmology cosmo("explanatory.ini", Cosmology::CLASS);
    info("delta_H = %f\n", cosmo.delta_H());
    
    // initialize the no-wiggle power spectrum
    double z = 1.0;
    LinearPS linPS(cosmo, z);
    
    cosmo.NormalizeTransferFunction(0.8);
    
    info("Computing linear power spectrum at z = %f\n", z);
    info("Note: sigma8 = %.3f\n", cosmo.sigma8());
    
    // the wavenumbers
    parray k = parray::logspace(1e-3, 1.0, 100);
    
    for (size_t i = 0; i < k.size(); i++){
        double Pk = linPS.Evaluate(k[i]);
        cout << k[i] << "\t" << Pk << endl;
    }
    
    return 0;
}
