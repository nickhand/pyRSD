// 
//  test_linearPS.cpp
//  test the LinearPS class
//  
//  author: Nick Hand
//  contact: nhand@berkeley.edu
//  creation date: 11/21/2014 
// 

#include "LinearPS.h"

#include <cstdio>
#include <string>
#include <iostream>

using namespace std;
using namespace Common;

int main(int argc, char** argv){
    
     
    // initialize the cosmology to planck base
    Cosmology cosmo("planck1_WP.ini", Cosmology::CLASS);
    info("delta_H = %f\n", cosmo.delta_H());
    
    // // initialize the no-wiggle power spectrum
    double z = 0.;
    LinearPS linPS(cosmo, z);
    
    info("Computing linear power spectrum at z = %f\n", z);
    info("Note: sigma8 = %.3f\n", cosmo.sigma8());
    
    // the wavenumbers in h/Mpc
    parray k = parray::logspace(1e-7, 100, 1000);
    
    // using CLASS transfer
    parray Pk_class = linPS(k);
    info("CLASS transfer: sigma_v = %.5f Mpc/h\n", sqrt(linPS.VelocityDispersion()));
    
    // using EH full
    cosmo.SetTransferFunction(Cosmology::EH);
    parray Pk_eh = linPS(k);
    info("EH transfer: sigma_v = %.5f Mpc/h\n", sqrt(linPS.VelocityDispersion()));
    
    // using EH no wiggle
    cosmo.SetTransferFunction(Cosmology::EH_NoWiggle);
    parray Pk_nw = linPS(k);
    info("EH no-wiggle transfer: sigma_v = %.5f Mpc/h\n", sqrt(linPS.VelocityDispersion()));
    
    // using BBKS
    cosmo.SetTransferFunction(Cosmology::BBKS);
    parray Pk_bbks = linPS(k);
    info("BBKS transfer: sigma_v = %.5f Mpc/h\n", sqrt(linPS.VelocityDispersion()));
    
    // print out the results in columns
    string filename = "data/test_linearPS.dat";
    FILE* fp = fopen(filename.c_str() , "w");
    for (size_t i = 0; i < k.size(); i++) {
        write(fp, "%.5e %.5e %.5e %.5e %.5e\n", k[i], Pk_class[i], Pk_eh[i], Pk_nw[i], Pk_bbks[i]);
    }
    fclose(fp);
    return 0;
}
