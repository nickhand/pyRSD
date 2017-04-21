// 
//  test_oneloopPS.cpp
//  test the one loop PS
//  
//  author: Nick Hand
//  contact: nhand@berkeley.edu
//  creation date: 11/25/2014 
// 

#include "LinearPS.h"
#include "OneLoopPS.h"
#include "pstring.h"

#include <cstdio>
#include <string>
#include <iostream>

using namespace std;
using namespace Common;

void write_results(const parray& k, const parray& Pk, const pstring& tag) {
    
    // print out the results in columns
    string filename = "data/test_" + tag + ".dat";
    FILE* fp = fopen(filename.c_str() , "w");
    for (size_t i = 0; i < k.size(); i++) {
        write(fp, "%.5e %.5e\n", k[i], Pk[i]);
    }
    fclose(fp);
    
}

int main(int argc, char** argv){
    
    // make sure we have the write number of arguments 
    if (argc != 2)
        error("Must specify which 1-loop spectra; one of {'Pdd', 'Pdv', 'Pvv', 'P22bar'}\n");
     
    pstring tag(argv[1]);
     
    // initialize the cosmology to planck base
    Cosmology cosmo("planck1_WP.ini", Cosmology::CLASS);
    info("delta_H = %f\n", cosmo.delta_H());
    
    // // initialize the linear power spectrum
    double z = 0.;
    LinearPS linPS(cosmo, z);
    
    info("Computing linear power spectrum at z = %f\n", z);
    info("Note: sigma8 = %.3f\n", cosmo.sigma8());
    
    // the wavenumbers in h/Mpc
    parray k = parray::logspace(1e-5, 1e1, 500);
    
    // initialize the one loop object
    if (tag == "Pdd") {
        
        OneLoopPdd P_1loop(linPS, 1e-4);
        parray Pk = P_1loop.EvaluateFull(k);
        write_results(k, Pk, tag);
    }
    else if (tag == "Pdv") {
        
        OneLoopPdv P_1loop(linPS, 1e-4);
        parray Pk = P_1loop.EvaluateFull(k);
        write_results(k, Pk, tag);
    }
    else if (tag == "Pvv") {
        
        OneLoopPvv P_1loop(linPS, 1e-4);
        parray Pk = P_1loop.EvaluateFull(k);
        write_results(k, Pk, tag);
    } else if (tag == "P22bar") {
        
        OneLoopP22Bar P_1loop(linPS, 1e-4);
        parray Pk = P_1loop.EvaluateFull(k);
        info("velocity kurtosis = %.4f\n", P_1loop.VelocityKurtosis());
        write_results(k, Pk, tag); 
    }
    
    return 0;
}
