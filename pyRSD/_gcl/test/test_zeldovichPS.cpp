// 
//  test_zeldovichPS.cpp
//  pyRSD
//  
//  author: Nick Hand
//  contact: nhand@berkeley.edu
//  creation date: 11/29/2014 
// 

#include "LinearPS.h"
#include "ZeldovichPS.h"
#include "pstring.h"
#include "Timer.h"

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
        error("Must specify which Zel'dovich spectra; one of {'P00', 'P01'}\n");
     
    pstring tag(argv[1]);
     
    // initialize the cosmology to planck base
    Cosmology cosmo("planck1_WP.ini", Cosmology::CLASS);
    
    // // initialize the linear power spectrum
    double z = 0.55;
    LinearPS linPS(cosmo, 0.);
    
    // the wavenumbers in h/Mpc
    parray k = parray::logspace(1e-2, 1e0, 500);
    
    // initialize the one loop object
    if (tag == "P00") {
        
        Timer T;
        ZeldovichP00 P(linPS);
        info("Elapsed time: %d seconds\n", T.WallTimeElapsed());
        P.SetRedshift(z);
        parray Pk = P(k);
        info("Elapsed time: %d seconds\n", T.WallTimeElapsed());
        write_results(k, Pk, tag);
    }
    else if (tag == "P01") {
        
        Timer T;
        ZeldovichP01 P(linPS);
        info("Elapsed time: %d seconds\n", T.WallTimeElapsed());
        P.SetRedshift(z);
        parray Pk = P(k);
        info("Elapsed time: %d seconds\n", T.WallTimeElapsed());
        write_results(k, Pk, tag);
    }
    
    return 0;
}
