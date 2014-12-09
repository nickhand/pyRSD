// 
//  test_Imn.cpp
//  testing of the Imn class
//  
//  author: Nick Hand
//  contact: nhand@berkeley.edu
//  creation date: 11/23/2014 
// 
#include <iostream>
#include <cstdio>
#include <string>

#include "Kmn.h"
#include "linearPS.h"
#include "pstring.h"
#include "Timer.h"

using namespace std;
using namespace Common;

int main(int argc, char** argv){
    
    // make sure we have the write number of arguments 
    if (argc < 3)
        error("Must specify integral indices (m, n) as two command line arguments\n");
        
    pstring m_str = pstring(argv[1]);
    pstring n_str = pstring(argv[2]);
    double m = double(m_str);
    double n = double(n_str);
    
    // initialize the cosmology
    Cosmology cosmo("planck1_WP.ini", Cosmology::CLASS);
    
    // initialize the linear power spectrum
    double z = 0.;
    LinearPS linPS(cosmo, z);
 
    // the wavenumbers to compute in h/Mpc
    int Nout = 1000;
    double kminout = 1e-5;
    double kmaxout = 10.;
    parray k = parray::logspace(kminout, kmaxout, Nout);
    
    // time the computation
    Kmn K(linPS, 1e-4);
    Timer T;
    parray out;
    
    bool tidal = false;
    int part = 0;
    string tidal_str = "";
    string part_str = "";
    
    // determine other K indices
    if (argc > 3) {
        if (string(argv[3]) == "1") {
            tidal = true;
            tidal_str = "s";  
        } 
    }
           
    if (argc > 4) {
        if (m == 2 && n == 0) {
            if (string(argv[4]) == "0") {
                part = 0;
                part_str = "_a";
            } else {
                part = 1;
                part_str = "_b";
            }
        }
    }    
    
    out = K(k, int(m), int(n), tidal, part);
    info("Elapsed time: %d seconds\n", T.WallTimeElapsed());
        
    // write out the results
    string filename = "data/K" + m_str + n_str + tidal_str + part_str + ".dat";
    FILE* fp = fopen(filename.c_str() , "w");
    for (int i = 0; i < Nout; i++){
        write(fp, "%e %e\n", k[i], out[i]);
    }
    fclose(fp);
    return 0;
}
