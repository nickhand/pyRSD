// 
//  test_ImnOneLoop.cpp
//  
//  author: Nick Hand
//  contact: nhand@berkeley.edu
//  creation date: 11/26/2014 
// 

#include <iostream>
#include <cstdio>
#include <string>

#include "ImnOneLoop.h"
#include "OneLoopPS.h"
#include "LinearPS.h"
#include "pstring.h"
#include "Timer.h"

using namespace std;
using namespace Common;

int main(int argc, char** argv){
    
    // make sure we have the write number of arguments 
    // if (argc != 3)
    //     error("Must specify integral indices (m, n) as two command line arguments\n");
    //     
    // pstring m_str = pstring(argv[1]);
    // pstring n_str = pstring(argv[2]);
    // double m = double(m_str);
    // double n = double(n_str);
    
    // initialize the cosmology
    Cosmology cosmo("planck1_WP.ini", Cosmology::CLASS);
    
    // initialize the linear power spectrum
    double z = 0.;
    LinearPS linPS(cosmo, z);
    
    // iniitalize the 1-loop spectra
    OneLoopPdd Pdd(linPS, 1e-4);
    OneLoopPdv Pdv(linPS, 1e-4);
    OneLoopPvv Pvv(linPS, 1e-4);
 
    // the wavenumbers to compute in h/Mpc
    int Nout = 1000;
    double kminout = 1e-3;
    double kmaxout = 1.;
    parray k = parray::logspace(kminout, kmaxout, Nout);
    
    // time the computation
    //ImnOneLoop Ivvdd(Pvv, Pdd, 1e-4);
    ImnOneLoop Idvdv(Pdv, 1e-4);
   
    //parray out = Idvdv.EvaluateLinear(k, 0, 3) + Idvdv.EvaluateCross(k, 0, 3) + Idvdv.EvaluateOneLoop(k, 0, 3);
    parray out = Idvdv.EvaluateLinear(k, 0, 4) + Idvdv.EvaluateCross(k, 0, 4) + Idvdv.EvaluateOneLoop(k, 0, 4);
    
    //parray out = Ivvdd.EvaluateLinear(k, 0, 1) + Ivvdd.EvaluateCross(k, 0, 1) + Ivvdd.EvaluateOneLoop(k, 0, 1);
    //parray out = Ivvdd.EvaluateLinear(k, 0, 2) + Ivvdd.EvaluateCross(k, 0, 2) + Ivvdd.EvaluateOneLoop(k, 0, 2);
   
    // // write out the results
    // string filename = "data/test.dat";
    // FILE* fp = fopen(filename.c_str() , "w");
    for (int i = 0; i < Nout; i++){
        info("%.5e %.5e\n", k[i], out[i]);
    }
    //fclose(fp);
    return 0;
}
