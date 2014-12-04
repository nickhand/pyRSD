//--------------------------------------------------------------------------
//
// Description:
// 	class Engine : see header file (Engine.hh) for description.
//
//------------------------------------------------------------------------

#include "Engine.h"

#include <numeric>
#include <iostream>
#include <stdexcept>

using namespace std;


// constructor
Engine::Engine():_lmax(-1) {}
Engine::~Engine() {}
/*----------------------------------------------------------------------------*/

// write out the Cls, in units of uK^2
void Engine::WriteCls(std::ostream &of){

    int status; 
    vector<unsigned> lvec(_lmax-1, 1);
    lvec[0] = 2;
    partial_sum(lvec.begin(), lvec.end(), lvec.begin());
      
    parray cltt, clte, clee, clbb, clpp, cltp, clep;
    
    // compute the Cls
    status = GetCls(lvec, cltt, clte, clee, clbb);
    if (status > 0) throw runtime_error("Failed to compute Cls");
    
    // and the lensing Cls
    status = GetLensing(lvec, clpp, cltp, clep);   
    
    of.precision(16);
    for (size_t i=0; i < lvec.size(); i++) {
        of << lvec[i] << "\t" 
           << cltt[i] << "\t" 
           << clte[i] << "\t" 
           << clee[i] << "\t" 
           << clbb[i];
       if (status == 0)
           of << "\t" << clpp[i] << "\t" << cltp[i] << "\t" << clep[i];
       of << endl;
   }   
}
/*----------------------------------------------------------------------------*/

// write out Pk in units of h/Mpc, (Mpc/h)^3
void Engine::WriteMatterSpectrum(std::ostream &of, double z, pktype t) {

    int status(0); 
    parray k, output;

    // get the computed spectrum
    if (t == Pklin) {
        status = GetPklin(z, k, output);
    } else if (t == Pknl) {
        status = GetPknl(z, k, output);
    } else if (t == Tk) {
        status = GetTk(z, k, output);
    }
    if (status > 0) throw runtime_error("Failed to compute spectrum");

    // write it out
    of.precision(16);
    for (size_t i=0; i < k.size(); i++) {
        of << k[i] << "\t" << output[i] << "\t" << endl;
       of << endl;
   }   
}
/*----------------------------------------------------------------------------*/

void Engine::WritePklin(std::ostream &of, double z) {
    WriteMatterSpectrum(of, z, Pklin);
}

/*----------------------------------------------------------------------------*/

void Engine::WritePknl(std::ostream &of, double z) {
    WriteMatterSpectrum(of, z, Pknl);
}

/*----------------------------------------------------------------------------*/

void Engine::WriteTk(std::ostream &of, double z) {
    WriteMatterSpectrum(of, z, Tk);
}
/*----------------------------------------------------------------------------*/

// return the computed Cls at the desired ell values
int Engine::GetCls(const vector<unsigned>&, parray&, parray&, parray&, parray&) {
    return 1;
}
/*----------------------------------------------------------------------------*/
 
// return the computed lensing spectra the desired ell values
int Engine::GetLensing(const vector<unsigned>&, parray&, parray&, parray&) {
    return 1;        
}


/*----------------------------------------------------------------------------*/

// compute the k, linear Pk in units of h/Mpc, (Mpc/h)^3
int Engine::GetPklin(double, const parray&, parray&) {
    return 1;
}

/*----------------------------------------------------------------------------*/

// compute the k, nonlinear Pk in units of h/Mpc, (Mpc/h)^3
int Engine::GetPknl(double, const parray&, parray&) {
    return 1;
}

/*----------------------------------------------------------------------------*/

// return the k, transfer function (in CAMB format) in units of h/Mpc, unitless
int Engine::GetTk(double, const parray&, parray&) {
    return 1;
}
