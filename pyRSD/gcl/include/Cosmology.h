// 
//  Cosmology.h
//  
//  author: Nick Hand
//  contact: nhand@berkeley.edu
//  creation date: 11/21/2014 
// 

#ifndef COSMOLOGY_H
#define COSMOLOGY_H

#include "ClassCosmology.h"
#include "parray.h"
#include "Spline.h"

/*----------------------------------------------------------------------------*/
/* class to encapsulate add Transfer function tracking to ClassCosmology  */
/*----------------------------------------------------------------------------*/
class Cosmology : public ClassCosmology
{

public:

    enum TransferFit {CLASS=0,          /* calls CLASS to compute */ 
                      EH,               /* Eisenstein & Hu 1998 (astro-ph/9709112) */
                      EH_NoWiggle,        
                      BBKS,             /* Bardeen, Bond, Kaiser, Szalay 1986*/
                      FromFile          /* loaded from file */
    };
    
        
    Cosmology();
    // construct from a ClassParams object
    Cosmology(const ClassParams& pars, TransferFit tf = CLASS, const std::string& tkfile = "", const std::string& precision_file = "");
    // construct directly from a parameter file
    Cosmology(const std::string& param_file, TransferFit tf = CLASS, const std::string& tkfile = "", const std::string& precision_file = "");
    
    ~Cosmology();
    
    // functions for setting the type of transfer TransferFit to use
    void SetTransferFunction(TransferFit tf, const std::string& tkfile = "");
    
    // load the transfer file from file
    void LoadTransferFunction(const std::string& tkfile, int kcol = 1, int tcol = 2);
    
    // normalize the transfer function
    void NormalizeTransferFunction(double sigma8);
    
    // normalization of linear power spectrum at z = 0 
    inline double delta_H() const { return delta_H_; }
    
    // return the sigma8 that we normalized too
    inline double sigma8() const { return sigma8_; }
    
    // the type of transfer fit
    inline TransferFit transfer_fit() const { return transfer_fit_; }
    
    // evaluate at k in h/Moc
    double EvaluateTransfer(double k) const;
    
    
private:
        
    double sigma8_;              /* power spectrum variance smoothed at 8 Mpc/h */
    double delta_H_;             /* normalization of linear power spectrum at z = 0 */
    TransferFit transfer_fit_;   /* the transfer fit method */
    
    parray ki, Ti;
    double k0, T0, T0_nw;        // k and T(k) of left-most data point
    double k1, T1, T1_nw;        // k and T(k) of right-most data point
    Spline Tk;                   // spline T(k) based on transfer function
    
    double f_baryon;
    double k_equality, sound_horizon, beta_c, alpha_c;
    double beta_node, alpha_b, beta_b, k_silk;
    double alpha_gamma, s;
    
    double GetEisensteinHuTransfer(double k) const;
    double GetNoWiggleTransfer(double k) const;
    double GetBBKSTransfer(double k) const;
    double GetSplineTransfer(double k) const;
    void InitializeTransferFunction(TransferFit tf, const std::string& tkfile);
    void SetEisensteinHuParameters();
        
};


;
#endif

