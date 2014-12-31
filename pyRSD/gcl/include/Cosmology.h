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
    // construct directly from a parameter file
    Cosmology(const std::string& param_file, TransferFit tf = CLASS, const std::string& tkfile = "", const std::string& precision_file = "");
    
    ~Cosmology();
    
    // functions for setting the type of transfer TransferFit to use
    void SetTransferFunction(TransferFit tf, const std::string& tkfile = "");
    
    // load the transfer file from file
    void LoadTransferFunction(const std::string& tkfile, int kcol = 1, int tcol = 2);
    
    // normalize the transfer function
    void NormalizeTransferFunction(double sigma8);
    
    // set the sigma8 value
    inline void SetSigma8(double sigma8) { sigma8_ = sigma8; NormalizeTransferFunction(sigma8); }
    
    // normalization of linear power spectrum at z = 0 
    inline double delta_H() const { return delta_H_; }
    
    // return the sigma8 that we normalized too
    inline double sigma8() const { return sigma8_; }
    
    // return the scalar amplitude values for the sigma8 we normalized to
    inline double A_s() const { return ClassCosmology::A_s() * Common::pow2(sigma8_/ClassCosmology::sigma8()); }
    inline double ln_1e10_A_s() const { return log(1e10*A_s()); }
    
    // accessors
    inline TransferFit GetTransferFit() const { return transfer_fit_; }
    inline const std::string& GetParamFile() const { return param_file_; }
    inline const std::string& GetTransferFile() const { return transfer_file_; }
    inline const std::string& GetPrecisionFile() const { return precision_file_; }
    
    // evaluate at k in h/Moc
    double EvaluateTransfer(double k) const;
    
    
private:
        
    double sigma8_;      /* power spectrum variance smoothed at 8 Mpc/h */
    double delta_H_;     /* normalization of linear power spectrum at z = 0 */
    TransferFit transfer_fit_;   /* the transfer fit method */
    std::string param_file_, transfer_file_, precision_file_;
    
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

