%{
#include "Cosmology.h"
%}

class Cosmology : public ClassCosmology {

public:    
    enum TransferFit {CLASS, EH, EH_NoWiggle, BBKS, FromFile};
    
    Cosmology(const ClassParams& pars, TransferFit tf = CLASS, const std::string& tkfile = "", const std::string& precision_file = "");
    Cosmology(const std::string& param_file, TransferFit tf = CLASS, const std::string& tkfile = "", const std::string& precision_file = "");
    ~Cosmology();

    void SetTransferFunction(TransferFit tf, const std::string& tkfile = "");
    void LoadTransferFunction(const std::string& tkfile, int kcol = 1, int tcol = 2);
    void NormalizeTransferFunction(double sigma8);
    void SetSigma8(double sigma8);
    
    // parameter accessors
    double A_s() const;
    double ln_1e10_A_s() const;
    double delta_H() const;
    double sigma8() const;
    
    TransferFit transfer_fit() const;
    double EvaluateTransfer(double k) const;

};
