%{
#include "Cosmology.h"
%}

class Cosmology : public ClassCosmology {

public:    
    enum TransferFit {CLASS, EH, EH_NoWiggle, BBKS};
    
    // defaults transfer fit to CLASS
    Cosmology(const std::string& param_file);
    // specify other transfer fit
    Cosmology(const std::string& param_file, TransferFit tf);
    // specify transfer file to read from
    Cosmology(const std::string& param_file, const std::string& tkfile);
    Cosmology(const std::string& param_file, TransferFit tf, double sigma8, 
                const std::string& tkfile, const parray& k, const parray& Tk);
    // from power spectrum file
    static Cosmology* FromPower(const std::string& param_file, const std::string& pkfile);
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
    TransferFit GetTransferFit() const;
    const std::string& GetParamFile() const;
    const std::string& GetTransferFile() const;
    parray GetDiscreteK() const;
    parray GetDiscreteTk() const;
    
    double EvaluateTransfer(double k) const;
    
};

