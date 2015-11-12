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
    
    double EvaluateTransfer(double k) const;
    
};

%extend Cosmology {
    %pythoncode {
        
        @classmethod
        def from_power(cls, param_file, pkfile):
            return cls.FromPower(param_file, pkfile)
            
        @classmethod
        def from_file(cls, param_file, tkfile):
            return cls(param_file, tkfile)
            
        def __getitem__(self, key):
            if hasattr(self, key):
                f = getattr(self, key)
                if callable(f):
                    return f()
            raise KeyError("Sorry, cannot return parameter '%s' in dict-like fashion" %key)
        
        def __setstate__(self, state):
            self.__init__(*state['args'])

        def __getstate__(self):
            args = self.GetParamFile(), self.GetTransferFit(), self.GetTransferFile(), self.GetPrecisionFile()
            return {'args': args}
    }
}
