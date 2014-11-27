%{
#include "PowerSpectrum.h"
%}

class PowerSpectrum {
public:
    PowerSpectrum();
    virtual ~PowerSpectrum();

    %extend {
        double __call__(double k) const { return $self->Evaluate(k); }
        parray __call__(const parray& k) const { return $self->EvaluateMany(k); }
    };

    const Cosmology& GetCosmology() const = 0;

    double Sigma(double R) const;
    double VelocityDispersion() const;
    double NonlinearScale() const;
    void Save(const char* filename, double kmin = 1e-3, double kmax = 1, int Nk = 1001, bool log = false);
};

