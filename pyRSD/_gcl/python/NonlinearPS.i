%{
#include "NonlinearPS.h"
%}

class NonlinearPS  {
public:
    enum NonlinearFit {FrankenEmu, Halofit};

    NonlinearPS();
    NonlinearPS(const std::string& param_file, double z = 0,
                NonlinearFit fit = FrankenEmu, bool use_cmbh = false);

    double Evaluate(double k);
    double operator()(double k);
    parray EvaluateMany(const parray& k);
    parray operator()(const parray& k);

    const double h() const;
    const double& GetRedshift() const;
    ClassEngine& GetCosmology();
    NonlinearFit GetNonlinearFit() const;
};
