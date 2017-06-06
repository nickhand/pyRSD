#ifndef NONLINEAR_PS_H
#define NONLINEAR_PS_H


#include "Common.h"
#include "parray.h"
#include "ClassEngine.h"
#include "Spline.h"

/**********************************************************************
 * NonlinearPS
 *
 * nonlinear spectrum, either computed via CLASS using Halofit (with
 * the updates) or from the Coyote Emulator
 *********************************************************************/
class NonlinearPS  {
public:

    enum NonlinearFit {FrankenEmu=0,    /* FrankenEmu emulator */
                       Halofit          /* Halofit via CLASS */
    };

    NonlinearPS();
    NonlinearPS(const std::string& param_file, double z = 0,
                NonlinearFit fit = FrankenEmu, bool use_cmbh = false);

    /* Evaluate power spectrum at a single k */
    double Evaluate(double k);
    double operator()(double k) { return Evaluate(k); }

    /* Evaluate power spectrum at many k values (parallelized for speed) */
    parray EvaluateMany(const parray& k);
    parray operator()(const parray& k) { return EvaluateMany(k); }

    // accessors
    inline double h() const { return h_; }
    inline const double& GetRedshift() const { return z; }
    inline ClassEngine& GetCosmology() { return C; }
    inline NonlinearFit GetNonlinearFit() const { return nonlinear_fit; }

private:

    ClassEngine C;
    double z;
    NonlinearFit nonlinear_fit;
    bool use_cmbh;
    double spline_kmin, spline_kmax;
    Spline emu_spline;
    double h_;

    // initialize the spline for the FrankenEmu
    void InitializeTheFrankenEmu();

    // verify that the cosmological parameters are in the right range
    void VerifyFrankenEmu();

};

#endif // NONLINEAR_PS_H
