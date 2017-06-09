#include "NonlinearPS.h"

using namespace Common;
using namespace std;

extern "C" void emu_noh(double *xstar, double *ystar, int *outtype);
extern "C" void emu(double *xstar, double *ystar, int *outtype);
extern "C" void getH0fromCMB(double *xstar, double *stuff);

/*----------------------------------------------------------------------------*/
NonlinearPS::NonlinearPS() {}

NonlinearPS::NonlinearPS(const string& param_file, double z_,
                            NonlinearFit fit, bool use_cmbh_) {

    // make the class params first and update the nonlinear value
    ClassParams pars(param_file);
    pars.add("non linear", "halofit");

    // initialize the private variables
    C.compute(pars);
    z = z_;
    nonlinear_fit = fit;
    use_cmbh = use_cmbh_;
    h_ = C.h();

    // initialize the FrankenEmu spline
    if (nonlinear_fit == FrankenEmu)
        InitializeTheFrankenEmu();
}

/*----------------------------------------------------------------------------*/
parray NonlinearPS::EvaluateMany(const parray& k) {
    if (nonlinear_fit == Halofit)
        return C.GetPknl(k, z);
    else if (nonlinear_fit == FrankenEmu) {
        if ( (k.min() < spline_kmin) || (k.max() > spline_kmax) )
            error("Can only compute P(k) for k between [%.2e, %.2e] h/Mpc", spline_kmin, spline_kmax);
        return emu_spline(k);
    }
    return k*0.; // this should never happen
}

/*----------------------------------------------------------------------------*/
double NonlinearPS::Evaluate(double k) {
    parray k_(1);
    k_[0] = k;
    return EvaluateMany(k_)[0];
}

/*----------------------------------------------------------------------------*/
void NonlinearPS::InitializeTheFrankenEmu() {

    // first verify the cosmo params
    VerifyFrankenEmu();

    size_t N(582);
    int type(2);
    double xstar[7], stuff[4], xstarcmb[6], ystar[2*N];

    // the cosmo params
    xstar[0] = C.Omega0_b()*pow(h_, 2);
    xstar[1] = C.Omega0_m()*pow(h_, 2);
    xstar[2] = C.n_s();
    if (!use_cmbh) xstar[3] = 100*h_;
    xstar[4] = C.w0_fld();
    xstar[5] = C.sigma8();
    xstar[6] = z;

    // call the emulator
    if(use_cmbh) {
        xstarcmb[0] = xstar[0];
        xstarcmb[1] = xstar[1];
        xstarcmb[2] = xstar[2];
        xstarcmb[3] = xstar[4];
        xstarcmb[4] = xstar[5];
        xstarcmb[5] = xstar[6];
        emu_noh(xstarcmb, ystar, &type);
        getH0fromCMB(xstarcmb, stuff);
        h_ = stuff[3];
        xstar[3] = 100.*stuff[3];
        verbose("FrankenEmu: H0 from CMB constraints = %.1f", xstar[3]);
    } else {
        emu(xstar, ystar, &type);
    }

    // extract k, Pk
    vector<double> k, Pk;
    for (size_t i=0; i<N; i++){
        k.push_back(ystar[i]/h_);
        Pk.push_back(ystar[N+i]*pow(h_, 3));
    }

    // setup the spline for k, Pk
    spline_kmin = k[0];
    spline_kmax = k[N-1];
    emu_spline = CubicSpline(k, Pk);

}

/*----------------------------------------------------------------------------*/
void NonlinearPS::VerifyFrankenEmu() {

    double ombh2 = C.Omega0_b()*pow(h(), 2);
    double ommh2 = C.Omega0_m()*pow(h(), 2);
    double ns = C.n_s();
    double sigma8 = C.sigma8();
    double w = C.w0_fld();
    double H0 = 100.*h();

    // omega_m h^2
    if ( (ommh2 < 0.120) || (ommh2 > 0.155) )
        error("Omega_m h^2 must be between [0.12, 0.155] to use the FrankenEmu");

    // omega_b h^2
    if ( (ombh2 < 0.0215) || (ombh2 > 0.0235) )
        error("Omega_b h^2 must be between [0.0215, 0.0235] to the FrankenEmu");

    // n_s
    if ( (ns < 0.85) || (ns > 1.05) )
        error("Spectral index must be between [0.85, 1.05] to use the FrankenEmu");

    // sigma_8
    if ( (sigma8 < 0.61) || (sigma8 > 0.9) )
        error("Sigma_8 must be between [0.61, 0.9] to use the FrankenEmu");

    // w
    if ( (w < -1.30) || (w > -0.70) )
        error("Dark energy equation of state must be between [-1.3, -0.7] to use the FrankenEmu");

    // redshift
    if ( (z < 0.) || (z > 1.) )
        error("Redshift must be between [0, 1] to use the FrankenEmu");

    // H0
    if (!use_cmbh) {
        if ((H0 < 55.) || (H0 > 85.0))
            error("H0 must be between [55, 85] to use the FrankenEmu");
    }
}
