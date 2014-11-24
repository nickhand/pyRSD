#include <unistd.h>
#include <functional>

#include "Cosmology.h"
#include "Datafile.h"
#include "SpecialFunctions.h"
#include "pstring.h"
#include "Quadrature.h"

using std::bind;
using std::cref;
using namespace std::placeholders;

Cosmology::Cosmology() {
    /* No initialization */
}

Cosmology::Cosmology(const ClassParams& pars,  TransferFit tf, const string& tkfile, const string& precision_file) 
: ClassCosmology(pars, precision_file), sigma8_(0.), delta_H_(1.), transfer_fit_(tf)
{
 
    // initialize the transfer object
    InitializeTransferFunction(tf, tkfile);
}

Cosmology::Cosmology(const string& param_file,  TransferFit tf, const string& tkfile, const string& precision_file)
: ClassCosmology(param_file, precision_file), sigma8_(0.), delta_H_(1.), transfer_fit_(tf)
{
    InitializeTransferFunction(tf, tkfile);
}


void Cosmology::LoadTransferFunction(const string& tkfile, int kcol, int tcol) {
    
    /* First check the current directory for a file named 'tkfile' */
    pstring tkpath(tkfile);
    FILE* fp = fopen(tkpath.c_str(), "r");

    if(!fp) {
        /* Next search in the default data directory */
        tkpath = DATADIR "/" + tkpath;
        fp = fopen(tkpath.c_str(), "r");
    }

    if(!fp) {
        error("Transfer: could not find transfer file '%s'\n  tried ./%s and %s/%s\n", \
                tkfile.c_str(), tkfile.c_str(), DATADIR, tkfile.c_str());
    }

    verbose("Reading transfer function from %s\n", tkpath.c_str());

    Datafile data(fp);
    ki = data.GetColumn(kcol);
    Ti = data.GetColumn(tcol);
    
    // set the transfer function to "FromFile"
    transfer_fit_ = FromFile;
}

double Cosmology::EvaluateTransfer(double k) const {
    
    if (transfer_fit_ == CLASS || transfer_fit_ == FromFile) {
        return GetSplineTransfer(k);
    } else if (transfer_fit_ == EH) {
        return GetEisensteinHuTransfer(k);
    } else if (transfer_fit_ == EH_NoWiggle) {
        return GetNoWiggleTransfer(k);
    } else if (transfer_fit_ == BBKS) {
        return GetBBKSTransfer(k);
    }
    return 0.;
}

static double f(const Cosmology& C, double R, double k) {
    double kr = k*R;
    double j1_kr = (kr < 1e-4) ? 1/3. - kr*kr/30. : SphericalBesselJ1(kr)/kr;
    return k*k/(2*M_PI*M_PI) * pow(k, C.n_s())*pow2(C.EvaluateTransfer(k))*pow2(3*j1_kr);
}

void Cosmology::NormalizeTransferFunction(double sigma8) {
    
    /* Calculate sigma8 directly from the unnormalized transfer function  */
    double sigma2 = Integrate<ExpSub>(bind(f, cref(*this), 8, _1), 1e-5, 1e2, 1e-5, 1e-12);
    
    /* Set delta_H to get the desired sigma8 */
    delta_H_ = sigma8/sqrt(sigma2);
    verbose("Normalizing linear P(k) to sigma8 = %g: delta_H = %e\n", sigma8_, delta_H_);
}

// return the transfer T(k), as computed from the spline function
double Cosmology::GetSplineTransfer(double k) const {
    if (k <= 0) 
        return 0;
    else if (k <= k0)
        // Eisenstein-Hu, scaled to match splined P(k) at k = k0
        return T0 * GetNoWiggleTransfer(k)/GetNoWiggleTransfer(k0);    
    else if(k >= k1)
        // Eisenstein-Hu, scaled to match splined P(k) at k = k1
        return T1 * GetNoWiggleTransfer(k)/GetNoWiggleTransfer(k1);     
    else
        return Tk(k);
}

// return the T(k) from the full Einsenstein-Hu model
double Cosmology::GetEisensteinHuTransfer(double k) const {
    
    double q, xx, qsq;
    double T_c_ln_beta, T_c_ln_nobeta, T_c_C_alpha, T_c_C_noalpha;
    double T_c_f, T_c;
    double s_tilde, xx_tilde;
    double T_b_T0, T_b;
            
    k *= h();
    q = k/13.41/k_equality;
    xx = k*sound_horizon;

    T_c_ln_beta = log(M_E + 1.8*beta_c*q);
    T_c_ln_nobeta = log(M_E + 1.8*q);
    T_c_C_alpha = 14.2/alpha_c + 386./(1 + 69.9*pow(q, 1.08));
    T_c_C_noalpha = 14.2 + 386./(1 + 69.9*pow(q, 1.08));

    qsq = pow2(q);
    T_c_f = 1./(1. + pow4(xx/5.4));
    T_c = T_c_f*T_c_ln_beta/(T_c_ln_beta+T_c_C_noalpha*qsq) + (1-T_c_f)*T_c_ln_beta/(T_c_ln_beta+T_c_C_alpha*qsq);

    s_tilde = sound_horizon*pow(1 + pow3(beta_node/xx),-1./3.);
    xx_tilde = k*s_tilde;
    
    T_b_T0 = T_c_ln_nobeta/(T_c_ln_nobeta+T_c_C_noalpha*qsq);
    T_b = sin(xx_tilde)/(xx_tilde)*(T_b_T0/(1+pow2(xx/5.2))+alpha_b/(1+pow3(beta_b/xx))*exp(-pow(k/k_silk,1.4)));
    
    return f_baryon*T_b + (1-f_baryon)*T_c;
}

// return T(k) from the no-wiggle Eisenstein-Hu fit
double Cosmology::GetNoWiggleTransfer(double k) const {
            
    // convenience variables
    double h2 = pow2(h());
    double om = Omega0_m();
    double omh2 = Omega0_m()*h2;
    double obh2 = Omega0_b()*h2;
    double theta_cmb = Tcmb()/2.7;
    double f_baryon = obh2 / omh2;
    
    double k_equality = 0.0746*omh2/pow2(theta_cmb);
    double s = 44.5*log(9.83/om)/sqrt(1 + 10*pow(om*f_baryon, 0.75));
    double alpha_gamma = 1 - 0.328*log(431*omh2)*f_baryon + 0.38*log(22.3*omh2)*pow2(f_baryon);
    
    k *= h();
    double q = k/13.41/k_equality;
    double gamma_eff = omh2*(alpha_gamma + (1 - alpha_gamma)/(1 + pow4(0.43*k*s)));
    double q_eff = q*omh2/gamma_eff;
    double T_nowiggles_L0 = log(2*M_E + 1.8*q_eff);
    double T_nowiggles_C0 = 14.2 + 731.0/(1 + 62.5*q_eff);
    
    return T_nowiggles_L0/(T_nowiggles_L0+T_nowiggles_C0*pow2(q_eff));
}

// return T(k) from BBKS fit
double Cosmology::GetBBKSTransfer(double k) const {
            
    double q, gamma;
    
    gamma = Omega0_m()*h();
    q = k / gamma * exp(Omega0_b() + sqrt(2.*h())*Omega0_b()/Omega0_m()); 
    
    return log(1. + 2.34*q)/(2.34*q)*pow((1. + 3.89*q + pow2(16.1*q) + pow3(5.47*q) + pow4(6.71*q)), -0.25);
}

// initialize the transfer function 
void Cosmology::InitializeTransferFunction(TransferFit tf, const string& tkfile) {

    // set ki, Ti if from file or using Class
    if (tf == FromFile) {
        LoadTransferFunction(tkfile);
    } else if (tf == CLASS) {
        GetTk(0., ki, Ti);
    }
    
    // set the spline
    if (tf == FromFile || tf == CLASS) {
        int N = ki.size();
        
        Tk = LinearSpline(ki, Ti);
        k0 = ki[0];
        k1 = ki[N-1];
        T0 = Ti[0];
        T1 = Ti[N-1];
    }
    
    // normalize to sigma8() and set sigma8/delta_H 
    sigma8_ = ClassCosmology::sigma8();
    NormalizeTransferFunction(sigma8_);
}

// set the transfer function
void Cosmology::SetTransferFunction(TransferFit tf, const string& tkfile) {
    
    transfer_fit_ = tf;
    InitializeTransferFunction(tf, tkfile);
}



