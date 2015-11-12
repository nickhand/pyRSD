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
using namespace Common;


Cosmology::Cosmology() {}

Cosmology::~Cosmology() {}

Cosmology::Cosmology(const std::string& param_file)
  : ClassCosmology(param_file), sigma8_(0.), delta_H_(1.), 
  transfer_fit_(CLASS), param_file_(param_file), transfer_file_("")
{
    InitializeTransferFunction();
}

Cosmology::Cosmology(const std::string& param_file,  TransferFit tf)
  : ClassCosmology(param_file), sigma8_(0.), delta_H_(1.), 
  transfer_fit_(tf), param_file_(param_file), transfer_file_("")
{
    InitializeTransferFunction();
}

Cosmology::Cosmology(const std::string& param_file, const std::string& tkfile)
  : ClassCosmology(param_file), sigma8_(0.), delta_H_(1.), 
  transfer_fit_(FromFile), param_file_(param_file), transfer_file_(tkfile)
{
    LoadTransferFunction(transfer_file_);
    InitializeTransferFunction();
}

Cosmology* Cosmology::FromPower(const std::string& param_file, const std::string& pkfile)
{
    auto toret = new Cosmology(param_file);
    toret->LoadTransferFunction(pkfile);
    auto Pk = toret->Ti;
    auto ki = toret->ki;
    Pk = (Pk / ki.pow(toret->n_s())).pow(0.5);
    toret->Ti = Pk/Pk.max();
    toret->transfer_file_ = pkfile;
    toret->transfer_fit_ = FromFile;
    toret->InitializeTransferFunction();
    return toret;
}

void Cosmology::LoadTransferFunction(const std::string& tkfile, int kcol, int tcol) {
    
    /* First check the current directory for a file named 'tkfile' */
    pstring tkpath(tkfile);
    FILE* fp = fopen(tkpath.c_str(), "r");

    if(!fp) {
        /* Next search in the default data directory */
        tkpath = DATADIR "/" + tkpath;
        fp = fopen(tkpath.c_str(), "r");
    }

    if(!fp) {
        error("Cosmology: could not find transfer file '%s'\n  tried ./%s and %s/%s\n", \
                tkfile.c_str(), tkfile.c_str(), DATADIR, tkfile.c_str());
    }

    verbose("Cosmology: reading transfer function from %s\n", tkpath.c_str());

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
        return T0 * GetNoWiggleTransfer(k)/T0_nw;    
    else if(k >= k1)
        // Eisenstein-Hu, scaled to match splined P(k) at k = k1
        return T1 * GetNoWiggleTransfer(k)/T1_nw;     
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
            
    k *= h();
    double omh2 = Omega0_m();
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
void Cosmology::InitializeTransferFunction() 
{
    // set ki, Ti if using Class
    if (transfer_fit_ == CLASS) {
        
        verbose("computing CLASS transfer function\n");
        ki = parray::logspace(1e-5, k_max(), 500);
        int ans = GetTk(0., ki, Ti);
        if (ans > 0) 
            error("error computing CLASS transfer spline in Cosmology\n");
    }
    
    // set EH params
    SetEisensteinHuParameters();
    
    // set the spline
    if (transfer_fit_ == FromFile || transfer_fit_ == CLASS) {
        int N = ki.size();
        
        Tk = LinearSpline(ki, Ti);
        k0 = ki[0];
        k1 = ki[N-1];
        T0 = Ti[0];
        T1 = Ti[N-1];

        T0_nw = GetNoWiggleTransfer(k0);
        T1_nw = GetNoWiggleTransfer(k1);
    }
    
    // normalize to sigma8() and set sigma8/delta_H 
    sigma8_ = ClassCosmology::sigma8();
    NormalizeTransferFunction(sigma8_);
}

// set the transfer function
void Cosmology::SetTransferFunction(TransferFit tf, const std::string& tkfile) {
    
    transfer_fit_ = tf;
    transfer_file_ = tkfile;
    InitializeTransferFunction();
}

void Cosmology::SetEisensteinHuParameters() {
    
    double h2 = pow2(h());
    double om = Omega0_m();
    double omh2 = Omega0_m()*h2;
    double obh2 = Omega0_b()*h2;
    double theta_cmb = Tcmb()/2.7;
    
    // baryon fraction
    f_baryon = obh2 / omh2;

    // wavenumber of equality
    double z_equality = 2.5e4*omh2/pow3(theta_cmb);
    k_equality = 0.0746*omh2/pow2(theta_cmb);
    
    // sound horizon and k_silk
    double z_drag_b1 = 0.313*pow(omh2,-0.419)*(1 + 0.607*pow(omh2, 0.674));
    double z_drag_b2 = 0.238*pow(omh2, 0.223);
    double z_drag = 1291*pow(omh2, 0.251)/(1 + 0.659*pow(omh2, 0.828))*(1+z_drag_b1*pow(obh2,z_drag_b2));
    double R_drag = 31.5*obh2/pow4(theta_cmb)*(1000/(1+z_drag));
    double R_equality = 31.5*obh2/pow4(theta_cmb)*(1000/z_equality);
    sound_horizon = 2./3./k_equality*sqrt(6./R_equality)*log((sqrt(1+R_drag)+sqrt(R_drag+R_equality))/(1+sqrt(R_equality)));
    k_silk = 1.6*pow(obh2,0.52)*pow(omh2,0.73)*(1+pow(10.4*omh2,-0.95));
    
    // alpha_c
    double alpha_c_a1 = pow(46.9*omh2,0.670)*(1+pow(32.1*omh2,-0.532));
    double alpha_c_a2 = pow(12.0*omh2,0.424)*(1+pow(45.0*omh2,-0.582));
    alpha_c = pow(alpha_c_a1,-f_baryon)*pow(alpha_c_a2,-pow3(f_baryon));   
    
    // beta_c
    double beta_c_b1 = 0.944/(1+pow(458*omh2,-0.708));
    double beta_c_b2 = pow(0.395*omh2, -0.0266);
    beta_c = 1.0/(1+beta_c_b1*(pow(1-f_baryon, beta_c_b2)-1));
    
    double y = z_equality/(1+z_drag);
    double alpha_b_G = y*(-6.*sqrt(1+y)+(2.+3.*y)*log((sqrt(1+y)+1)/(sqrt(1+y)-1)));
    alpha_b = 2.07*k_equality*sound_horizon*pow(1+R_drag,-0.75)*alpha_b_G;
    beta_node = 8.41*pow(omh2, 0.435);
    beta_b = 0.5+f_baryon+(3.-2.*f_baryon)*sqrt(pow(17.2*omh2,2.0)+1);
    
    // no wiggle params
    s = 44.5*log(9.83/om)/sqrt(1 + 10*pow(om*f_baryon, 0.75));
    alpha_gamma = 1 - 0.328*log(431*omh2)*f_baryon + 0.38*log(22.3*omh2)*pow2(f_baryon);
    
    
}



