#include "HaloZeldovichPS.h"
#include <algorithm>

using namespace Common;

// K interpolation
const double K_MIN = 1e-3;
const double K_MAX = 1.0;
const parray K_INTERP = parray::logspace(K_MIN, K_MAX, 100);

// sigma8 interpolation
const double SIGMA8_MIN = 0.1;
const double SIGMA8_MAX = 2.5;
const parray SIGMA8_INTERP = parray::linspace(SIGMA8_MIN, SIGMA8_MAX, 100);


/*----------------------------------------------------------------------------*/
HaloZeldovichPS::HaloZeldovichPS(double z_, double sigma8_) 
                                    : z(z_), sigma8(sigma8_), interpolated(false)
{
}

/*----------------------------------------------------------------------------*/
HaloZeldovichPS::~HaloZeldovichPS() {}

/*----------------------------------------------------------------------------*/
void HaloZeldovichPS::MakeInterpolationTable() {
    
    if (table.empty())
        table.resize(SIGMA8_INTERP.size());
    
    // store current values
    bool interp_0(interpolated);
    double sigma8_0(sigma8);
    
    interpolated = false;
    for (size_t i = 0; i < SIGMA8_INTERP.size(); i++) {
        SetSigma8(SIGMA8_INTERP[i]);
        table[i] = CubicSpline(K_INTERP, ZeldovichPower(K_INTERP));
    }
    
    // reset values
    interpolated = interp_0;
    SetSigma8(sigma8_0);
    
}

/*----------------------------------------------------------------------------*/
void HaloZeldovichPS::set_interpolated(bool interp) { 
    if (interp != interpolated) {
        interpolated = interp;
        if (interpolated) MakeInterpolationTable();
    }   
}

/*----------------------------------------------------------------------------*/
double HaloZeldovichPS::A0(double s8_z) const {
    /*
    Returns the A0 radius parameter (see eqn 4 of arXiv:1501.07512)

    Note: the units are power [(h/Mpc)^3]
    */
    return 750*pow(s8_z/0.8, 3.75);
}

double HaloZeldovichPS::R(double s8_z) const {
    /* 
    Returns the R radius parameter (see eqn 4 of arXiv:1501.07512)

    Note: the units are length [Mpc/h]
    */
    return 26*pow(s8_z/0.8, 0.15);
}

double HaloZeldovichPS::R1(double s8_z) const {
    /*
    Returns the R1 radius parameter (see eqn 5 of arXiv:1501.07512)
    
    Note: the units are length [Mpc/h]
    */
    return 3.33*pow(s8_z/0.8, 0.88);
}

double HaloZeldovichPS::R1h(double s8_z) const {
    /*
    Returns the R1h radius parameter (see eqn 5 of arXiv:1501.07512)

    Note: the units are length [Mpc/h]
    */
    return 3.87*pow(s8_z/0.8, 0.29);
}

double HaloZeldovichPS::R2h(double s8_z) const {
    /*
    Returns the R2h radius parameter (see eqn 5 of arXiv:1501.07512)

    Note: the units are length [Mpc/h]
    */
    return 1.69*pow(s8_z/0.8, 0.43);
}

double HaloZeldovichPS::CompensationFunction(double k, double R) const {
    /*
    The compensation function F(k) that causes the broadband power to go
    to zero at low k, in order to conserver mass/momentum
    
    The functional form is given by 1 - 1 / (1 + k^2 R^2), where R(z) 
    is given by Eq. 4 in arXiv:1501.07512.
    */
    return 1. - 1./(1. + pow2(k*R));
}

double HaloZeldovichPS::Sigma8_z() const {
    /*
    Return sigma8(z), normalized to the desired sigma8 at z = 0
    */
    const Cosmology& cosmo(GetZeldovichPower().GetCosmology());
    return sigma8 * (cosmo.Sigma8_z(z) / cosmo.sigma8());
}


/*----------------------------------------------------------------------------*/
parray HaloZeldovichPS::EvaluateMany(const parray& k) const {
    return BroadbandPower(k) + ZeldovichPower(k);
}

/*----------------------------------------------------------------------------*/
double HaloZeldovichPS::Evaluate(double k) const {
    return BroadbandPower(k) + ZeldovichPower(k);
}

/*----------------------------------------------------------------------------*/
parray HaloZeldovichPS::ZeldovichPower(const parray& k) const {
    
    // setup the return
    int n = (int)k.size();
    parray pk(n);
    
    if (!interpolated) {

        #pragma omp parallel for
        for(int i = 0; i < n; i++)
            pk[i] = ZeldovichPower(k[i]);

    } else {
        if ((sigma8 < SIGMA8_MIN) || (sigma8 > SIGMA8_MAX))
            error("Cannont interpolate Halo Zeldovich PS -- sigma8 out of range");
            
        auto high = lower_bound(SIGMA8_INTERP.begin(), SIGMA8_INTERP.end(), sigma8);
        int ihi = high - SIGMA8_INTERP.begin();
        double s8_lo = SIGMA8_INTERP[ihi-1];
        double s8_hi = SIGMA8_INTERP[ihi];
        double w = (sigma8 - s8_lo) / (s8_hi - s8_lo);
        
        #pragma omp parallel for
        for(int i = 0; i < n; i++)
            pk[i] = (1 - w)*table[ihi-1](k[i]) + w*table[ihi](k[i]);
    }
    return pk;
}

/*----------------------------------------------------------------------------*/
double HaloZeldovichPS::BroadbandPower(double k) const {
    /*
    The broadband power correction in units of (Mpc/h)^3
    
    The functional form is given by: 
    
    P_BB = A0 * F(k) * [ (1 + (k*R1)^2) / (1 + (k*R1h)^2 + (k*R2h)^4) ], 
    as given by Eq. 1 in arXiv:1501.07512.
    */
    double s8_z(Sigma8_z());
    double F(CompensationFunction(k, R(s8_z)));
    return F*A0(s8_z)*(1 + pow2(k*R1(s8_z)))/(1 + pow2(k*R1h(s8_z)) + pow4(k*R2h(s8_z)));
}


/*----------------------------------------------------------------------------*/
parray HaloZeldovichPS::BroadbandPower(const parray& k) const {
    
    int n = (int)k.size();
    parray pk(n);
    #pragma omp parallel for
    for(int i = 0; i < n; i++)
        pk[i] = BroadbandPower(k[i]);
    return pk;
}

/*----------------------------------------------------------------------------*/
HaloZeldovichP00::HaloZeldovichP00(const Cosmology& C_, double z_, double sigma8_)
                                    : HaloZeldovichPS(z_, sigma8_), Pzel(C_, z_) 
{
    // need to explicitly set ZeldovichPS sigma8
    Pzel.SetSigma8(sigma8);
}

/*----------------------------------------------------------------------------*/
HaloZeldovichP00::HaloZeldovichP00(ZeldovichP00 Pzel_, double z_, double sigma8_, 
                                    bool interpolated_) 
                                    : HaloZeldovichPS(z_, sigma8_),
                                    Pzel(Pzel_)
{
    set_interpolated(interpolated_);
}

/*----------------------------------------------------------------------------*/
void HaloZeldovichP00::SetRedshift(double z_) { 
    
    Pzel.SetRedshift(z_); 
    z=z_; 
    if (interpolated)
        MakeInterpolationTable();
}


/*----------------------------------------------------------------------------*/
HaloZeldovichP01::HaloZeldovichP01(const Cosmology& C_, double z_, double sigma8_, double f_) 
                                    : HaloZeldovichPS(z_, sigma8_), Pzel(C_, z_), f(f_) 
{
    // need to explicitly set ZeldovichPS sigma8
    Pzel.SetSigma8(sigma8);
}

/*----------------------------------------------------------------------------*/
HaloZeldovichP01::HaloZeldovichP01(ZeldovichP01 Pzel_, double z_, double sigma8_, 
                                    double f_, bool interpolated_) 
                                    : HaloZeldovichPS(z_, sigma8_), Pzel(Pzel_), f(f_)

{   
    set_interpolated(interpolated_);
}

/*----------------------------------------------------------------------------*/
double HaloZeldovichP01::BroadbandPower(double k) const {
    /*
    The broadband power correction for P01 in units of (Mpc/h)^3

    This is basically the derivative of the broadband band term for P00, taken
    with respect to lna
    */
    double s8(Sigma8_z());
    double F(CompensationFunction(k, R(s8)));
    
    // the P00_BB parameters
    double iA0(A0(s8));
    double iR1(R1(s8));
    double iR1h(R1h(s8));
    double iR2h(R2h(s8));
    double iR(R(s8));
    
    // derivs wrt to lna
    double dA0(dA0_dlna(s8));
    double dR1(dR1_dlna(s8));
    double dR1h(dR1h_dlna(s8));
    double dR2h(dR2h_dlna(s8));
    double dR(dR_dlna(s8));
    
    // store these for convenience
    double norm(1 + pow2(k*iR1h) + pow4(k*iR2h));
    double C((1. + pow2(k*iR1)) / norm);
    
    // 1st term of tot deriv
    double term1(dA0*F*C);
    
    // 2nd term
    double term2((iA0*C) * (2*k*k*iR*dR) / pow2(1 + pow2(k*iR)));
    
    // 3rd term
    double term3_a((2*k*k*iR1*dR1) / norm);
    double term3_b(-(1 + pow2(k*iR1)) / pow2(norm) * (2*k*k*iR1h*dR1h + 4*pow4(k)*pow3(iR2h)*dR2h));
    double term3((iA0*F) * (term3_a + term3_b));
    
    return term1 + term2 + term3;
}

/*----------------------------------------------------------------------------*/

double HaloZeldovichP01::dA0_dlna(double s8_z) const {

    return f * 3.75 * A0(s8_z);
}

double HaloZeldovichP01::dR_dlna(double s8_z) const {

    return f * 0.15 * R(s8_z);
}

double HaloZeldovichP01::dR1_dlna(double s8_z) const {

    return f * 0.88 * R1(s8_z);
}

double HaloZeldovichP01::dR1h_dlna(double s8_z) const {

    return f * 0.29 * R1h(s8_z);
}

double HaloZeldovichP01::dR2h_dlna(double s8_z) const {

    return f * 0.43 * R2h(s8_z);
}

/*----------------------------------------------------------------------------*/


/*----------------------------------------------------------------------------*/
parray HaloZeldovichP01::EvaluateMany(const parray& k) const {
    return HaloZeldovichPS::BroadbandPower(k) + 2*f*HaloZeldovichPS::ZeldovichPower(k);
}

/*----------------------------------------------------------------------------*/
double HaloZeldovichP01::Evaluate(double k) const {
    return BroadbandPower(k) + 2*f*ZeldovichPower(k);
}

/*----------------------------------------------------------------------------*/
void HaloZeldovichP01::SetRedshift(double z_) { 
    
    Pzel.SetRedshift(z_); 
    z=z_; 
    if (interpolated)
        MakeInterpolationTable();
}

 
