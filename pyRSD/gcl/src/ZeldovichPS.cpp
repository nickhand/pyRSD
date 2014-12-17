#include "ZeldovichPS.h"

using namespace Common;

static const int NUM_PTS = 1024;
static const double RMIN = 1e-2;
static const double RMAX = 1e5;
static const int NMAX = 15;


ZeldovichPS::ZeldovichPS(const PowerSpectrum& P_L_) 
    : P_L(P_L_), z(P_L.GetRedshift()), sigma8(P_L.GetCosmology().sigma8())
{    
    // compute the integrals and mesh parameters
    sigma_sq = P_L.VelocityDispersion();
   
    r = parray(NUM_PTS);     
    nc = 0.5*double(NUM_PTS+1);
    double logrmin = log10(RMIN); 
    double logrmax = log10(RMAX);
    logrc = 0.5*(logrmin+logrmax);
    dlogr = (logrmax - logrmin)/NUM_PTS; 
    
    for (int i = 1; i <= NUM_PTS; i++)
        r[i-1] = pow(10., (logrc+(i-nc)*dlogr));
    
    XX = P_L.X_Zel(r) + 2.*sigma_sq;
    YY = P_L.Y_Zel(r); 
}

ZeldovichPS::~ZeldovichPS() {}


static void nearest_interp_1d(int nd, double *xd, double *yd, int ni, double *xi, double *yi) {
    double d, d2;
    int i, j, k, l;
    
    for (i = 0; i < ni; i++) {
        k = 0;
        d = fabs(xi[i] - xd[k]);
        for(j = 1; j < nd; j++) {
            d2 = fabs(xi[i] - xd[j]);
            if (d2 < d) {
                k = j;
                d = d2;
            }
        }
        if (xi[i] > xd[k]) {
            l = k+1;
            yi[i] = yd[l] + (yd[k] - yd[l])/(xd[k] - xd[l])*(xi[i]-xd[l]);
        } else if (xi[i] < xd[k]) {
            l = k-1;
            yi[i] = yd[l] + (yd[k] - yd[l] )/(xd[k] - xd[l])*(xi[i]-xd[l]);
        } else
            yi[i] = yd[k];
    }
}

void ZeldovichPS::SetRedshift(double new_z) {
    
    const Cosmology& cosmo = P_L.GetCosmology();
    double old_Dz = cosmo.D_z(z);
    double new_Dz = cosmo.D_z(new_z);
    double ratio = pow2(new_Dz / old_Dz);
    
    // integrals are proportional to the square of the ratio of growth factors
    XX *= ratio;
    YY *= ratio;
    sigma_sq *= ratio;
    
    // store the new redshift
    z = new_z;
}

parray ZeldovichPS::EvaluateMany(const parray& k) const {
    
    int n = (int)k.size();
    parray pk(n);
    #pragma omp parallel for
    for(int i = 0; i < n; i++)
        pk[i] = Evaluate(k[i]);
    return pk;
}

void ZeldovichPS::SetSigma8(double new_sigma8) {
     
    double ratio = pow2(new_sigma8 / sigma8);
    // integrals are proportional to the square of the ratio of sigma8
    XX *= ratio;
    YY *= ratio;
    sigma_sq *= ratio;
    
    // store the new redshift
    sigma8 = new_sigma8;
}

double ZeldovichPS::fftlog_compute(double k, double factor) const {
    
    double q = 0; // unbiased
    double mu;
    
    // the input/output arrays
    parray a(NUM_PTS);
    parray kmesh(NUM_PTS);

    // logspaced between RMIN and RMAX
    double this_Pk = 0.;   
    for (int n = 0; n <= NMAX; n++) {
        
        // the order of the Bessel function
        mu = 0.5 + double(n);
        
        // compute a(r)
        if (n == 0) 
            Fprim(a, r, k);
        else
            Fsec(a, r, k, double(n));
        
        // do the fft
        FFTLog fftlogger(NUM_PTS, dlogr*log(10.), mu, q, 1.0, 1);
                    
        bool ok = fftlogger.Transform(a, 1);
        if (!ok) error("FFTLog failed\n");
        double kr = fftlogger.KR();
        double logkc = log10(kr) - logrc;

        for(int j = 1; j <= NUM_PTS; j++) 
            kmesh[j-1] = pow(10., (logkc+(j-nc)*dlogr));
        
        // sum it up
        double out;
        nearest_interp_1d(NUM_PTS, (double*)(kmesh), (double*)(a), 1, &k, &out);
        double toadd = factor*sqrt(0.5*M_PI)*pow(k, -1.5)*out; 
        
        this_Pk += toadd;        
        if (fabs(toadd/this_Pk) < 0.005) break;
    }
        
    return this_Pk;
}

void ZeldovichPS::Fprim(parray&, const parray&, double) const {
    error("In ZeldovichPS::Fprim; something has gone horribly wrong\n");
}

void ZeldovichPS::Fsec(parray&, const parray&, double, double) const {
    error("In ZeldovichPS::Fsec; something has gone horribly wrong\n");
}

double ZeldovichPS::Evaluate(double k) const {
    return fftlog_compute(k);
}

/*----------------------------------------------------------------------------*/
ZeldovichP00::ZeldovichP00(const PowerSpectrum& P_L) : ZeldovichPS(P_L) {}

ZeldovichP00::ZeldovichP00(const ZeldovichPS& ZelPS) : ZeldovichPS(ZelPS) {}

double ZeldovichP00::Evaluate(double k) const {
    return fftlog_compute(k, 4*M_PI);
}

void ZeldovichP00::Fprim(parray& a, const parray& r, double k) const {
    
    for (int i = 0; i < NUM_PTS; i++) {
        a[i] = pow(r[i], 1.5) * (exp(-0.5*pow2(k)*(XX[i] + YY[i])) - exp(-pow2(k)*sigma_sq));
    }
    
}

void ZeldovichP00::Fsec(parray& a, const parray& r, double k, double n) const {
    
    for (int i = 0; i < NUM_PTS; i++) {
        a[i] = pow(r[i], 1.5-n)*pow(k*YY[i], n)*exp(-0.5*pow2(k)*(XX[i] + YY[i]));
    }
}

/*----------------------------------------------------------------------------*/

ZeldovichP01::ZeldovichP01(const PowerSpectrum& P_L) : ZeldovichPS(P_L) {}

ZeldovichP01::ZeldovichP01(const ZeldovichPS& ZelPS) : ZeldovichPS(ZelPS) {}


double ZeldovichP01::Evaluate(double k) const {
    return fftlog_compute(k, -2*M_PI);
}

void ZeldovichP01::Fprim(parray& a, const parray& r, double k) const {
    
    for (int i = 0; i < NUM_PTS; i++) {
        a[i] =  pow(r[i], 1.5)*pow2(k)*((XX[i] + YY[i])*exp(-0.5*pow2(k)*(XX[i] + YY[i])) - 2*sigma_sq*exp(-pow2(k)*sigma_sq));
    }
    
}

void ZeldovichP01::Fsec(parray& a, const parray& r, double k, double n) const {
    
    for (int i = 0; i < NUM_PTS; i++) {
        a[i] = pow(r[i], 1.5-n)*pow(k*YY[i], n)*(pow2(k)*(XX[i] + YY[i]) - 2*n)*exp(-0.5*pow2(k)*(XX[i] + YY[i]));
    }
}



 
