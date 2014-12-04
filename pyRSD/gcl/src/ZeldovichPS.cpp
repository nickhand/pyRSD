#include "ZeldovichPS.h"

using namespace Common;

static const int NUM_PTS = 1024;
static const double RMIN = 1e-2;
static const double RMAX = 1e5;
static const int NMAX = 30;

ZeldovichPS::ZeldovichPS(const PowerSpectrum& P_L_) 
    : P_L(P_L_), z(P_L.GetRedshift()), sigma8(P_L.GetCosmology().sigma8())
{    
    // compute the integrals and mesh parameters
    sigma_sq = P_L.VelocityDispersion();
    
    // compute the integrals
    r = parray::logspace(RMIN, RMAX, NUM_PTS); 
    XX = P_L.X_Zel(r) + 2.*sigma_sq;
    YY = P_L.Y_Zel(r); 
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
    double kcrc = 1; // good first guess
    double mu;
    
    // the input/output arrays
    dcomplex* a = new dcomplex[NUM_PTS];
    dcomplex* b = new dcomplex[NUM_PTS];
    double* kmesh = new double[NUM_PTS];
    double* b_real = new double[NUM_PTS];

    // logspaced between RMIN and RMAX
    double this_Pk = 0.;
    
    for (int n = 0; n < NMAX; n++) {
        
        // the order of the Bessel function
        mu = 0.5 + double(n);
        
        // compute a(r)
        if (n == 0)
            Fprim(a, (const double *)(r), k);
        else
            Fsec(a, (const double *)(r), k, double(n));

        // do the fft
        fht(NUM_PTS, (const double*)(r), a, kmesh, b, mu, q, kcrc, true, NULL);
        
        // spline it
        for (int j = 0; j < NUM_PTS; j++)
            b_real[j] = b[j].real();
        Spline spl = LinearSpline(NUM_PTS, kmesh, b_real);
        
        // sum it up
        this_Pk += factor*sqrt(0.5*M_PI)*pow(k, -1.5)*spl(k);
    }
    
    delete[] a;
    delete[] b;
    delete[] kmesh;
    delete[] b_real;
    
    return this_Pk;
}

void ZeldovichPS::Fprim(dcomplex a[], const double r[], double k) const {
    error("In ZeldovichPS::Fprim; something has gone horribly wrong\n");
}

void ZeldovichPS::Fsec(dcomplex a[], const double r[], double k, double n) const {
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

void ZeldovichP00::Fprim(dcomplex a[], const double r[], double k) const {
    
    for (int i = 0; i < NUM_PTS; i++) {
        a[i] = pow(r[i], 1.5) * (exp(-0.5*pow2(k)*(XX[i] + YY[i])) - exp(-pow2(k)*sigma_sq));
    }
    
}

void ZeldovichP00::Fsec(dcomplex a[], const double r[], double k, double n) const {
    
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

void ZeldovichP01::Fprim(dcomplex a[], const double r[], double k) const {
    
    for (int i = 0; i < NUM_PTS; i++) {
        a[i] =  pow(r[i], 1.5)*pow2(k)*((XX[i] + YY[i])*exp(-0.5*pow2(k)*(XX[i] + YY[i])) - 2*sigma_sq*exp(-pow2(k)*sigma_sq));
    }
    
}

void ZeldovichP01::Fsec(dcomplex a[], const double r[], double k, double n) const {
    
    for (int i = 0; i < NUM_PTS; i++) {
        a[i] = pow(r[i], 1.5-n)*pow(k*YY[i], n)*(pow2(k)*(XX[i] + YY[i]) - 2*n)*exp(-0.5*pow2(k)*(XX[i] + YY[i]));
    }
}



 

