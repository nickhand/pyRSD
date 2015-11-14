#include "ZeldovichPS.h"
#include "LinearPS.h"

using namespace Common;

static const int NUM_PTS = 1024;
static const double RMIN = 1e-2;
static const double RMAX = 1e5;
static const int NMAX = 30;

/*----------------------------------------------------------------------------*/
ZeldovichPS::ZeldovichPS(const Cosmology& C_, double z_) 
    : C(C_), sigma8_z(C_.Sigma8_z(z_))
{    
    // initialize the R array
    InitializeR();

    // compute integrals at z = 0 first
    LinearPS P_L(C_, 0.);
    double norm = pow2(sigma8_z/C.sigma8());
    
    // the integrals we need
    sigma_sq = norm*P_L.VelocityDispersion();    
    X0 = norm*P_L.X_Zel(r);
    XX = X0 + 2.*sigma_sq;
    YY = norm*P_L.Y_Zel(r); 
}

/*----------------------------------------------------------------------------*/
ZeldovichPS::ZeldovichPS(const Cosmology& C_, double sigma8_z_, 
                         double sigmasq, const parray& X0_, const parray& XX_, const parray& YY_) 
: C(C_), sigma8_z(sigma8_z_), sigma_sq(sigmasq), X0(X0_), XX(XX_), YY(YY_)
{
    InitializeR();     
}

/*----------------------------------------------------------------------------*/
void ZeldovichPS::InitializeR() {
  r = parray::logspace(RMIN, RMAX, NUM_PTS);
}

/*----------------------------------------------------------------------------*/
ZeldovichPS::~ZeldovichPS() {}

/*----------------------------------------------------------------------------*/
parray ZeldovichPS::EvaluateMany(const parray& k) const {
    
    int n = (int)k.size();
    parray pk(n);
    //#pragma omp parallel for
    for(int i = 0; i < n; i++)
        pk[i] = Evaluate(k[i]);
    return pk;
}

/*----------------------------------------------------------------------------*/
void ZeldovichPS::SetSigma8AtZ(double new_sigma8_z) {
     
    double ratio = pow2(new_sigma8_z / sigma8_z);
    // integrals are proportional to the square of the ratio of sigma8
    X0 *= ratio;
    XX *= ratio;
    YY *= ratio;
    sigma_sq *= ratio;
    
    // store the sigma8_z
    sigma8_z = new_sigma8_z;
}

/*----------------------------------------------------------------------------*/
double ZeldovichPS::fftlog_compute(double k, double factor) const {
    
    double q = 0; // unbiased
    double mu;
    
    dcomplex* a = new dcomplex[NUM_PTS];
    dcomplex* b = new dcomplex[NUM_PTS];
    double* kmesh = new double[NUM_PTS];
    double* b_real = new double[NUM_PTS];

    // logspaced between RMIN and RMAX
    double this_Pk = 0.;
    double toadd;   
    for (int n = 0; n <= NMAX; n++) {
        
        // the order of the Bessel function
        mu = 0.5 + double(n);
        
        // compute a(r)
        if (n == 0) 
            Fprim(a, r, k);
        else
            Fsec(a, r, k, double(n));
        
        fht(NUM_PTS, (const double*)(r), a, kmesh, b, mu, q, 1., true, NULL);
        
        // spline it
        for (int j = 0; j < NUM_PTS; j++)
          b_real[j] = b[j].real();
        Spline spl = LinearSpline(NUM_PTS, kmesh, b_real);
                
          
        toadd = factor*sqrt(0.5*M_PI)*pow(k, -1.5)*spl(k);        
        this_Pk += toadd;
        //if (fabs(toadd/this_Pk) < 0.005) break;
    }
        
    return this_Pk;
}

/*----------------------------------------------------------------------------*/
void ZeldovichPS::Fprim(dcomplex[], const double[], double) const {
    error("In ZeldovichPS::Fprim; something has gone horribly wrong\n");
}

void ZeldovichPS::Fsec(dcomplex[], const double[], double, double) const {
    error("In ZeldovichPS::Fsec; something has gone horribly wrong\n");
}

double ZeldovichPS::Evaluate(double k) const {
    return fftlog_compute(k);
}

/*----------------------------------------------------------------------------*/
ZeldovichP00::ZeldovichP00(const Cosmology& C, double z) : ZeldovichPS(C, z) {}

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

ZeldovichP01::ZeldovichP01(const Cosmology& C, double z) : ZeldovichPS(C, z) {}

ZeldovichP01::ZeldovichP01(const ZeldovichPS& ZelPS) : ZeldovichPS(ZelPS) {}


double ZeldovichP01::Evaluate(double k) const {
    return fftlog_compute(k, -2*M_PI);
}

void ZeldovichP01::Fprim(dcomplex a[], const double r[], double k) const {
    
    double k2 = pow2(k);
    double term1;
    
    for (int i = 0; i < NUM_PTS; i++) {
        term1 = exp(-0.5*k2*(XX[i] + YY[i])) - 2*sigma_sq*exp(-k2*sigma_sq);
        a[i] =  pow(r[i], 1.5)*k2*(XX[i] + YY[i])*term1;
    }
    
}

void ZeldovichP01::Fsec(dcomplex a[], const double r[], double k, double n) const {
      
    for (int i = 0; i < NUM_PTS; i++) { 
        a[i] = pow(r[i], 1.5-n)*pow(k*YY[i], n)*(pow2(k)*(XX[i] + YY[i]) - 2*n)*exp(-0.5*pow2(k)*(XX[i] + YY[i]));
    }
}

/*----------------------------------------------------------------------------*/

ZeldovichP11::ZeldovichP11(const Cosmology& C, double z) : ZeldovichPS(C, z) {}

ZeldovichP11::ZeldovichP11(const ZeldovichPS& ZelPS) : ZeldovichPS(ZelPS) {}


double ZeldovichP11::Evaluate(double k) const {
    return fftlog_compute(k, M_PI);
}

void ZeldovichP11::Fprim(dcomplex a[], const double r[], double k) const 
{
    double k2 = pow2(k), k4 = pow4(k);
    double term1, term2;
    
    for (int i = 0; i < NUM_PTS; i++) {
        term1 = (k4*pow2(XX[i]) - 2*k2*X0[i] + 2*(k2*XX[i]-1)*k2*YY[i] + k4*pow2(YY[i]))*exp(-0.5*k2*(XX[i]+YY[i]));
        term2 = -4*k4*pow2(sigma_sq)*exp(-k2*sigma_sq);
        a[i] =  pow(r[i],1.5)*(term1 + term2);
    }
}

void ZeldovichP11::Fsec(dcomplex a[], const double r[], double k, double n) const {
    
    double k2 = pow2(k), k4 = pow4(k);
    double term1, term2;
    
    for (int i = 0; i < NUM_PTS; i++) {
        term1 = k4*pow2(XX[i]) - 2*k2*X0[i] + 2*(k2*XX[i]-1)*(k2*YY[i]-2*n);
        term2 = k4*pow2(YY[i]) - 4*k2*n*YY[i] + 4*n*(n-1);
        a[i] = pow(r[i],1.5-n)*pow(k*YY[i],n) * (term1 + term2) * exp(-0.5*k2*(XX[i]+YY[i]));
    }
}



 
