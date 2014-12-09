
#include "Common.h"
#include "FFTLog.h"

extern "C" void fhti_(int*, double*, double*, double*, double*, int*, double*, bool*);
extern "C" void fht_(int*, double*, int*, double*);
 
 
// the constructor
FFTLog::FFTLog(int N_, double dlnr_, double mu_, double q_, double kr_, int kropt_):
  N(N_), kropt(kropt_), mu(mu_), q(q_), dlnr(dlnr_), kr(kr_) 
{
    int size = 2*N + 3*(N/2) + 19;
    wsave = new double[size];
    fhti_(&N, &mu, &q, &dlnr, &kr, &kropt, wsave, &ok);
}

// this does the actual transform
bool FFTLog::Transform(parray& a, int dir){
  
    if (a.size() != (size_t)N) Common::error("size mismatch");
    
    // initialize to values of a
    double *tempa = new double[N];
    for (int i=0; i<N; i++) tempa[i] = a[i]; 
    
    // call the fortran FFTlog driver
    fht_(&N, (double*)(tempa), &dir, (double*)(wsave)); 
    for (int i=0; i<N; i++) a[i] = tempa[i]; 
    
    delete[] tempa;
    return ok;
}

// length of the wsave array
int FFTLog::Nwsave() const{
    return 2*N + 3*(N/2) + 19;
}

// the value of kr used
double FFTLog::KR() const { 
    return kr;
}

double FFTLog::GetWSave(int i) {
    return wsave[i];
}

// copy constructor
FFTLog::FFTLog(const FFTLog& old) : N(old.N), kropt(old.kropt), mu(old.mu),
               q(old.q), dlnr(old.dlnr), kr(old.kr), ok(old.ok)

{
    wsave = new double[Nwsave()];
    for(int i=0; i < Nwsave(); i++) wsave[i] = old.wsave[i]; 
}

// assignment operator
FFTLog& FFTLog::operator=(const FFTLog& old) {
  
    if (this==&old) return *this;
    if (N > 0) delete[] wsave;
 
    N = old.N;
    wsave = new double[Nwsave()];
    for (int i=0; i<Nwsave(); i++) wsave[i] = old.wsave[i]; 
    kropt = old.kropt;
    mu = old.mu;
    q = old.q; 
    dlnr = old.dlnr;
    return *this;
}

FFTLog::~FFTLog(){
    if(N>0) delete[] wsave;
}

