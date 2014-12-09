
#include "Common.h"
#include "FFTLog.h"

extern "C" void fhti_(int*, double*, double*, double*, double*, int*, double*, bool*);
extern "C" void fht_(int*, double*, int*, double*);
 
 
// the constructor
FFTLog::FFTLog(int N, double dlnr, double mu, double q, double kr, int kropt):
  N(N), kropt(kropt), mu(mu), q(q), dlnr(dlnr), kr(kr) 
{
    wsave = parray(Nwsave());
    fhti_(&N, &mu, &q, &dlnr, &kr, &kropt, (double*)(wsave), &ok);
}

// this does the actual transform
bool FFTLog::Transform(parray& a, int dir){
  
    if (a.size() != (size_t)N) Common::error("size mismatch");
    
    // initialize to values of a
    parray tempa(a);    
    
    // call the fortran FFTlog driver
    fht_(&N, (double*)(tempa), &dir, wsave); 
    for (int i=0; i<N; i++) a[i] = tempa[i]; 
    return ok;
}

// length of the wsave array
int FFTLog::Nwsave() const{
  return   (7*N)/2+19;
}

// the value of kr used
double FFTLog::KR() const { 
  return kr;
}

// copy constructor
FFTLog::FFTLog(const FFTLog& old) : N(old.N), kropt(old.kropt), mu(old.mu),
               q(old.q), dlnr(old.dlnr), kr(old.kr), ok(old.ok)

{
    wsave = parray(Nwsave());
    for(int i=0; i < Nwsave(); i++) wsave[i] = old.wsave[i]; 
}

// assignment operator
FFTLog& FFTLog::operator=(const FFTLog& old) {
  
    if (this==&old) return *this;
    //if (N > 0) NR::free_dvector(wsave,0,Nwsave()-1);
 
    N = old.N;
    wsave = parray(Nwsave());
    for (int i=0; i<Nwsave(); i++) wsave[i] = old.wsave[i]; 
    kropt = old.kropt;
    mu = old.mu;
    q = old.q; 
    dlnr = old.dlnr;
    return *this;
}

FFTLog::~FFTLog(){
  // if(N>0) NR::free_dvector(wsave,0,Nwsave()-1);
}

