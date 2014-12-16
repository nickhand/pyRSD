
#ifndef FFTLOG_FLAG
#define FFTLOG_FLAG

#include "parray.h"

// wrapper for FFTLog fortran code (implementation in FFTLogWrapper.cpp)
class FFTLog {
    
public:
    // for reverse transform, dlnr will be dlnk
    FFTLog(int N, double dlnr, double mu=0.5, double q=0, double kr=1, int kropt=1);
    bool Transform(parray& a, int dir=1);
    int Nwsave() const;
    double KR() const;
    double GetWSave(int i);
    FFTLog(const FFTLog& old);
    FFTLog& operator=(const FFTLog& old); 

    // needed
    ~FFTLog();

private:
    
    // note that there is no way to change these, and probably never should be
    // mostly don't even need to store, just do that for records
    int N, kropt;
    double mu, q, dlnr;
    double kr;
    bool ok;
    double *wsave;
};

#endif

