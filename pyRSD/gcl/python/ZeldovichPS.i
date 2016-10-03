%{
#include "ZeldovichPS.h"
%}

class ZeldovichPS {
public:
    
    ZeldovichPS(const Cosmology& C, double z, bool approx_lowk=false);
    ZeldovichPS(const Cosmology& C, bool approx_lowk, double sigma8_z, double k0_low, 
                double sigmasq, const parray& X0, const parray& X, const parray& Y);
    virtual ~ZeldovichPS();
    
    // translated to __call__ -> calls Evaluate(K)
    double operator()(const double k) const;
    
    // translated to __call__ -> calls EvaluateMany(K)
    parray operator()(const parray& k) const;
    
    // functions for doing low-k approximation
    void SetLowKApprox(bool approx_lowk_=true);
    void SetLowKTransition(double k0);
    virtual double LowKApprox(double k) const;
    
    const double& GetSigma8AtZ() const;
    const Cosmology& GetCosmology() const;
    const bool& GetApproxLowKFlag() const;
    const double& GetK0Low() const;
    parray GetXZel() const;
    parray GetYZel() const;
    parray GetX0Zel() const;
    const double& GetSigmaSq() const;
 
    void SetSigma8AtZ(double sigma8);
    
};

class ZeldovichP00 : public ZeldovichPS {

public:
    
    ZeldovichP00(const Cosmology& C, double z, bool approx_lowk=false);
    ZeldovichP00(const ZeldovichPS& ZelPS);
    
    double LowKApprox(double k) const;

};


class ZeldovichP01 : public ZeldovichPS {

public:
    
    ZeldovichP01(const Cosmology& C, double z, bool approx_lowk=false);
    ZeldovichP01(const ZeldovichPS& ZelPS);
    
    double LowKApprox(double k) const;

};



class ZeldovichP11 : public ZeldovichPS {

public:
    
    ZeldovichP11(const Cosmology& C, double z, bool approx_lowk=false);
    ZeldovichP11(const ZeldovichPS& ZelPS);
    
    double LowKApprox(double k) const;
    
};
