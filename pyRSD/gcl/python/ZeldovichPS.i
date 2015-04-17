%{
#include "ZeldovichPS.h"
%}

class ZeldovichPS {
public:
    
    ZeldovichPS(const Cosmology& C, double z);
    ZeldovichPS(const Cosmology& C, double z, double sigma8, double sigmasq, const parray& X, const parray& Y);
    virtual ~ZeldovichPS();
    
    // translated to __call__ -> calls Evaluate(K)
    double operator()(const double k) const;
    
    // translated to __call__ -> calls EvaluateMany(K)
    parray operator()(const parray& k) const;
    
    const double& GetRedshift() const;
    const double& GetSigma8() const;
    const Cosmology& GetCosmology() const;
    parray GetXZel() const;
    parray GetYZel() const;
    const double& GetSigmaSq() const;
    
    void SetRedshift(double z); 
    void SetSigma8(double sigma8);
    
};

class ZeldovichP00 : public ZeldovichPS {

public:
    
    ZeldovichP00(const Cosmology& C, double z);
    ZeldovichP00(const ZeldovichPS& ZelPS);

};


class ZeldovichP01 : public ZeldovichPS {

public:
    
    ZeldovichP01(const Cosmology& C, double z);
    ZeldovichP01(const ZeldovichPS& ZelPS);
    
};
