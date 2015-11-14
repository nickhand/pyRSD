%{
#include "ZeldovichPS.h"
%}

class ZeldovichPS {
public:
    
    ZeldovichPS(const Cosmology& C, double z);
    ZeldovichPS(const Cosmology& C, double sigma8_z, double sigmasq, const parray& X0, const parray& X, const parray& Y);
    virtual ~ZeldovichPS();
    
    // translated to __call__ -> calls Evaluate(K)
    double operator()(const double k) const;
    
    // translated to __call__ -> calls EvaluateMany(K)
    parray operator()(const parray& k) const;
    
    const double& GetSigma8AtZ() const;
    const Cosmology& GetCosmology() const;
    parray GetXZel() const;
    parray GetYZel() const;
    parray GetX0Zel() const;
    const double& GetSigmaSq() const;
 
    void SetSigma8AtZ(double sigma8);
    
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



class ZeldovichP11 : public ZeldovichPS {

public:
    
    ZeldovichP11(const Cosmology& C, double z);
    ZeldovichP11(const ZeldovichPS& ZelPS);
    
};
