%{
#include "HaloZeldovichPS.h"
%}

class HaloZeldovichPS {
public:
    
    HaloZeldovichPS(double z, double sigma8);
    virtual ~HaloZeldovichPS();
    
    void set_interpolated(bool interp = true);
    
    // translated to __call__ -> calls Evaluate(K)
    double operator()(const double k) const;
    
    // translated to __call__ -> calls EvaluateMany(K)
    parray operator()(const parray& k) const;
    
    double ZeldovichPower(double k) const = 0;
    parray ZeldovichPower(const parray& k) const;
    
    double BroadbandPower(double k) const;
    parray BroadbandPower(const parray& k) const;
    
    const double& GetRedshift() const;
    const double& GetSigma8() const;
    const ZeldovichPS& GetZeldovichPower() const = 0;
    const bool& GetInterpolated() const;
    
    void SetRedshift(double z) = 0;
    void SetSigma8(double sigma8) = 0;
    
};

class HaloZeldovichP00 : public HaloZeldovichPS {

public:
    
    HaloZeldovichP00(const Cosmology& C, double z, double sigma8);
    HaloZeldovichP00(ZeldovichP00 Pzel, double z, double sigma8, bool interpolated);

    double ZeldovichPower(double k) const;
    const ZeldovichPS& GetZeldovichPower() const;
    void SetRedshift(double z); 
    void SetSigma8(double sigma8);
};


class HaloZeldovichP01 : public HaloZeldovichPS {

public:
    
    HaloZeldovichP01(const Cosmology& C, double z, double sigma8, double f);
    HaloZeldovichP01(ZeldovichP01 Pzel, double z, double sigma8, double f, bool interpolated);
    
    double BroadbandPower(double k) const;
    double ZeldovichPower(double k) const;
    
    const ZeldovichPS& GetZeldovichPower() const;
    void SetGrowthRate(double f);
    void SetRedshift(double z); 
    void SetSigma8(double sigma8);
};
