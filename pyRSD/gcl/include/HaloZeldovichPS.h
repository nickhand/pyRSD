#ifndef HALO_ZELDOVICH_PS_H
#define HALO_ZELDOVICH_PS_H

#include "Cosmology.h"
#include "ZeldovichPS.h"
#include "Spline.h"

/*------------------------------------------------------------------------------
    HaloZeldovichPS
    ---------
    Power spectra using the Halo Zel'dovich method for P00 and P01 (see 
    Seljak and Vlah 2015) for details
------------------------------------------------------------------------------*/

// base class
class HaloZeldovichPS  {
public:
       
    // constructors
    HaloZeldovichPS(double z, double sigma8);
    virtual ~HaloZeldovichPS();
    
    // use a sigma8 interpolation table for Zel'dovich part?
    void set_interpolated(bool interp = true);
    
    // zeldovich power
    virtual double ZeldovichPower(double k) const = 0;
    parray ZeldovichPower(const parray& k) const;
    
    // broadband power
    virtual double BroadbandPower(double k) const;
    parray BroadbandPower(const parray& k) const;
    
    virtual double Evaluate(double k) const; 
    double operator()(double k) const { return Evaluate(k); }
    
    virtual parray EvaluateMany(const parray& k) const;
    parray operator()(const parray& k) const { return EvaluateMany(k); }
        
    // get references to various attributes
    virtual const ZeldovichPS& GetZeldovichPower() const = 0;
    const double& GetRedshift() const { return z; }
    const double& GetSigma8() const { return sigma8; }
    const bool& GetInterpolated() const { return interpolated; }
            
    // set the redshift and sigma8
    virtual void SetRedshift(double z_) = 0;
    virtual void SetSigma8(double sigma8_) = 0;
         
protected:
    
    // keep track of a few variables for easy scaling
    double z, sigma8;
    
    // interpolation table
    bool interpolated;
    std::vector<Spline> table;
    
    // make the interpolation table
    void MakeInterpolationTable();
    
    // model parameters
    double Sigma8_z() const;
    double A0(double s8_z) const;
    double R(double s8_z) const;
    double R1(double s8_z) const;
    double R1h(double s8_z) const;
    double R2h(double s8_z) const;
    double CompensationFunction(double k, double R) const;
        
};


class HaloZeldovichP00 : public HaloZeldovichPS {
public:
    
    HaloZeldovichP00(const Cosmology& C, double z, double sigma8);
    HaloZeldovichP00(ZeldovichP00 Pzel, double z, double sigma8, bool interpolated);
    
    double ZeldovichPower(double k) const { return Pzel(k); }
    const ZeldovichPS& GetZeldovichPower() const { return Pzel; }
    
    void SetSigma8(double sigma8_) { Pzel.SetSigma8(sigma8_); sigma8=sigma8_; }
    void SetRedshift(double z_);
    
private:
    
    ZeldovichP00 Pzel;
   
};

class HaloZeldovichP01 : public HaloZeldovichPS {
public:
    
    HaloZeldovichP01(const Cosmology& C, double z, double sigma8, double f);
    HaloZeldovichP01(ZeldovichP01 Pzel, double z, double sigma8, double f, bool interpolated);

    // factor of 2*f is included in Evaluate/EvaluateMany
    double ZeldovichPower(double k) const { return Pzel(k); }
    double BroadbandPower(double k) const;
    
    parray EvaluateMany(const parray& k) const;
    double Evaluate(double k) const;
    
    const ZeldovichPS& GetZeldovichPower() const { return Pzel; }
    
    void SetSigma8(double sigma8_) { Pzel.SetSigma8(sigma8_); sigma8=sigma8_; }
    void SetGrowthRate(double f_) { f=f_; }
    void SetRedshift(double z_);
    

private:
    
    ZeldovichP01 Pzel;
        
    // logarithmic growth rate
    double f; 
   
    // derivatives of model params
    double dA0_dlna(double s8_z) const;
    double dR_dlna(double s8_z) const;
    double dR1_dlna(double s8_z) const;
    double dR1h_dlna(double s8_z) const;
    double dR2h_dlna(double s8_z) const;
    
};

#endif // HALO_ZELDOVICH_PS_H