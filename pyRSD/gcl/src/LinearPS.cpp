#include "LinearPS.h"


LinearPS::LinearPS(Cosmology& C_, double z_)
    : C(C_), z(z_) {}

double LinearPS::Evaluate(double k) const {
    
    double Dz = C.D_z(z);
    double A = pow2(C.delta_H()) * pow2(Dz);
    
    return A * pow(k, C.n_s())*(pow2(C.EvaluateTransfer(k)));
}
