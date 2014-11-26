#include "GSL_Quadrature.h"

// constructor for the InnerParams struct
InnerParams::InnerParams(gsl_function_pp* gsl_funcp_, const double &x_) : 
                         gsl_funcp(gsl_funcp_), x(x_) {}


// constructor for the OuterParams struct
OuterParams::OuterParams(std::function<double (double, double)> integrand_, const double& ymin_, const double& ymax_,
                         const double& epsrel_, const double& epsabs_, gsl_integration_cquad_workspace * w_) : 
                     integrand(integrand_), ymin(ymin_), ymax(ymax_), epsrel(epsrel_), epsabs(epsabs_), w(w_) {}


// constructor for wrapper class for gsl_function
gsl_function_pp::gsl_function_pp(std::function<double(double, double)> const& func, double x) 
                                 : _func(func) 
{
        
    function = &gsl_function_pp::invoke;
    params = new InnerParams(this, x);
}    

double gsl_function_pp::invoke(double y, void *params) {
    
    InnerParams *p = static_cast<InnerParams*>(params);
    return static_cast<gsl_function_pp*>(p->gsl_funcp)->_func(p->x, y);
}

double OuterIntegrand(double x, void *params) {

    double result, error;
    OuterParams *p = static_cast<OuterParams*>(params);
    
    // set up the gsl function for the inner integral
    gsl_function_pp Fp(p->integrand, x);
    gsl_function *F = static_cast<gsl_function*>(&Fp);
    
    // do the integration
    gsl_integration_cquad(F, p->ymin, p->ymax, p->epsabs, p->epsrel, p->w, &result, &error, NULL);

    return result;
}


