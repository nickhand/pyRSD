#ifndef GSL_QUADRATURE_H
#define GSL_QUADRATURE_H

#include <gsl/gsl_integration.h>
#include <functional>

class gsl_function_pp;
double OuterIntegrand(double x, void *params);

// struct for keeping track of inner integrand params
struct InnerParams {
    gsl_function_pp* gsl_funcp;
    const double& x; 
    
    InnerParams(gsl_function_pp* gsl_funcp_, const double &x_);
};

// struct for keeping track of outer integrand params
struct OuterParams {
    
    std::function<double (double, double)> integrand;
    const double& ymin; 
    const double& ymax; 
    const double& epsrel;
    const double& epsabs; 
    gsl_integration_cquad_workspace * w;
    
    OuterParams(std::function<double (double, double)> integrand_, const double& ymin_, const double& ymax_,
                const double& epsrel_, const double& epsabs_, gsl_integration_cquad_workspace * w_);
};

// wrapper class for gsl_function to make it a function f(x, y) in 2D
class gsl_function_pp : public gsl_function {

public:
    gsl_function_pp(std::function<double(double, double)> const& func, double x);


private:
    std::function<double(double, double)> _func;  // store the function we want to call
    static double invoke(double y, void *params); // function to call _func
};



// the function we will actually call to the double integral
template<typename Func>
double DoubleIntegrateCQUAD(Func func, double* a, double* b, double epsrel, double epsabs) {
    
    
    gsl_integration_cquad_workspace * w;
    double result, error;
    w = gsl_integration_cquad_workspace_alloc(1000);
    gsl_integration_cquad_workspace * w_inner = gsl_integration_cquad_workspace_alloc(1000);
    
    // make the gsl function
    OuterParams params(func, a[1], b[1], epsrel, epsabs, w_inner);
    gsl_function F;
    F.function = &OuterIntegrand;
    F.params = &params;
    
    // do the integration
    gsl_integration_cquad(&F, a[0], b[0], epsabs, epsrel, w, &result, &error, NULL);
    
    //free the integration workspace
    gsl_integration_cquad_workspace_free(w);
    gsl_integration_cquad_workspace_free(w_inner);
    return result;
    
}


#endif


