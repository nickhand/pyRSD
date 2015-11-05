#include <cfloat>
#include <cmath>

#include "Common.h"
#include "SpecialFunctions.h"

/* Use standard library implementations */
double BesselJ0(double x) { return j0(x); }
double BesselJ1(double x) { return j1(x); }
double BesselJn(int n, double x) { return jn(n, x); }

double SphericalBesselJ0(double x) {
    if(fabs(x) < 1e-4)
        return 1 - Common::pow2(x)/6 + Common::pow4(x)/120 - Common::pow6(x)/5040;
    else
        return sin(x)/x;
}

double SphericalBesselJ1(double x) {
    if(fabs(x) < 1e-4)
        return x/3 - Common::pow3(x)/30 + Common::pow5(x)/840 - Common::pow7(x)/45360;
    else
        return (sin(x) - x*cos(x))/Common::pow2(x);
}

double SphericalBesselJ2(double x) {
    if(fabs(x) < 1e-4)
        return Common::pow2(x)/15 - Common::pow4(x)/210 + Common::pow6(x)/7560;
    else
        return ((3 - Common::pow2(x))*sin(x) - 3*x*cos(x))/Common::pow3(x);
}

double SphericalBesselJ3(double x) {
    if(fabs(x) < 1e-4)
        return Common::pow3(x)/105 - Common::pow5(x)/1890 + Common::pow7(x)/83160;
    else
        return ((15 - 6*Common::pow2(x))*sin(x) - (15*x - Common::pow3(x))*cos(x))/Common::pow4(x);
}

double SphericalBesselJ4(double x) {
    if(fabs(x) < 1e-4)
        return Common::pow4(x)/945 - Common::pow6(x)/20790;
    else
        return 5.*(2.*Common::pow2(x) - 21.)*cos(x)/Common::pow4(x) + (Common::pow4(x) - 45.*Common::pow2(x) + 105.)*sin(x)/Common::pow5(x);
}

double SphericalBesselJ6(double x) {
    if(fabs(x) < 1e-4)
        return Common::pow6(x)/135135 - Common::pow8(x)/4054050;
    else
        return ((-Common::pow6(x) + 210*Common::pow4(x) - 4725*Common::pow2(x) + 10395)*sin(x) - (21*x*(Common::pow4(x) - 60*Common::pow2(x) + 495))*cos(x))/Common::pow7(x);
}

double SphericalBesselJ8(double x) {
    if(fabs(x) < 1e-4)
        return Common::pow8(x)/34459425 - Common::pow10(x)/1309458150;
    else
        return ((Common::pow8(x) - 630*Common::pow6(x) + 51975*Common::pow4(x) - 945945*Common::pow2(x) + 2027025)*sin(x) + (9*x*(4*Common::pow6(x) - 770*Common::pow4(x) + 30030*Common::pow2(x) - 225225))*cos(x))/Common::pow9(x);
}

#if 0
/* Based on the Netlib routine (D)GAMMA by W. J. Cody and L. Stoltz.  Original
 * source and documentation available at
 *   http://netlib.org/specfun/gamma */
double Gamma(double x) {
    /* Constants */
    const double sqrtpi = 0.9189385332046727417803297;
    const double pi = 3.1415926535897932384626434;

    /* Numerator and denominator coefficients for rational minimax approximation over (1,2). */
    const double P[] = { -1.71618513886549492533811e0,
        2.47656508055759199108314e1, -3.79804256470945635097577e2,
        6.29331155312818442661052e2, 8.66966202790413211295064e2,
        -3.14512729688483675254357e4, -3.61444134186911729807069e4,
        6.64561438202405440627855e4 };
    const double Q[] = { -3.08402300119738975254353e1,
        3.15350626979604161529144e2, -1.01515636749021914166146e3,
        -3.10777167157231109440444e3, 2.25381184209801510330112e4,
        4.75584627752788110767815e3, -1.34659959864969306392456e5,
        -1.15132259675553483497211e5 };

    /* Coefficients for minimax approximation over (12,infty). */
    const double C[] = { -1.910444077728e-3,8.4171387781295e-4,
        -5.952379913043012e-4, 7.93650793500350248e-4,
        -2.777777777777681622553e-3, 8.333333333333333331554247e-2,
        5.7083835261e-3 };

    /* Machine dependent parameters (reasonable values here) */
    const double xbig = 171.624;
    const double xminin = 2.23e-308;
    const double eps = 2.22e-16;
    const double xinf = 1.79e308;

    double fact, res, y;
    int i;

    fact = 1;
    y = x;

    if(x <= 0) {
        y = -x;
        if(y - (int)y == 0) {
            /* x is a negative integer */
            return xinf;
        }
        else {
            /* Use the reflection formula Gamma(1-x) Gamma(x) = pi/sin(pi x) */
            fact = pi/sin(pi*x);
            y = 1 - x;
        }
    }

    /* y is now positive, and we seek Gamma(y) */

    if(y < eps) {
        /* 0 < y < eps: use limiting formula Gamma(y) -> 1/y */
        if(y >= xminin)
            res = 1/y;
        else
            res = xinf;
    }
    else if(y < 12) {
        /* eps < y < 12: use rational function approximation and recursion formula */
        int n = ((int)y) - 1;
        double z = y - (n + 1);
        double xnum, xden;

        /* Evaluate minimax approximation for Gamma(1+z) with 0 < z < 1 */
        xnum = 0;
        xden = 1;
        for(i = 0; i < 8; i++) {
            xnum = (xnum + P[i])*z;
            xden = xden*z + Q[i];
        }
        res = 1 + xnum/xden;

        /* Adjust result for y < 1 or y > 2 */
        if(n == -1)
            res /= y;
        else
            for(i = 0; i < n; i++)
                res *= (z + 1 + i);
    }
    else {
        /* y >= 12: use asymptotic expansion */
        if(y > xbig)
            res = xinf;
        else {
            double ysq = y*y;
            double sum = C[6];
            for(i = 0; i < 6; i++)
                sum = sum/ysq + C[i];
            sum = sum/y - y + sqrtpi + (y-0.5)*log(y);
            res = exp(sum);
        }
    }

    if(fact != 1)
        res = fact/res;
    return res;
}
#endif

#if 0

/******************************************************************************
 * Implementation of complete and incomplete gamma functions, following
 *   N. M. Temme, "A set of algorithms for the incomplete gamma function", 1994.
 * Adapted from the Pascal source code presented in that paper.  Results are
 * accurate to 9 significant digits.
 ******************************************************************************/

struct MachineConstants {
    double machtol, dwarf, giant;
    double sqrtgiant, sqrtdwarf, lndwarf, lnmachtol, sqrtminlnmachtol, oneoversqrt2mt,
           explow, sqrtminexplow, exphigh, sqrttwopi, lnsqrttwopi, sqrtpi, oneoversqrtpi;

    MachineConstants() {
        machtol = DBL_EPSILON;
        dwarf = DBL_MIN;
        giant = DBL_MAX;
        sqrtgiant = sqrt(giant);
        sqrtdwarf = sqrt(dwarf);
        lndwarf = log(dwarf);
        lnmachtol = log(machtol);
        sqrtminlnmachtol = sqrt(-lnmachtol);
        oneoversqrt2mt = 1/sqrt(2*machtol);
        explow = lndwarf;
        sqrtminexplow = sqrt(-explow);
        exphigh = log(giant);
        sqrttwopi = sqrt(2*M_PI);
        lnsqrttwopi = log(sqrttwopi);
        sqrtpi = sqrt(M_PI);
        oneoversqrtpi = 1/sqrtpi;
    }
};

static MachineConstants constants;

/* Compute the rational function
 *   a_m x^m + ... + a_1 x + a_0
 *   ---------------------------
 *   b_n x^n + ... + b_1 x + b_0 */
static double ratfun(double x, int m, double a[], int n, double b[]) {
    int k;
    double num = a[m], den = b[n];
    for(k = m-1; k >= 0; k--)
        num = num*x + a[k];
    for(k = n-1; k >= 0; k--)
        den = den*x + b[k];
    return num/den;
}

/* Compute e^x - 1 */
static double exmin1(double x) {
    static double ak[4] = { 9.999999998390e-1, 6.652950247674e-2, 2.331217139081e-2, 1.107965764952e-3 };
    static double bk[4] = { 1.000000000000e+0,-4.334704979491e-1, 7.338073943202e-2,-5.003986850699e-3 };

    if(x < constants.lnmachtol)
        return -1.;
    else if(x > constants.exphigh)
        return constants.giant;
    else if(x < -0.69 || x > 0.41)
        return exp(x) - 1.;
    else if(fabs(x) < constants.machtol)
        return x;
    else
        return x * ratfun(x, 3, ak, 3, bk);
}

/* Compute ln(1+x) - x */
static double auxln(double x) {
    static double ak[5] = {-4.999999994526e-1,-5.717084236157e-1,-1.423751838241e-1,-8.310525299547e-4, 3.899341537646e-5 };
    static double bk[4] = { 1.000000000000e+0, 1.810083408290e+0, 9.914744762863e-1, 1.575899184525e-1 };

    if(x <= -1.)
        return -constants.giant;
    else if(x < -0.70 || x > 1.36)
        return log(1+x) - x;
    else if(fabs(x) < constants.machtol)
        return -0.5*x*x;
    else {
        if(x > 0)
            return x*x*ratfun(x, 4, ak, 3, bk);
        else {
            double z = -x/(1+x);
            if(z > 1.36)
                return -(log(1+z) - z) + x*z;
            else
                return -z*z*ratfun(z, 4, ak, 3, bk) + x*z;
        }
    }
}

/* Compute Gamma^*(x) */
static double gammastar(double x) {
    static double ak12[3] = { 1.000000000949e+0, 9.781658613041e-1, 7.806359425652e-2 };
    static double bk12[2] = { 1.000000000000e+0, 8.948328926305e-1 };
    static double ak[4] = { 5.115471897484e-2, 4.990196893575e-1, 9.404953102900e-1, 9.999999625957e-1 };
    static double bk[4] = { 1.544892866413e-2, 4.241288251916e-1, 8.571609363101e-1, 1.000000000000e+0 };

    if(x > 1e10) {
        if(x > 1./(12.*constants.machtol))
            return 1.;
        else
            return 1. + 1./(12.*x);
    }
    else if(x >= 12.)
        return ratfun(1/x, 2, ak12, 1, bk12);
    else if(x >= 1.)
        return ratfun(x, 3, ak, 3, bk);
    else if(x > constants.dwarf)
        return gammastar(x+1) * sqrt(1+1/x) * exp(-1 + x*log(1+1/x));
    else
        return 1./(constants.sqrttwopi*constants.sqrtdwarf);
}

double Gamma(double x) {
    static double ak[5] = { 1.000000000000e+0,-3.965937302325e-1, 2.546766167439e-1,-4.880928874015e-2, 9.308302710346e-3 };
    static double bk[5] = {-1.345271397926e-1, 1.510518912977e+0,-6.508685450017e-1, 9.766752854610e-2,-5.024949667262e-3 };
    double a, g, s, dw;
    int j, k, m;

    if(x <= constants.dwarf)
        return 1./constants.dwarf;
    else {
        k = (int) round(x);
        m = (int) trunc(x);
        dw = (k == 0) ? constants.dwarf : (1+x)*constants.machtol;
        if(fabs(k - x) < dw && x <= 15.) {
            /* x = k  ==>  Gamma(x) = (k-1)! */
            g = 1.;
            for(j = 1; j <= k-1; j++)
                g *= j;
            return g;
        }
        else if(fabs((x-m)-0.5) < (1+x)*constants.machtol && x <= 15.) {
            /* x = m + 1/2  ==> use recursion and Gamma(0.5) = sqrt(pi) */
            g = constants.sqrtpi;
            for(j = 1; j <= m; j++)
                g *= (j-0.5);
            return g;
        }
        else if(x < 1.)
            return ratfun(x+2, 4, ak, 4, bk) / (x*(x+1));
        else if(x < 2.)
            return ratfun(x+1, 4, ak, 4, bk) / x;
        else if(x < 3.)
            return ratfun(x, 4, ak, 4, bk);
        else if(x < 10.) {
            g = 1.;
            while(x >= 3.) {
                x -= 1;
                g *= x;
            }
            return g * ratfun(x, 4, ak, 4, bk);
        }
        else if(x < constants.exphigh) {
            a = 1/(x*x);
            g = (1. + a*(-3.33333333333e-2 + a*9.52380952381e-3)) / (12.*x);
            a = -x + (x-0.5)*log(x) + g + constants.lnsqrttwopi;
            return (a < constants.exphigh) ?  exp(a) : constants.giant;
        }
        else {
        }
    }
}

/* Compute the function g(x) in the representation
 *   1/Gamma(1+x) = 1 + x*(x-1)*g(x) */
static double auxgam(double x) {
    static double ak[4] = {-5.772156647338e-1,-1.087824060619e-1, 4.369287357367e-2,-6.127046810372e-3 };
    static double bk[5] = { 1.000000000000e+0, 3.247396119172e-1, 1.776068284106e-1, 2.322361333467e-2, 8.148654046054e-3 };

    if(x <= -1.)
        return -0.5;
    else if(x < 0.)
        return -(1 + (x+1)*(x+1)*ratfun(x+1, 3, ak, 4, bk)) / (1-x);
    else if(x <= 1.)
        return ratfun(x, 3, ak, 4, bk);
    else if(x <= 2.)
        return ((x-2)*ratfun(x-1, 3, ak, 4, bk) - 1) / (x*x);
    else
        return (1/Gamma(x+1) - 1.) / (x*(x-1));
}

/* Compute ln Gamma(x) */
static double lngamma(double x) {
    static double ak4[5] =  {-2.12159572323e5, 2.30661510616e5, 2.74647644705e4,-4.02621119975e4,-2.29660729780e3 };
    static double bk4[5] =  {-1.16328495004e5,-1.46025937511e5,-2.42357409629e4,-5.70691009324e2, 1.00000000000e0 };
    static double ak15[5] = {-7.83359299449e1,-1.42046296688e2, 1.37519416416e2, 7.86994924154e1, 4.16438922228 };
    static double bk15[5] = { 4.70668766060e1, 3.13399215894e2, 2.63505074721e2, 4.33400022514e1, 1.00000000000 };
    static double ak0[5] =  {-2.66685511495,  -2.44387534237e1,-2.19698958928e1, 1.11667541262e1, 3.13060547623 };
    static double bk0[5] =  { 6.07771387771e-1,1.19400905721e1, 3.14690115749e1, 1.52346874070e1, 1.00000000000 };

    double a, g, y;
    if(x > 12.) {
        g = 1./(12.*x);
        a = -x + (x-0.5)*log(x) + constants.lnsqrttwopi;
        if(a + g == a)
            return a;
        y = 1./(x*x);
        return a + g*(1. + y*(-3.33333333333e-2 + y*9.52380952381e-3));
    }
    else if(x >= 4.)
        return ratfun(x, 4, ak4, 4, bk4);
    else if(x > 1.5)
        return (x-2) * ratfun(x, 4, ak15, 4, bk15);
    else if(x >= 0.5)
        return (x-1) * ratfun(x, 4, ak0, 4, bk0);
    else if(x > constants.machtol)
        return -log(x) + x*ratfun(x+1, 4, ak0, 4, bk0);
    else if(x > constants.dwarf)
        return -log(x);
    else
        return -constants.lndwarf;
}

/* Compute the normalized (upper) incomplete gamma function
 *   Q(a,x) = \Gamma(a,x) / \Gamma(a) */
static double incomgam(double a, double x, double eps = 1e-10) {
    double lnx, mu, auxlnmu, dp, pqasymp;
    if(a == 0. && x == 0.)
        return 0.5;
    else if(x == 0.)
        return 1.;
    else if(a == 0.)
        return 0.;
    else {
        lnx = (x <= constants.dwarf) ? constants.lndwarf : log(x);

        /* function dax */
        mu = (x-a)/a;
        auxlnmu = auxln(mu);
        dp = a*auxlnmu - 0.5*log(2*M_PI*a);
        if(dp < constants.explow)
            dp = 0.;
        else
            dp = exp(dp) / gammastar(a);

        if(dp >= constants.dwarf) {
            if(a > 25. && fabs(mu) < 0.2)
                return pqasymp();
            else if(a > alfa(x))
                return ptaylor();
            else if(x < 1.)
                return qtaylor();
            else
                return qfraction();
        }
        else
            return (a > x) ? 1. : 0.;
    }
}

double Gamma(double x) {
    return tgamma(x);
}

double LowerGamma(double a, double x) {
    if(x < 0.1)
        return pow(x,a)*(1/a - x/(a+1) + 0.5*x*x/(a+2));
    else
        return gsl_sf_gamma(a) - gsl_sf_gamma_inc(a, x);
}

double UpperGamma(double a, double x) {
    return gsl_sf_gamma_inc(a, x);
}
#endif // 0

double Gamma(double x) {
    /* Use POSIX call for simplicity */
    return tgamma(x);
}

double LogGamma(double x) {
    /* Use POSIX call for simplicity */
    return lgamma(x);
}


#if 0
/* Modified from http://people.sc.fsu.edu/~jburkardt/c_src/asa239/asa239.c .
 * Original comments follow:
 *    Purpose:
 *      ALNORM computes the cumulative density of the standard normal distribution.
 *    Licensing:
 *      This code is distributed under the GNU LGPL license. 
 *    Modified:
 *      13 November 2010
 *    Author:
 *      Original FORTRAN77 version by David Hill.
 *      C version by John Burkardt.
 *    Reference:
 *      David Hill,
 *      Algorithm AS 66:
 *      The Normal Integral,
 *      Applied Statistics,
 *      Volume 22, Number 3, 1973, pages 424-427.
 *    Parameters:
 *     -Input, double X, is one endpoint of the semi-infinite interval
 *      over which the integration takes place.
 *     -Input, int UPPER, determines whether the upper or lower
 *      interval is to be integrated:
 *      1  => integrate from X to + Infinity;
 *      0 => integrate from - Infinity to X.
 *     -Output, double ALNORM, the integral of the standard normal
 *      distribution over the desired interval.  */
static double asa239_alnorm(double x, int upper) {
  double a1 = 5.75885480458;
  double a2 = 2.62433121679;
  double a3 = 5.92885724438;
  double b1 = -29.8213557807;
  double b2 = 48.6959930692;
  double c1 = -0.000000038052;
  double c2 = 0.000398064794;
  double c3 = -0.151679116635;
  double c4 = 4.8385912808;
  double c5 = 0.742380924027;
  double c6 = 3.99019417011;
  double con = 1.28;
  double d1 = 1.00000615302;
  double d2 = 1.98615381364;
  double d3 = 5.29330324926;
  double d4 = -15.1508972451;
  double d5 = 30.789933034;
  double ltone = 7.0;
  double p = 0.398942280444;
  double q = 0.39990348504;
  double r = 0.398942280385;
  int up;
  double utzero = 18.66;
  double value;
  double y;
  double z;

  up = upper;
  z = x;

  if ( z < 0.0 )
  {
    up = !up;
    z = - z;
  }

  if ( ltone < z && ( ( !up ) || utzero < z ) )
  {
    if ( up )
    {
      value = 0.0;
    }
    else
    {
      value = 1.0;
    }
    return value;
  }

  y = 0.5 * z * z;

  if ( z <= con )
  {
    value = 0.5 - z * ( p - q * y
      / ( y + a1 + b1
      / ( y + a2 + b2
      / ( y + a3 ))));
  }
  else
  {
    value = r * exp ( - y )
      / ( z + c1 + d1
      / ( z + c2 + d2
      / ( z + c3 + d3
      / ( z + c4 + d4
      / ( z + c5 + d5
      / ( z + c6 ))))));
  }

  if ( !up )
  {
    value = 1.0 - value;
  }

  return value;
}
#endif



/* Modified from http://people.sc.fsu.edu/~jburkardt/c_src/asa239/asa239.c .
 * Original comments follow:
 *    Purpose:
 *      GAMMAD computes the Incomplete Gamma Integral
 *    Licensing:
 *      This code is distributed under the GNU LGPL license. 
 *    Modified:
 *      13 November 2010
 *    Author:
 *      Original FORTRAN77 version by B Shea.
 *      C version by John Burkardt.
 *    Reference:
 *      B Shea,
 *      Algorithm AS 239:
 *      Chi-squared and Incomplete Gamma Integral,
 *      Applied Statistics,
 *      Volume 37, Number 3, 1988, pages 466-473.
 *    Parameters:
 *     -Input, double X, P, the parameters of the incomplete
 *      gamma ratio.  0 <= X, and 0 < P.
 *     -Output, int IFAULT, error flag.
 *      0, no error.
 *      1, X < 0 or P <= 0.
 *     -Output, double GAMMAD, the value of the incomplete
 *      Gamma integral. */
/* Note: alnorm(x) = 0.5*(1 + erf(x/sqrt(2)) */
static double asa239_gammad(double x, double p, int *ifault) {
    const double elimit = -88.0;
    const double oflo = 1.0E+37;
    const double plimit = 1000.0;
    const double tol = 1.0E-14;
    const double xbig = 1.0E+08;
    double a;
    double an;
    double arg;
    double b;
    double c;
    double pn1, pn2, pn3, pn4, pn5, pn6;
    double rn;
    double value = 0.0;

    value = 0.0;
    *ifault = 0;

    /* Check the input. */
    if(x < 0.0 || p <= 0.0) {
        *ifault = 1;
        return 0;
    }

    if(x == 0.0)
        return 0;

    /* If P is large, use a normal approximation. */
    if(p > plimit) {
        pn1 = 3*sqrt(p) * ( pow(x/p, 1/3.) + 1/(9*p) - 1 );
        value = 0.5 + 0.5*erf(pn1/M_SQRT2);
        return value;
    }

    /* If X is large set value = 1. */
    if(xbig < x)
        return 1;

    if(x <= 1.0 || x < p) {
        /* Use Pearson's series expansion. */

        arg = p*log(x) - x - lgamma(p + 1);
        c = 1.0;
        value = 1.0;
        a = p;

        while(c > tol) {
            a = a + 1;
            c = c * x / a;
            value += c;
        }

        arg += log(value);

        if(arg >= elimit)
            value = exp(arg);
        else
            value = 0.0;
    }
    else {
        /* Use a continued fraction expansion. */

        arg = p*log(x) - x - lgamma(p);
        a = 1 - p;
        b = a + x + 1;
        c = 0.0;
        pn1 = 1.0;
        pn2 = x;
        pn3 = x + 1;
        pn4 = x * b;
        value = pn3 / pn4;

        for ( ; ; ) {
            a += 1;
            b += 2;
            c += 1;
            an = a * c;
            pn5 = b * pn3 - an * pn1;
            pn6 = b * pn4 - an * pn2;
   
            if(pn6 != 0.0) {
                rn = pn5 / pn6;
     
                if(fabs(value - rn) <= fmin(tol, tol * rn))
                    break;
                value = rn;
            }
   
            pn1 = pn3;
            pn2 = pn4;
            pn3 = pn5;
            pn4 = pn6;
   
            /* Re-scale terms in continued fraction if terms are large. */
            if(fabs(pn5) >= oflo) {
                pn1 = pn1 / oflo;
                pn2 = pn2 / oflo;
                pn3 = pn3 / oflo;
                pn4 = pn4 / oflo;
            }
        }

        arg += log(value);

        if(arg >= elimit)
            value = 1 - exp(arg);
        else
            value = 1;
    }

    return value;
}

double LowerGamma(double a, double x) {
    int ifault;
    return Gamma(a) * asa239_gammad(x, a, &ifault);
}

double UpperGamma(double a, double x) {
    int ifault;
    return Gamma(a) * (1 - asa239_gammad(x, a, &ifault));
}
