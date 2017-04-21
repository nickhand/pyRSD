#ifndef SPECIALFUNCTIONS_H
#define SPECIALFUNCTIONS_H

/* Cylindrical Bessel functions */
double BesselJ0(double x);
double BesselJ1(double x);
double BesselJn(int n, double x);

/* Spherical Bessel functions */
double SphericalBesselJ0(double x);
double SphericalBesselJ1(double x);
double SphericalBesselJ2(double x);
double SphericalBesselJ3(double x);
double SphericalBesselJ4(double x);
double SphericalBesselJ6(double x);
double SphericalBesselJ8(double x);

/* Gamma function:
 *   \Gamma(a) = \int_0^\infty t^{a-1} e^{-t} dt */
double Gamma(double a);

/* Natural logarithm of gamma function:
 *   \log \Gamma(a) */
double LogGamma(double a);

/* Unnormalized lower incomplete gamma function:
 *  \gamma(a,x) = \int_0^x t^{a-1} e^{-t} dt */
double LowerGamma(double a, double x);

/* Unnormalized upper incomplete gamma function:
 *  \Gamma(a,x) = \int_x^\infty t^{a-1} e^{-t} dt */
double UpperGamma(double a, double x);

#endif // SPECIALFUNCTIONS_H
