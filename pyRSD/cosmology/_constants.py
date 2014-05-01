"""
 constants.py
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 04/10/2012
"""

from numpy import pi

# set up cgs units
cm      = 1.0
gram    = 1.0
second  = 1.0
erg     = 1.0
kelvin  = 1.0
radian = 1.0    

# prefixes
giga  = 1e9
mega  = 1e6
kilo  = 1e3
centi = 1e-2
milli = 1e-3
micro = 1e-6
nano  = 1e-9
pico  = 1e-12

# fundamental constants
h_planck = 6.62606957e-27     # planck constant
h_bar    = h_planck/(2.*pi)   # reduced planck constant
c_light  = 2.99792458e10      # speed of light
k_b      = 1.3806488e-16      # Boltzmann constant
m_p      = 1.672621777e-24    # proton mass
m_e      = 9.10938291e-28     # electron mass
q_e      = 4.80320425e-10     # e.s.u.
G        = 6.67384e-8         # Newton's constant
eV       = 1.60217657e-12     # ergs
N_a      = 6.02214129e23      # Avogadro's constant
sigma_sb = 5.670373e-5        # Stefan-Boltzmann constant
a_rad    = 4*sigma_sb/c_light # radiation constant
sigma_T  = 6.652458734e-25    # Thomson cross section
T_cmb    = 2.72528            # temperature of the CMB
H_0      = 100.0              # in units of h km/s/Mpc
a_0      = 5.2917721092e-9    # Bohr radius

# length conversion factors
km       = 1.0e+5 * cm  
meter    = 1.0e+2 * cm  
inch     = 2.54   * cm  
mm       = 0.1    * cm  
micron   = 1.0e-4 * cm  
angstrom = 1e-8   * cm
jansky   = 1.0e-23            # erg/s/cm/cm/Hz
barn     = 1e-24  * cm**2

# energy and power conversion factors
joule           = 1e7 * erg
watt            = 1e7 * erg / second
rydberg         = 10973731.568539 * meter**(-1.) * h_planck*c_light

# astronomical constants
au        = 149597870700*meter      # the astronomical unit
degree    = pi/180.
arcminute = degree/60.
arcsecond = arcminute/60.            # radians
parsec    = au / arcsecond          # cm
minute    = 60.                     # seconds
hour      = 60. * 60.               # seconds
day       = 8.64e4                  # seconds
year      = 365.2425 * day          # seconds
Mpc       = mega*parsec
lyr       = c_light*year            # light year
                                                                    
L_sun = 3.826e33              # erg/s
M_sun = 1.9891e33             # g
R_sun = 6.9598e10             # cm
T_sun = 5770.0                # kelvin


M_earth   = 5.976e27            # Earth mass in g
R_earth   = 6371 * km           # Earth's equatorial radius in cm 
M_jupiter = 1898.8e27           # Jupiter in g
R_jupiter = 70850 * km          # Jupiter's equatorial radius in cm


# AB magnitude zero point
AB_flux_zero = 3631 # in Janskys

