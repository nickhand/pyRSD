"""
 plot_integral.py
 plot the specificied PT integral
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 11/25/2014
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np

def main(args):
    
    # read the data file
    data = np.loadtxt("data/%s.dat" %args.integral)
   
    # plot
    plt.loglog(data[:,0], abs(data[:,1]))
    
    # make it look nice
    indices = args.integral[1:]
    plt.xlabel(r"$k$ ($h$/Mpc)", fontsize=16)
    plt.ylabel(r"$|%s_{%s}(k)|$" %(args.integral[0], indices), fontsize=16)
    plt.savefig("plots/%s.pdf" %args.integral)
    plt.show()
    
#-------------------------------------------------------------------------------
if __name__ == '__main__':
    
    # parse the input arguments
    parser = argparse.ArgumentParser(description="plot the specificied PT integral")
    parser.add_argument('integral', type=str, help="the name of the integral to plot") 
    args = parser.parse_args()
    
    main(args)