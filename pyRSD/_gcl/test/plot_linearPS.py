"""
 plot_linearPS.py
 plot the linear PS test comparison
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 11/24/2014
"""
import matplotlib.pyplot as plt
import numpy as np

def main():
    
    # read the data file
    data = np.loadtxt("data/test_linearPS.dat")

    
    #CLASS
    plt.loglog(data[:,0], data[:,1], label='CLASS')
    
    # EH full transfer
    plt.loglog(data[:,0], data[:,2], label='EH full transfer')
    
    # EH no wiggle transfer
    plt.loglog(data[:,0], data[:,3], label='EH no-wiggle transfer')
    
    # BBKS transfer
    plt.loglog(data[:,0], data[:,4], label='BBKS transfer')
    
    # make it look nice
    plt.xlabel(r"$k$ ($h$/Mpc)", fontsize=16)
    plt.ylabel(r"$P^X_\mathrm{lin} / P^\mathrm{CLASS}_\mathrm{lin} (k)$", fontsize=16)
    plt.legend(loc=0)
    plt.savefig("plots/test_linearPS.pdf")
    plt.show()
    
#-------------------------------------------------------------------------------
if __name__ == '__main__':
    main()