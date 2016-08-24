from matplotlib import pyplot as plt
import os

def new_axes(ylabel, xlims=None, ylims=None, nticks=5):
    """
    Return a new, formatted axes
    """
    from matplotlib.ticker import AutoMinorLocator
    plt.clf()
    ax = plt.gca()
    
    # axes limits
    if xlims is not None: ax.set_xlim(xlims)
    if ylims is not None: ax.set_ylim(ylims)

    # axes labels
    ax.set_xlabel(r"$k$ [$h$/Mpc]", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    
    # add minor ticks
    ax.xaxis.set_minor_locator(AutoMinorLocator(nticks))
    ax.yaxis.set_minor_locator(AutoMinorLocator(nticks))
    
    return plt.gcf(), ax

def savefig(fig, f, dirname, filename, dpi=200):
    """
    Save the input figure
    """
    # the output dir
    currdir = os.path.split(f)[0]
    d = os.path.join(currdir, 'figures')
    if dirname: d = os.path.join(d, dirname)
    if not os.path.exists(d): os.makedirs(d)
        
    # save
    filename = os.path.join(d, filename)
    fig.savefig(filename, dpi=dpi)
    
def teardown_module(module):
    """
    Teardown the module by syncing to NERSC
    """
    thismod = module.__name__.split('.')[-2]
    remote_dir = os.path.join("/project/projectdirs/m779/www/nhand/pyRSDTests", thismod)
    
    cmd = "rsync -e ssh -avzl --progress --delete"
    cmd += " --exclude='.*'"
    
    # add the directories and run the command
    host = 'edison'
    cmd += " figures/* nhand@%s:%s/" %(host, remote_dir)
    print("executing command:", cmd)
    ret = os.system(cmd)
    
    print("teardown_module   module:%s" % module.__name__)