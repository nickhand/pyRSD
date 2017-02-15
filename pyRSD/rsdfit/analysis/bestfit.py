"""
    bestfit.py
    rsdfit.analyze

    __author__ : Nick Hand
    __email__  : nhand@berkeley.edu
    __desc__   : class to store and manipulate a set of bestfit parameters
"""
import pandas as pd
from ... import numpy
from . import tex_names
from .tools import format_scaled_param

#------------------------------------------------------------------------------
# format functions
#------------------------------------------------------------------------------
def format_scale(row):
    name, number = row['index'], row['scale']
    if number == 1:
        return name
        
    if number < 1000 and number > 1:
        name = "%0.d_%s" %(number, name)
    else:
        name = "%.2e_%s" %(number, name)
    return name
        
def median_plus_error_fmt(row, fmt):
    """
    Function to apply to a `pandas.DataFrame` to transform the `median`,
    `upper_1sigma`, and `lower_1sigma` into a median +/- errors formatted
    string
    """
    args = (row['median'], row['upper_1sigma'], abs(row['lower_1sigma']))
    if fmt == 'latex': 
        return r"\num{%.7g} \numpmerr{%.7g}{%.7g}" %args
    elif fmt == 'ipynb':
        return r"$%.5g^{+%.2g}_{-%.2g}$" %args
    else:
        raise NotImplementedError("output format `%s` is not valid for `median_plus_error`" %fmt)
    
def default_fmt(row, col):
    return r"\num{%.5g}" %row[col]
    
def two_sigma_fmt(row, col):
    return r"\num{%.5g}" %(row['median']+row[col])
 
#------------------------------------------------------------------------------
# transfromation functions
#------------------------------------------------------------------------------
def transform_latex(df):
    """
    Transform the input `DataFrame` (or subclass) to a `DataFrame` with columns
    appropriate for being output as a latex table
    
    Parameters
    ----------
    df : DataFrame, or subclass
        the frame holding the data which will be formatted 
    
    Returns
    -------
    toret : DataFrame
        the frame holding the formatted data
    """
    cols = ['median', 'lower_1sigma', 'upper_1sigma', 'best_fit', 'lower_2sigma', 'upper_2sigma']
    if not all(col in df for col in cols):
        raise ValueError("all columns %s must be present to transform to latex table" %str(cols))
    
    # to return
    toret = pd.DataFrame(index=df.index)
    
    # format the columns
    toret["best-fit"] = df.apply(default_fmt, args=('best_fit',), axis=1)
    toret["median $\pm$ $1\sigma$"] = df.apply(median_plus_error_fmt, args=('latex',), axis=1)
    toret["95\% lower"] = df.apply(two_sigma_fmt, args=('lower_2sigma',), axis=1)
    toret["95\% upper"] = df.apply(two_sigma_fmt, args=('upper_2sigma',), axis=1)
        
    # use the tex_name for an index if you can
    if 'tex_name' in df:
        toret = toret.set_index(df['tex_name'])
        
    return toret
    
def transform_ipynb(df):
    """
    Transform the input `DataFrame` (or subclass) to a `DataFrame` with columns
    appropriate for being output as an html table
    
    Parameters
    ----------
    df : DataFrame, or subclass
        the frame holding the data which will be formatted 
    
    Returns
    -------
    toret : DataFrame
        the frame holding the formatted data
    """
    cols = ['median', 'lower_1sigma', 'upper_1sigma', 'best_fit', 'lower_2sigma', 'upper_2sigma']
    if not all(col in df for col in cols):
        raise ValueError("all columns %s must be present to transform to latex table" %str(cols))
    
    # to return
    toret = pd.DataFrame(index=df.index)
    
    # format the columns
    toret["best-fit"] = df['best_fit']
    
    if not df['median'].isnull().all():
        toret["median $\pm$ $1\sigma$"] = df.apply(median_plus_error_fmt, args=('ipynb',), axis=1)
    
        if not df['lower_2sigma'].isnull().all():
            toret["95% lower"] = df['median'] + df['lower_2sigma']
        if not df['upper_2sigma'].isnull().all():
            toret["95% upper"] = df['median'] + df['upper_2sigma']
        
    # use the tex_name for an index if you can
    if 'tex_name' in df:
        toret = toret.set_index(df['tex_name'])
    else:
        if 'scale' in df and (df['scale'] != 1.0).sum():
            df['index'] = df.index
            toret['index'] = df.apply(format_scale, axis=1)
            toret = toret.set_index('index')
    toret.index.name = 'parameter'

    return toret.sort_index()
    
   
#------------------------------------------------------------------------------
# main class/functions
#------------------------------------------------------------------------------ 
def to_comparison_table(names, data, filename=None, params=None, fmt='latex', 
                            add_reduced_chi2=True, tabular_only=False):
    """
    Write out a comparison table from multiple BestfitParameterSet instances
    """
    if len(data) < 2:
        raise ValueError("data length in ``to_comparison_table`` must be greater than 1")
    if len(names) != len(data):
        raise ValueError("shape mismatch between specified column names and supplied data")
        
    # the names of the columns in the table
    chi2_meta = ['min_minus_lkl', 'Np', 'Nb']
    
    # slice the data and add the columns
    red_chi2 = []
    for i, name in enumerate(names):
        df = data[i]
        
        # check tex name
        if 'tex_name' not in df:
            for n in df.index:
                if n in tex_names:
                    df.loc[n, 'tex_name'] = format_scaled_param(tex_names[n], 1.0 / df.loc[n, 'scale'])                        
        
        # check scale
        if 'scale' in df and (df['scale'] != 1.0).sum():
            df['index'] = df.index
            df['index'] = df.apply(format_scale, axis=1)
            df = df.set_index('index')
        
        if params == 'free':
            if not 'free' in df:
                raise ValueError('cannot output only `free` parameters with no `free` column')
            df = df.loc[df.free == True]
        elif params == 'constrained':
            df = df.loc[df.free == False]
        elif params is not None:
            trim = [p for p in params if p in df.index]
            df = df.loc[trim]
        
        # initialize the output set, if we haven't yet
        if i == 0:
            out = BestfitParameterSet(index=df.index)
            
        # add the columms
        if not df['median'].isnull().all():
            out[name] = df.apply(median_plus_error_fmt, args=(fmt,), axis=1)
        else:
            out[name] = df['best_fit']
        
        if add_reduced_chi2 and all(hasattr(df, col) for col in chi2_meta):
            red_chi2 = 2*df.min_minus_lkl/(df.Nb-df.Np)
            args = (2*df.min_minus_lkl, df.Nb, df.Np, red_chi2)
            out.loc['red_chi2', name] = "$%d/(%d - %d) = %.2f$" %args
            
        if 'tex_name' in df and 'tex_name' not in out:
            out['tex_name'] = df['tex_name']
    
    if add_reduced_chi2 and "red_chi2" in out.index and "tex_name" in out:
        out.loc["red_chi2", 'tex_name'] = r'$\chi^2$/d.o.f.'
        
    # sort by param name first
    if not isinstance(params, list): 
        out = out.sort_index()

    # move red_chi2 column to first
    if add_reduced_chi2 and "red_chi2" in out.index:
        index = out.index.values.tolist()
        i = index.index("red_chi2")
        first = index.pop(i) 
        out = out.reindex([first] + index)
    
    # index by tex_name?
    if 'tex_name' in out:
        out = out.set_index(out['tex_name'])

    if fmt == 'latex':
        pd.set_option('max_colwidth', 1000)
        kwargs = {'escape':False, 'index_names':False, 'columns':names}
        toret = pd.DataFrame.to_latex(out, **kwargs)
        return out._finalize_latex(toret, len(names), filename=filename, 
                    include_fit_info=False, tabular_only=tabular_only)
    else:
        out.index.name = 'parameter'
        return out.loc[:, names]


class BestfitParameterSet(pd.DataFrame):
    """
    A subclass of ``pandas.DataFrame`` to store and manipulate a 
    set of best-fit parameters from an RSD fit
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize with constructor signature that matches 
        `pandas.DataFrame`
        
        Parameters
        ----------
        data : numpy ndarray (structured or homogeneous), dict, or DataFrame
            Dict can contain Series, arrays, constants, or list-like objects
        index : Index or array-like
            Index to use for resulting frame. Will default to numpy.arange(n) if
            no indexing information part of input data and no index provided
        columns : Index or array-like
            Column labels to use for resulting frame. Will default to
            numpy.arange(n) if no column labels are provided
        dtype : dtype, default None
            Data type to force, otherwise infer
        copy : boolean, default False
            Copy data from inputs. Only affects DataFrame / 2d ndarray input
        """
        super(BestfitParameterSet, self).__init__(*args, **kwargs)
        self.__dict__['_user_metadata'] = []
    
    #--------------------------------------------------------------------------
    # internal wraps
    #--------------------------------------------------------------------------
    def __finalize__(self, other, method=None, **kwargs):
        if hasattr(other, '_metadata'):
            for name in other._metadata:
                object.__setattr__(self, name, getattr(other, name, None))
        return self
        
    @property
    def _metadata(self):
        return self.__dict__['_user_metadata'] + ['latex_cols', 'latex_fmts', '_user_metadata']
    
    @property
    def _constructor(self):
        return BestfitParameterSet

    def add_metadata(self, **meta):
        """
        Attach the keyword arguments to the `BestfitParameterSet` as attributes
        and keep track of names in `_user_metadata`
        """
        for k, v in meta.items():
            self._user_metadata.append(k)
            setattr(self, k, v)
            
    #--------------------------------------------------------------------------
    # class method constructors
    #--------------------------------------------------------------------------     
    @classmethod   
    def from_info(cls, filename):
        """
        Initialize the `BestfitParameterSet` class from a plaintext ``info`` file
        
        Parameters
        ----------
        filename : str
            the name of the `info` file to load
        """
        lines = open(filename, 'r').readlines()
        
        # read any metadata
        meta = {}
        for line in lines:
            if line[0] == '#':
                fields = line[1:].split()
                cast = fields[-1]
                if cast in __builtins__:
                    meta[fields[0]] = __builtins__[cast](fields[1])
                elif hasattr(numpy, cast):
                     meta[fields[0]] = getattr(numpy, cast)(fields[1])
                else:
                    raise TypeError("metadata must have builtin or numpy type")          
        
        # parse the file, ignoring comments
        df = pd.read_csv(filename, delim_whitespace=True, comment='#')
        toret = cls(df)
        if len(meta): toret.add_metadata(**meta)
        return toret
    
    @classmethod   
    def from_nlopt(cls, r, **meta):
        """
        Initialize the `BestfitParameterSet` class from an `LBFGSResults`
        instance from a nonlinear optimization fit using LBFGS
        
        Parameters
        ----------
        r : LBFGSResults
            an instance holding the best-fit parameters from an nlopt LBFGS fit
        **meta : kwargs
            any additional keywords to attach as meta-data
        """
        # initialize empty data frame
        index = pd.Index(r.free_names+r.constrained_names)
        columns = ['best_fit', 'median', 'lower_1sigma', 'upper_1sigma', 
                    'lower_2sigma', 'upper_2sigma', 'free', 'scale', 
                    'gt_1sigma', 'gt_2sigma', 'lt_1sigma', 'lt_2sigma']
        toret = cls(columns=columns, index=index)
        
        # bestfits
        toret.loc[r.free_names, 'best_fit'] = r.min_chi2_values
        toret.loc[r.constrained_names, 'best_fit'] = numpy.array([r.min_chi2_constrained_values[name] for name in r.constrained_names])
        
        # free vs constrained
        toret.loc[r.free_names, 'free'] = True
        toret.loc[r.constrained_names, 'free'] = False
        
        # set the scale
        toret['scale'] = 1.0

        if len(meta): toret.add_metadata(**meta)
        return toret


    @classmethod   
    def from_mcmc(cls, r, **meta):
        """
        Initialize the `BestfitParameterSet` class from an `EmceeResults`
        instance from an MCMC fit using `emcee`
        
        Parameters
        ----------
        r : EmceeResults
            an instance holding the best-fit parameters from an emcee mcmc fit
        **meta : kwargs
            any additional keywords to attach as meta-data
        """
        # initialize empty data frame
        index = pd.Index(r.free_names+r.constrained_names)
        
        columns = ['best_fit', 'median', 'lower_1sigma', 'upper_1sigma', 
                    'lower_2sigma', 'upper_2sigma', 'free', 'scale', 
                    'gt_1sigma', 'gt_2sigma', 'lt_1sigma', 'lt_2sigma']
        toret = cls(columns=columns, index=index)
        
        # bestfits
        toret.loc[r.free_names, 'best_fit'] = r.max_lnprob_values()
        toret.loc[r.constrained_names, 'best_fit'] = r.max_lnprob_constrained_values()
        
        # do free and constrained params
        for params, is_free in zip([r.free_names, r.constrained_names], [True, False]):
        
            # median
            toret.loc[params, 'median'] = [r[p].median for p in params]
           
            # 1-sigmas
            one_sigma = numpy.array([r[p].one_sigma for p in params])
            toret.loc[params, 'lower_1sigma'] = one_sigma[:,0]
            toret.loc[params, 'upper_1sigma'] = one_sigma[:,1]
        
            # 2-sigmas
            two_sigma = numpy.array([r[p].two_sigma for p in params])
            toret.loc[params, 'lower_2sigma'] = two_sigma[:,0]
            toret.loc[params, 'upper_2sigma'] = two_sigma[:,1]
            
            # free
            toret.loc[params, 'free'] = is_free
        
        toret['scale'] = 1.0
        toret['sigma'] = 0.5*(abs(toret['lower_1sigma']) + toret['upper_1sigma'])
        toret['gt_1sigma'] = toret['median'] + toret['lower_1sigma']
        toret['lt_1sigma'] = toret['median'] + toret['upper_1sigma']
        toret['gt_2sigma'] = toret['median'] + toret['lower_2sigma']
        toret['lt_2sigma'] = toret['median'] + toret['lower_2sigma']

        if len(meta): toret.add_metadata(**meta)
        return toret

    #--------------------------------------------------------------------------
    # output methods
    #--------------------------------------------------------------------------
    def to_info(self, filename):
        """
        Write out the parameter set as a plaintext ``info`` file
        
        Parameters
        ----------
        filename : str
            the name of the file to write to
        """
        if len(self._user_metadata):
            with open(filename, 'w') as ff:
                for k in self._user_metadata:
                    val = getattr(self, k)
                    ff.write("# %s %s %s\n" %(k, val, type(val).__name__))
        self.to_csv(filename, sep=" ", mode='a')
        
    def _finalize_latex(self, table, Ncols, filename=None, caption="", include_fit_info=True, tabular_only=False):
        """
        Internal function to format the default latex table and output
        """
        # sort out the meta info
        needed_meta = ['min_minus_lkl', 'Np', 'Nb']
        if include_fit_info and not (hasattr(self, col) for col in needed_meta):
            raise ValueError("need all of %s attributes to include fit info in table" %str(needed_meta))
            
        # do some formatting of the standard pandas `to_latex` output
        table = table.replace("".join(['l']*(Ncols+1)), 'c'+"".join(['l']*(Ncols)))
        
        # add a header
        header = ""
        if not tabular_only:
            header = "\\begin{table}\n\\centering\n"
            precision = "\\sisetup{round-mode=places, round-precision=3}\n"
            header += precision
        #header += 
        
        if include_fit_info: header += "\\begin{threeparttable}\n"
        table = header + table
        
        # add a footer
        note = ""
        if include_fit_info:
            args = (self.min_minus_lkl, self.Np, self.Nb, 2*self.min_minus_lkl/(self.Nb-self.Np))
            note = r'$-\ln\mathcal{L} = %.2f$, $N_p = %d$, $N_b = %d$, $\chi^2_\mathrm{red} = %.2f$' %args
            note = "\\begin{tablenotes}\n\t\item %s\n\\end{tablenotes}\n" %note
            note += "\\end{threeparttable}\n"
        table = table + note
        
        if not tabular_only:
            table += "\\caption{%s}\n\\label{tab:}\n" %caption + "\\end{table}"
        
        if filename is None:
            return table
        else:
            with open(filename, 'w') as ff:
                ff.write(table)
        
    def to_latex(self, filename=None, caption="", include_fit_info=True, tabular_only=False):
        """
        Write out the bestfit parameters as a latex table
        
        Parameters
        ----------
        filename : str or None, optional (`None`)
            the name of the file to write to, or if `None`, simply return
            the output string
        caption : str, optional (`""`)
            an optional string which is intepreted as the caption for the table
        include_fit_info : bool, optional (`True`)
            if `True`, include information about the best log-likelihood, number 
            of parameters, number of data points, and reduced chi2 as a footer
            of the table
        """
        # get the string output
        df = transform_latex(self)
        pd.set_option('max_colwidth', 1000)
        kwargs = {'escape':False, 'index_names':False}
        out = df.to_latex(**kwargs)
        
        # finalize
        Ncols = len(df.columns)
        kw = {'filename':filename, 'caption':caption, 
                'include_fit_info':include_fit_info, 'tabular_only':tabular_only}
        return self._finalize_latex(out, Ncols, **kw)
        
    def to_ipynb(self, params=None):
        """
        Return a table representation in html format, suitable for nice
        viewing in `ipython notebook`
        
        Parameters
        ----------
        params : list of str, optional (`None`)
            restrict the output table to only include the specified parameters
        """
        if params is None:
            return transform_ipynb(self)
        else:
            valid = [p for p in params if p in self.index]
            if not len(valid):
                raise ValueError("no valid parameter names in specified `params`...try again")
            return transform_ipynb(self.loc[valid])
        
        
        
        
    