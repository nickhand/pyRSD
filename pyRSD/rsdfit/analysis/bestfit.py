"""
    bestfit.py
    rsdfit.analyze

    __author__ : Nick Hand
    __email__  : nhand@berkeley.edu
    __desc__   : class to store and manipulate a set of bestfit parameters
"""
import pandas as pd
from ... import numpy

#------------------------------------------------------------------------------
# latex format functions
#------------------------------------------------------------------------------

def median_plus_error_fmt(row):
    args = (row['median'], row['upper_1sigma'], abs(row['lower_1sigma']))
    return r"\num{%.7g} \numpmerr{%.7g}{%.7g}" %args
    
def default_fmt(col, row):
    return r"\num{%.5g}" %row[col]
    
def two_sigma_fmt(col, row):
    return r"\num{%.5g}" %(row['median']+row[col])
  
   
#------------------------------------------------------------------------------
# main class/functions
#------------------------------------------------------------------------------ 
def to_comparison_table(data, filename=None, free_only=False, constrained_only=False, params=None):
    """
    Write out a comparison table from multiple BestfitParameterSet instances
    """
    if len(data) < 2:
        raise ValueError("data length in ``to_comparison_table`` must be greater than 1")
    names = data.keys()
    for i, name in enumerate(data):
        df = data[name]
        if free_only:
            df = df.loc[df.free == True]
        elif constrained_only:
            df = df.loc[df.free == False]
        if params is not None:
            for p in params:
                if p not in df.index:
                    raise ValueError("cannot trim by params; `%s` is not present" %p)
            df = df.loc[params]
        
        if i == 0:
            out = BestfitParameterSet(index=df.index)    
            out['tex_name'] = df['tex_name']
        out[name] = df.apply(df.latex_fmts["median $\pm$ $1\sigma$"], axis=1)
    
    out.loc["red_chi2"] = [0]*len(out.columns)
    out.loc["red_chi2", 'tex_name'] = r'$\chi^2$/d.o.f.'
    for k in names:
        red_chi2 = 2*data[k].min_minus_lkl/(data[k].Nb-data[k].Np)
        args = (2*data[k].min_minus_lkl, data[k].Nb, data[k].Np, red_chi2)
        out.loc['red_chi2', k] = "$%d/(%d - %d) = %.2f$" %args
        
    index = out.index.values
    out = out.reindex([index[-1]] + list(index[:-1]))
    return out.to_latex(filename=filename, columns=names)


class BestfitParameterSet(pd.DataFrame):
    """
    A class to store and manipulate a set of bestfit parameters
    """
    latex_cols = ["best-fit", "median $\pm$ $1\sigma$", "95\% lower", "95\% upper"]
    latex_fmts = {r"median $\pm$ $1\sigma$" : median_plus_error_fmt, 
                    "best-fit" : lambda row: default_fmt('best_fit', row),
                     "95\% lower" : lambda row: two_sigma_fmt('lower_2sigma', row),
                     "95\% upper" : lambda row: two_sigma_fmt('upper_2sigma', row)}

    def __init__(self, *args, **kwargs):
        super(BestfitParameterSet, self).__init__(*args, **kwargs)
        self.__dict__['_user_metadata'] = []
    
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
        for k, v in meta.iteritems():
            self._user_metadata.append(k)
            setattr(self, k, v)
        
    def to_info(self, filename):
        """
        Write out the parameter set as a plaintext ``info`` file
        """
        if len(self._user_metadata):
            with open(filename, 'w') as ff:
                for k in self._user_metadata:
                    val = getattr(self, k)
                    ff.write("# %s %s %s\n" %(k, val, type(val).__name__))
        self.to_csv(filename, sep=" ", mode='a')
     
    @classmethod   
    def from_info(cls, filename):
        """
        Write out the parameter set as a plaintext ``info`` file
        """
        lines = open(filename, 'r').readlines()
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
        df = pd.read_csv(filename, delim_whitespace=True, comment='#')
        toret = cls(df)
        if len(meta):
            toret.add_metadata(**meta)
        return toret
        
    def to_latex(self, filename=None, caption="", columns=None):
        """
        Write out the bestfit parameters as a latex table
        """
        have_meta = len(self._user_metadata)
        pd.set_option('max_colwidth', 1000)
        if columns is None: columns = self.latex_cols
        for k in columns:
            if k in self.latex_cols:
                self[k] = self.apply(self.latex_fmts[k], axis=1)
            elif k not in self:
                raise ValueError("cannot output column with name `%s`, as it is not present" %k)
        
        kwargs = {}
        kwargs['escape'] = False
        kwargs['index_names'] = False
        kwargs['columns'] = columns
        
        df = self.reset_index()
        df = df.set_index('tex_name')
        out = pd.DataFrame.to_latex(df, **kwargs)
        
        # do some formatting
        Ncols = len(columns)
        out = out.replace("".join(['l']*(Ncols+1)), 'c'+"".join(['l']*(Ncols)))
        out = out.replace(r"\midrule", r'\colrule')
        out = out.replace(r"\bottomrule", r"\botrule")
        
        # add a header
        header = "\\begin{table}\n\\centering\n"
        precision = "\\sisetup{round-mode=places, round-precision=3}\n"
        header += precision
        header += "\\caption{%s}\n\\label{tab:}\n" %caption
        if have_meta: header += "\\begin{threeparttable}\n"
        out = header + out
        
        # add a footer
        note = ""
        if have_meta:
            args = (self.min_minus_lkl, self.Np, self.Nb, 2*self.min_minus_lkl/(self.Nb-self.Np))
            note = r'$-\ln\mathcal{L} = %.2f$, $N_p = %d$, $N_b = %d$, $\chi^2_\mathrm{red} = %.2f$' %args
            note = "\\begin{tablenotes}\n\t\item %s\n\\end{tablenotes}\n" %note
            note += "\\end{threeparttable}\n"
        out = out + note + "\\end{table}"
        
        if filename is None:
            return out
        else:
            with open(filename, 'w') as ff:
                ff.write(out)
        
        
        
        
    