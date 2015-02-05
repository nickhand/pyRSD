from ... import os, data_dir
import ast 
import sys

#-------------------------------------------------------------------------------
def find_file(filename):
    """
    Return the filename, searching in the `pyRSD.data_dir` as well
    """
    # try to find the file 
    if os.path.exists(filename):
        return filename
    elif (os.path.exists("%s/%s" %(data_dir, filename))):
        return "%s/%s" %(data_dir, filename)
    else:
        raise ValueError("Could not find file '%s', tried ./%s and %s/%s" \
                         %(filename, filename, data_dir, filename))
                         
#-------------------------------------------------------------------------------
def is_floatable(val):
    if isinstance(val, (basestring, bool)):
        return False
    try:
        float(val)
        return True
    except:
        return False
        
#-------------------------------------------------------------------------------
def constraining_function(param_set, constraint, keys, modules, props='value'):
    """
    A constraining function to return the value of a `Parameter`, which 
    depends on other `Parameters` in a `ParameterSet`
    """
    if not isinstance(props, list):
        props = [props]*len(keys)
        
    formatted_constraint = constraint.format(**{k:param_set[k][att] for k, att in zip(keys, props)})
    return eval(formatted_constraint, globals().update(modules))
        
#-------------------------------------------------------------------------------
def text_between_chars(s, start_char="{", stop_char="}"):
    """
    Find the text between two characters, probably curly braces or 
    parentheses. 
    """
    toret = []
    start = 0
    while True:
        start = s.find( '{' , start, len(s))
        end = s.find( '}', start, len(s))
        if start != -1 and end != -1:
            toret.append(s[start+1:end])
            start = end
        else:
            break
    return toret

#-------------------------------------------------------------------------------
class ParseCall(ast.NodeVisitor):
    def __init__(self):
        self.ls = []
    def visit_Attribute(self, node):
        ast.NodeVisitor.generic_visit(self, node)
        self.ls.append(node.attr)
    def visit_Name(self, node):
        self.ls.append(node.id)


class FindFuncs(ast.NodeVisitor):
    def __init__(self):
        self.names = []
    def visit_Call(self, node):
        p = ParseCall()
        p.visit(node.func)
        self.names.append(".".join(p.ls))
        ast.NodeVisitor.generic_visit(self, node)
        
#-------------------------------------------------------------------------------
def stringToFunction(astr):
    """
    Given a string containing the name of a function, convert it to a function
    
    Parameters
    ----------
    astr : str 
        the string to convert to a function name
    """
    module, _, function = astr.rpartition('.')
    if module:
        __import__(module)
        mod = sys.modules[module]
        return getattr(mod, function)
    else:
        try:
            mod = sys.modules['__main__']
            return getattr(mod, function)
        except:
            mod = sys.modules['__builtin__']
            return getattr(mod, function)

#end stringToFunction

#-------------------------------------------------------------------------------
def replace_vars(s, D):
    """
    Given a string s and a dictionary of variables D, replace all variable
    names with the value from the dict D

    variable names in s are denoted by '$(' at the beginning and ')' at
    the end, or '$' at the beginning and a non-variable character at the
    end.  Variables must be valid python variable names, that is they
    consist of only alphanumeric characters (A-Z,a-z,0-9) and underscores,
    and cannot start with a number.

    example:
    >> D = {'my_var1' : 'abc',
            'my_var2' : '123' }
    >> s = "I know my $(my_var1)s and $my_var2's"
    >> print replace_vars(s,D)

    I know my abcs and 123's
    """
    s_in = str(s) 
    s_out = ''

    while True:
        i = s_in.find('$')
        if i==-1:
            s_out += s_in
            break

        s_out += s_in[:i]
        s_in = s_in[i+1:]

        if len(s_in)==0:
            raise ValueError, "trailing $"

        elif s_in[0] == '(':
            i = s_in.find(')')
            if i==-1:
                raise ValueError, "unmatched '('"
            var = s_in[1:i]

            s_in = s_in[i+1:]
            try:
                s_out += str(D[var])
            except:
                s_out += os.environ[var]

        else:
            var = ''
            i = 0
            while True:
                if i>=len(s_in):
                    break
                s = s_in[i]
                if (s >= 'a' and s <= 'z') \
                        or (s >= 'A' and s <= 'Z') \
                        or (s >= '0' and s <= '9') \
                        or s=='_':
                    var += s
                    i += 1
                else:
                    break
            s_in = s_in[i:]
            s_out += str(D[var])
    return s_out
    
#end replace_vars

#-------------------------------------------------------------------------------
def import_function_modules(line):
    """
    Find any function calls in the string specified by `line`, import the
    necessary modules, and return a dict of the imported modules
    """
    modules = {}
    
    # find the functions
    tree = ast.parse(line)
    function_finder = FindFuncs()
    function_finder.visit(tree)

    # now import
    for function in function_finder.names:
        mod, _, function = function.rpartition('.')
        mod = __import__(mod)
        modules[mod.__name__] = mod
        
    return modules

#end import_function_modules

#-------------------------------------------------------------------------------
    