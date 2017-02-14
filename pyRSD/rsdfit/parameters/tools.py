from ... import os, sys
import ast
from six import string_types

def get_abspath(value):
    if isinstance(value, string_types) and os.path.exists(value):
        return os.path.abspath(value)
    else:
        return value


def is_floatable(val):
    if isinstance(val, (bool,)+string_types):
        return False
    try:
        float(val)
        return True
    except:
        return False

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

def string_to_function(astr):
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
            raise ValueError("trailing $")

        elif s_in[0] == '(':
            i = s_in.find(')')
            if i==-1:
                raise ValueError("unmatched '('")
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
        if len(mod):
            mod = __import__(mod)
            modules[mod.__name__] = mod

    return modules
    
def verify_line(line, length, lineno):
    """
    Verify the line read makes sense
    """
    if (len(line) == 0 or line[0] == ''): 
        return False
    if len(line) == 1:
        raise ValueError("Error reading parameter %s on line %d" %(line[0], lineno))
    elif len(line) != length:
        raise ValueError("Cannot understand line %d" %lineno)
        
    return True
