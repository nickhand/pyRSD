import sys; sys.path.pop(0)
import matplotlib as mpl
mpl.use("Agg")

from runtests import Tester
import os.path

tester = Tester(os.path.abspath(__file__), "pyRSD")
tester.main(sys.argv[1:])
