import sys; sys.path.pop(0)
from runtests import Tester
import os.path

tester = Tester(os.path.abspath(__file__), "pyRSD")
tester.main(sys.argv[1:])
