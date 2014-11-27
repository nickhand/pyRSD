%module pygcl

%{
#define SWIG_FILE_WITH_INIT
%}

%naturalvar;

%include "parray.i"
%include "numpy.i"
%include "typemaps.i"
%include "std_string.i"
using std::string;

%init %{
    import_array();
%}


%include "Common.i"
%include "Spline.i"

/*%include "Datafile.i"
%include "Quadrature.i"
%include "Timer.i"*/

%include "Engine.i"
%include "ClassParams.i"
%include "ClassCosmology.i"
%include "Cosmology.i"
%include "PowerSpectrum.i"
%include "LinearPS.i"
%include "OneLoopPS.i"
%include "Imn.i"
%include "Jmn.i"
%include "Kmn.i"
%include "ImnOneLoop.i"


