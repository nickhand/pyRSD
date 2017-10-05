%module gcl

%{
#define SWIG_FILE_WITH_INIT
%}

%naturalvar;

%include "_gcl/python/numpy.i"

%init %{
    import_array();
%}

%include "typemaps.i"
%include "std_string.i"
%include "std_vector.i"
%include exception.i

using std::string;

%exception {
    try {
        $action
    } catch(const std::exception& e) {
        SWIG_exception(SWIG_UnknownError, e.what());
    } catch(...) {
        SWIG_exception(SWIG_RuntimeError, "unknown exception");
    }
}


%feature("kwargs");
%feature("autodoc");

%include "_gcl/python/parray.i"
%include "_gcl/python/Common.i"
%include "_gcl/python/Spline.i"
%include "_gcl/python/SpecialFunctions.i"

%include "_gcl/python/DiscreteQuad.i"
%include "_gcl/python/ClassEngine.i"
%include "_gcl/python/ClassParams.i"
%include "_gcl/python/Cosmology.i"
%include "_gcl/python/PowerSpectrum.i"
%include "_gcl/python/LinearPS.i"
%include "_gcl/python/OneLoopPS.i"
%include "_gcl/python/CorrelationFunction.i"
%include "_gcl/python/Kaiser.i"
%include "_gcl/python/Imn.i"
%include "_gcl/python/Jmn.i"
%include "_gcl/python/Kmn.i"
%include "_gcl/python/ImnOneLoop.i"
%include "_gcl/python/ZeldovichPS.i"
%include "_gcl/python/NonlinearPS.i"
%include "_gcl/python/ZeldovichCF.i"
