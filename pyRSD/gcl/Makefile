# Makefile for GCL
# Nick Hand 11/21/14

MDIR := $(shell pwd)
WRKDIR = $(MDIR)/build
DATADIR := $(shell cd ../data; pwd )

.base:
	if ! [ -e $(WRKDIR) ]; then mkdir $(WRKDIR) ; mkdir $(WRKDIR)/lib; fi;
	touch build/.base

vpath %.cpp src test
vpath %.f src 
vpath %.o build
vpath .base build

include ../../setup.config

########################################################
###### IN PRINCIPLE THE REST SHOULD BE LEFT UNCHANGED ##
########################################################

# pass current working directory to the code
MCK_CXXFLAGS += -DGCLDIR='"$(MDIR)"'
MCK_CXXFLAGS += -DDATADIR='"$(DATADIR)"'

# optimization flag
MCK_OPT = $(MCK_OPT_WARN) $(MCK_OPT_HIGH)
ifdef OMP_FLAG
	MCK_CXXFLAGS += $(OMP_FLAG)
endif

# turn on debugging
ifdef DEBUG
	ifeq ($(DEBUG), 1) 
		MCK_CXXFLAGS += -DDEBUG
	endif
endif

# turn on verbosity
# turn on debugging
ifdef VERBOSE
	ifeq ($(VERBOSE), 1) 
		MCK_CXXFLAGS += -DVERBOSE
	endif
endif

# add standard fortran libraries
MCK_CXXFLAGS += -lgfortran

# where to find include files *.h
INCLUDES = -I../include	 -I$(FFTW_INC) -I$(CLASS_INC)

# libraries
LIBS = -L$(CLASS_LIB) 
    
# make the libraries and includes to pass to python bindings
PY_INCLUDES = "$(CLASS_INC):$(MDIR)/src:$(MDIR)/include"
PY_LIBS = -lgfortran -lgcl -lgomp
PY_LIBNAMES = $(subst -l,, "$(PY_LIBS)")

#-------------------------------------------------------------------------------
# CLASS sources
#-------------------------------------------------------------------------------
CLASS_SOURCES = arrays.o background.o common.o dei_rkck.o evolver_ndf15.o \
				evolver_rkck.o growTable.o helium.o history.o hydrogen.o \
				hyperspherical.o hyrectools.o input.o lensing.o nonlinear.o \
				output.o parser.o perturbations.o primordial.o quadrature.o \
				sparse.o spectra.o thermodynamics.o transfer.o 
CLASS_COBJS = $(addprefix $(CLASS_PATH)/build/, $(CLASS_SOURCES))

#-------------------------------------------------------------------------------
# GCL sources
#-------------------------------------------------------------------------------
GCLTOOLS_COBJS = Spline.o SpecialFunctions.o pstring.o Datafile.o parray.o \
				 Quadrature.o Common.o Timer.o FFTLogWrapper.o

GCLSOURCE_COBJS = Engine.o ClassParams.o ClassCosmology.o Cosmology.o \
				  PowerSpectrum.o  LinearPS.o Imn.o Jmn.o Kmn.o OneLoopPS.o \
				  ImnOneLoop.o	CorrelationFunction.o Kaiser.o ZeldovichPS.o

GCLTEST_COBJS = test_linearPS.o test_Imn.o test_Jmn.o test_Kmn.o \
				test_oneloopPS.o test_ImnOneLoop.o test_zeldovichPS.o

COBJS = $(GCLTOOLS_COBJS) $(GCLSOURCE_COBJS)

#-------------------------------------------------------------------------------
# FFTLog sources
#-------------------------------------------------------------------------------
FOBJS = fftlog.o cdgamma.o drfftb.o drffti.o drfftf.o

#-------------------------------------------------------------------------------
# Libraries
#-------------------------------------------------------------------------------
LIBRARY = libgcl.a
PY_LIBRARY = pygcl

#-------------------------------------------------------------------------------
# Rules
#-------------------------------------------------------------------------------
.PHONY: all test $(PY_LIBRARY)

.SUFFIXES: .o .cpp .f
    
.cpp.o:	 %.cpp .base
	@echo Compiling $<
	cd $(WRKDIR); $(MCK_CXX) $(MCK_OPT) $(OMPFLAG) $(MCK_CXXFLAGS) $(INCLUDES) $(LIBS) -c ../$< -o $*.o

.f.o:  %.f 
	@echo Compiling $<
	cd $(WRKDIR); $(MCK_FC) $(MCK_FFLAGS) -c ../$< -o $*.o

all: .base $(LIBRARY)

$(LIBRARY): $(COBJS) $(FOBJS)
	$(AR_EXEC) $(AR_FLAGS) $@  $(addprefix build/, $(COBJS) $(FOBJS)) $(CLASS_COBJS)

test : .base $(basename $(GCLTEST_COBJS))

test_linearPS : test_linearPS.o $(COBJS) $(FOBJS)
	$(MCK_CXX) $(MCK_OPT) $(OMPFLAG) $(MCK_CXXFLAGS) $(INCLUDES) $(LIBS) -o test/$@ $(addprefix build/, $(addsuffix .o, $@)) $(addprefix build/, $(COBJS) $(FOBJS)) $(CLASS_COBJS)
    
test_Imn : test_Imn.o $(COBJS) $(FOBJS)
	$(MCK_CXX) $(MCK_OPT) $(OMPFLAG) $(MCK_CXXFLAGS) $(INCLUDES) $(LIBS) -o test/$@ $(addprefix build/, $(addsuffix .o, $@)) $(addprefix build/, $(COBJS) $(FOBJS)) $(CLASS_COBJS)
    
test_Jmn : test_Jmn.o $(COBJS) $(FOBJS)
	$(MCK_CXX) $(MCK_OPT) $(OMPFLAG) $(MCK_CXXFLAGS) $(INCLUDES) $(LIBS) -o test/$@ $(addprefix build/, $(addsuffix .o, $@)) $(addprefix build/, $(COBJS) $(FOBJS)) $(CLASS_COBJS)
    
test_Kmn : test_Kmn.o $(COBJS) $(FOBJS)
	$(MCK_CXX) $(MCK_OPT) $(OMPFLAG) $(MCK_CXXFLAGS) $(INCLUDES) $(LIBS) -o test/$@ $(addprefix build/, $(addsuffix .o, $@)) $(addprefix build/, $(COBJS) $(FOBJS)) $(CLASS_COBJS)
    
test_oneloopPS : test_oneloopPS.o $(COBJS) $(FOBJS)
	$(MCK_CXX) $(MCK_OPT) $(OMPFLAG) $(MCK_CXXFLAGS) $(INCLUDES) $(LIBS) -o test/$@ $(addprefix build/, $(addsuffix .o, $@)) $(addprefix build/, $(COBJS) $(FOBJS)) $(CLASS_COBJS)
    
test_ImnOneLoop : test_ImnOneLoop.o $(COBJS) $(FOBJS)
	$(MCK_CXX) $(MCK_OPT) $(OMPFLAG) $(MCK_CXXFLAGS) $(INCLUDES) $(LIBS) -o test/$@ $(addprefix build/, $(addsuffix .o, $@)) $(addprefix build/, $(COBJS) $(FOBJS)) $(CLASS_COBJS)
    
test_zeldovichPS : test_zeldovichPS.o $(COBJS) $(FOBJS)
	$(MCK_CXX) $(MCK_OPT) $(OMPFLAG) $(MCK_CXXFLAGS) $(INCLUDES) $(LIBS) -o test/$@ $(addprefix build/, $(addsuffix .o, $@)) $(addprefix build/, $(COBJS) $(FOBJS)) $(CLASS_COBJS)

$(PY_LIBRARY): .base $(LIBRARY) python/*.i
	cd python; $(PYTHON) setup.py build_ext --inplace -I$(PY_INCLUDES) -l$(PY_LIBNAMES)

clean: .base
	rm -rf $(WRKDIR)
	rm -f $(LIBRARY)
	rm -rf $(MDIR)/python/build
	rm -f $(addprefix test/, $(basename $(TEST)))
