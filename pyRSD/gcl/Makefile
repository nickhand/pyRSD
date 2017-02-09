# Makefile for GCL
# Nick Hand 11/21/14

MDIR := $(shell pwd)
WRKDIR = $(MDIR)/build

.base:
	if ! [ -e $(WRKDIR) ]; then mkdir $(WRKDIR) ; mkdir $(WRKDIR)/lib; fi;
	touch build/.base

vpath %.cpp src test
vpath %.c src
ifdef COMPILE_EMU
    ifeq ($(COMPILE_EMU), 1) 
	    vpath %.c src
    endif
endif
vpath %.f src 
vpath %.o build
vpath .base build

include ../../setup.config

########################################################
###### IN PRINCIPLE THE REST SHOULD BE LEFT UNCHANGED ##
########################################################

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
INCLUDES = -I../include -I$(FFTW_INC) -I$(CLASS_INC)

# libraries
LIBS = -L$(CLASS_LIB) -L$(FFTW_LIBDIR) $(FFTW_LIBS)
    
# make the libraries and includes to pass to python bindings
PY_INCLUDES = $(FFTW_INC):$(CLASS_INC):$(MDIR)/src:$(MDIR)/include
ifdef GSL_INC
    PY_INCLUDES := $(PY_INCLUDES):$(GSL_INC)
endif
PY_LIBDIRS = $(FFTW_LIBDIR)
PY_LIBS = -lgfortran -lgcl -lgomp $(FFTW_LIBS)

# add emulator libraries
ifdef COMPILE_EMU
    ifeq ($(COMPILE_EMU), 1)
        # c flags
        INCLUDES += -I$(GSL_INC)
        LIBS += -L$(GSL_LIBDIR)
        MCK_CXXFLAGS += $(GSL_LIBS)
        
        # python flags
        PY_LIBS += $(GSL_LIBS)
        PY_LIBDIRS := $(PY_LIBDIRS):$(GSL_LIBDIR)
    endif
endif

# python library names
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
# FrankenEmu sources
#-------------------------------------------------------------------------------
EMU_SOURCES = 
ifdef COMPILE_EMU
    ifeq ($(COMPILE_EMU), 1)
        EMU_COBJS = emu.o emu_noh.o hubble.o
    endif
endif

#-------------------------------------------------------------------------------
# GCL sources
#-------------------------------------------------------------------------------
GCLTOOLS_COBJS = Spline.o SpecialFunctions.o pstring.o Datafile.o parray.o \
				 Quadrature.o Common.o Timer.o MyFFTLog.o DiscreteQuad.o \
				 FortranFFTLog.o

GCLSOURCE_COBJS = Engine.o ClassParams.o ClassCosmology.o Cosmology.o \
				  PowerSpectrum.o  LinearPS.o Imn.o Jmn.o Kmn.o OneLoopPS.o \
				  ImnOneLoop.o	CorrelationFunction.o Kaiser.o ZeldovichPS.o \
				  NonlinearPS.o ZeldovichCF.o

GCLTEST_COBJS = test_linearPS.o test_Imn.o test_Jmn.o test_Kmn.o \
				test_oneloopPS.o test_ImnOneLoop.o test_zeldovichPS.o

COBJS = $(EMU_COBJS) $(GCLTOOLS_COBJS) $(GCLSOURCE_COBJS)

 #-------------------------------------------------------------------------------
 # FFTLog sources
 #-------------------------------------------------------------------------------
FOBJS = fftlog.o cdgamma.o drfftb.o drffti.o drfftf.o

#-------------------------------------------------------------------------------
# Libraries
#-------------------------------------------------------------------------------
LIBRARY = libgcl.a
PY_LIBRARY = gcl

#-------------------------------------------------------------------------------
# Rules
#-------------------------------------------------------------------------------
.PHONY: check_env all test $(PY_LIBRARY)

.SUFFIXES: .o .c .cpp .f 
    
.c.o:   %.c .base
	@echo Compiling $<
	cd $(WRKDIR); $(MCK_CC) $(MCK_OPT) $(OMPFLAG) $(MCK_CXXFLAGS) $(INCLUDES) $(LIBS) -c ../$< -o $*.o
    	
.cpp.o:	 %.cpp .base
	@echo Compiling $<
	cd $(WRKDIR); $(MCK_CXX) $(MCK_OPT) $(OMPFLAG) $(MCK_CXXFLAGS) $(INCLUDES) $(LIBS) -c ../$< -o $*.o

.f.o:  %.f 
	@echo Compiling $<
	cd $(WRKDIR); $(MCK_FC) $(MCK_FFLAGS) -c ../$< -o $*.o

all: .base check_env $(LIBRARY)
    
check_env:    
    ifdef COMPILE_EMU
        ifeq ($(COMPILE_EMU), 1) 
            ifndef GSL_PATH
                $(error GSL path must be specified to compile FrankenEmu)
            endif
        endif
    endif

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

$(PY_LIBRARY): .base check_env $(LIBRARY) python/*.i
	cd python; env CC=$(MCK_CXX) CXX=$(MCK_CXX) $(PYTHON) setup.py build_ext --inplace -I$(PY_INCLUDES) -l$(PY_LIBNAMES) -L$(PY_LIBDIRS)

clean: .base
	rm -rf $(WRKDIR)
	rm -f $(LIBRARY)
	rm -rf $(MDIR)/python/build
	rm -f $(addprefix test/, $(basename $(TEST)))