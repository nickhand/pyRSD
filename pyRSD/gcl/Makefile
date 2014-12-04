# Makefile for GCL
# Nick Hand 11/21/14

MDIR := $(shell pwd)
WRKDIR = $(MDIR)/build
DATADIR := $(shell cd ../data; pwd )

.base:
	if ! [ -e $(WRKDIR) ]; then mkdir $(WRKDIR) ; mkdir $(WRKDIR)/lib; fi;
	touch build/.base

vpath %.cpp src test
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

# where to find include files *.h
INCLUDES = -I../include	 -I$(FFTW_INC) -I$(CLASS_INC)

# libraries
LIBS = -L$(CLASS_LIB) -L$(FFTW_LIBDIR) $(FFTW_LIBS)

# make the libraries and includes to pass to python bindings
PY_INCLUDES = "$(FFTW_INC):$(CLASS_INC):$(MDIR)/src:$(MDIR)/include"
PY_LIBDIRS = $(FFTW_LIBDIR)
PY_LIBS = gcl gomp
PY_LIBS += $(foreach lib, $(FFTW_LIBS), $(subst -l, , $(lib)))

# class sources
CLASS_SOURCES = arrays.o background.o common.o dei_rkck.o evolver_ndf15.o evolver_rkck.o growTable.o helium.o \
				history.o hydrogen.o hyperspherical.o hyrectools.o input.o lensing.o nonlinear.o output.o parser.o \
				perturbations.o primordial.o quadrature.o sparse.o spectra.o thermodynamics.o transfer.o 
CLASS_OBJS = $(addprefix $(CLASS_PATH)/build/, $(CLASS_SOURCES))

TOOLS = fftlog.o Spline.o SpecialFunctions.o pstring.o Datafile.o parray.o Quadrature.o Common.o Timer.o 

SOURCE = Engine.o ClassParams.o ClassCosmology.o Cosmology.o PowerSpectrum.o  LinearPS.o Imn.o Jmn.o Kmn.o OneLoopPS.o \
		 ImnOneLoop.o ZeldovichPS.o CorrelationFunction.o Kaiser.o

TEST =	test_linearPS.o test_Imn.o test_Jmn.o test_Kmn.o test_oneloopPS.o test_ImnOneLoop.o test_zeldovichPS.o

LIBRARY = libgcl.a

PY_LIBRARY = pygcl

%.o:  %.cpp .base
	@echo Compiling $<
	cd $(WRKDIR); $(MCK_CXX) $(MCK_OPT) $(OMPFLAG) $(MCK_CXXFLAGS) $(INCLUDES) $(LIBS) -c ../$< -o $*.o

all: $(LIBRARY)

$(LIBRARY): $(TOOLS) $(SOURCE) 
	$(AR_EXEC) $(AR_FLAGS) $@  $(addprefix build/, $(TOOLS) $(SOURCE)) $(CLASS_OBJS)

test : $(basename $(TEST))

test_linearPS : test_linearPS.o $(TOOLS) $(SOURCE) 
	$(MCK_CXX) $(MCK_OPT) $(OMPFLAG) $(MCK_CXXFLAGS) $(INCLUDES) $(LIBS) -o test/$@ $(addprefix build/, $(addsuffix .o, $@)) $(addprefix build/, $(TOOLS) $(SOURCE)) $(CLASS_OBJS) 

test_Imn : test_Imn.o $(SOURCE) 
	$(MCK_CXX) $(MCK_OPT) $(OMPFLAG) $(MCK_CXXFLAGS) $(INCLUDES) $(LIBS) -o test/$@ $(addprefix build/, $(addsuffix .o, $@)) $(addprefix build/, $(TOOLS) $(SOURCE)) $(CLASS_OBJS) 

test_Jmn : test_Jmn.o $(TOOLS) $(SOURCE) 
	$(MCK_CXX) $(MCK_OPT) $(OMPFLAG) $(MCK_CXXFLAGS) $(INCLUDES) $(LIBS) -o test/$@ $(addprefix build/, $(addsuffix .o, $@)) $(addprefix build/, $(TOOLS) $(SOURCE)) $(CLASS_OBJS) 

test_Kmn : test_Kmn.o $(TOOLS) $(SOURCE) 
	$(MCK_CXX) $(MCK_OPT) $(OMPFLAG) $(MCK_CXXFLAGS) $(INCLUDES) $(LIBS) -o test/$@ $(addprefix build/, $(addsuffix .o, $@)) $(addprefix build/, $(TOOLS) $(SOURCE)) $(CLASS_OBJS)

test_oneloopPS : test_oneloopPS.o $(TOOLS) $(SOURCE) 
	$(MCK_CXX) $(MCK_OPT) $(OMPFLAG) $(MCK_CXXFLAGS) $(INCLUDES) $(LIBS) -o test/$@ $(addprefix build/, $(addsuffix .o, $@)) $(addprefix build/, $(TOOLS) $(SOURCE)) $(CLASS_OBJS) 

test_ImnOneLoop : test_ImnOneLoop.o $(TOOLS) $(SOURCE) 
	$(MCK_CXX) $(MCK_OPT) $(OMPFLAG) $(MCK_CXXFLAGS) $(INCLUDES) $(LIBS) -o test/$@ $(addprefix build/, $(addsuffix .o, $@)) $(addprefix build/, $(TOOLS) $(SOURCE)) $(CLASS_OBJS)

test_zeldovichPS : test_zeldovichPS.o $(TOOLS) $(SOURCE) 
	$(MCK_CXX) $(MCK_OPT) $(OMPFLAG) $(MCK_CXXFLAGS) $(INCLUDES) $(LIBS) -o test/$@ $(addprefix build/, $(addsuffix .o, $@)) $(addprefix build/, $(TOOLS) $(SOURCE)) $(CLASS_OBJS) 

$(PY_LIBRARY): $(LIBRARY) python/*.i
	cd python; $(PYTHON) setup.py build_ext --inplace -I$(PY_INCLUDES) -l"$(PY_LIBS)" -L$(PY_LIBDIRS) 

clean: .base
	rm -rf $(WRKDIR)
	rm -f $(LIBRARY)
	rm -rf $(MDIR)/python/build
	rm -f $(addprefix test/, $(basename $(TEST)))
