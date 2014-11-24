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

########################################################
######### LINES TO ADAPT TO YOUR PLATFORM ##############
########################################################

# your C++ compiler:
CC		 = g++ -Wall -std=c++11


# your tool for creating shared object libraries:
SO		 = g++ -shared

# (OPT) your python interpreter
PYTHON = python

# your optimization flag
OPTFLAG = -O4 -ffast-math #-march=native
#OPTFLAG = -Ofast -ffast-math #-march=native
#OPTFLAG = -fast

# your openmp flag (comment for compiling without openmp)
OMPFLAG	  = -fopenmp
#OMPFLAG   = -mp -mp=nonuma -mp=allcores -g
#OMPFLAG   = -openmp

# all other compilation flags
CFLAG = -g -fPIC
LDFLAG = -g -fPIC 

# the ClASS directory
CLASS = /Users/Nick/Research/Programs/class
CLASSLIB = $(CLASS)/build/

########################################################
###### IN PRINCIPLE THE REST SHOULD BE LEFT UNCHANGED ##
########################################################

# pass current working directory to the code
CCFLAG += -DGCLDIR='"$(MDIR)"'
CCFLAG += -DDATADIR='"$(DATADIR)"'

# turn on debugging
CCFLAG += -DDEBUG

# turn on verbosity
#CCFLAG += -DVERBOSE

# where to find include files *.h
INCLUDES = -I../include

# automatically add external programs if needed. First, initialize to blank.
EXTERNAL =

# eventually update flags for including HyRec
vpath %.o $(CLASS)/build
INCLUDES += -I$(CLASS)/include
EXTERNAL += arrays.o background.o common.o dei_rkck.o evolver_ndf15.o evolver_rkck.o growTable.o helium.o history.o hydrogen.o hyperspherical.o hyrectools.o input.o lensing.o nonlinear.o output.o parser.o perturbations.o primordial.o quadrature.o sparse.o spectra.o thermodynamics.o transfer.o 

#endif

%.o:  %.cpp .base
	cd $(WRKDIR);$(CC) $(OPTFLAG) $(OMPFLAG) $(CCFLAG) $(INCLUDES) -c ../$< -o $*.o

TOOLS = Spline.o SpecialFunctions.o pstring.o Datafile.o parray.o Quadrature.o Common.o Timer.o

SOURCE = Engine.o ClassCosmology.o Cosmology.o PowerSpectrum.o NoWigglePS.o LinearPS.o Imn.o

TEST =  test_linearPS.o test_Imn.o

#LinearPS.o 


C_TOOLS =  $(addprefix src/, $(addsuffix .cpp,$(basename $(TOOLS))))
C_SOURCE = $(addprefix src/, $(addsuffix .cpp,$(basename $(SOURCE))))
C_TEST = $(addprefix test/, $(addsuffix .cpp,$(basename $(TEST) )))
# C_MAIN = $(addprefix main/, $(addsuffix .c,$(basename $(CLASS))))
# C_ALL = $(C_MAIN) $(C_TOOLS) $(C_SOURCE)
# H_ALL = $(addprefix include/, common.h svnversion.h $(addsuffix .h, $(basename $(notdir $(C_ALL)))))
# PRE_ALL = cl_ref.pre clt_permille.pre
# INI_ALL = explanatory.ini lcdm.ini
# MISC_FILES = Makefile CPU psd_FD_single.dat myselection.dat myevolution.dat README bbn/sBBN.dat external_Pk/* cpp
# PYTHON_FILES = python/classy.pyx python/setup.py python/cclassy.pxd python/test_class.py




all: libgcl.so

libgcl.so: $(TOOLS) $(SOURCE) $(EXTERNAL)
	$(SO) $(OPTFLAG) $(OMPFLAG) $(addprefix build/, $(TOOLS) $(SOURCE)) $(addprefix $(CLASSLIB), $(EXTERNAL)) -o $@ 

test_linearPS: $(TOOLS) $(SOURCE) $(EXTERNAL) $(TEST)
	$(CC) $(OPTFLAG) $(OMPFLAG) $(LDFLAG) -o test/$@ $(addprefix build/, $(addsuffix .o, $@)) $(addprefix build/, $(TOOLS) $(SOURCE)) $(addprefix $(CLASSLIB), $(EXTERNAL)) -lm

test_Imn: $(TOOLS) $(SOURCE) $(EXTERNAL) $(TEST)
	$(CC) $(OPTFLAG) $(OMPFLAG) $(LDFLAG) -o test/$@ $(addprefix build/, $(addsuffix .o, $@)) $(addprefix build/, $(TOOLS) $(SOURCE)) $(addprefix $(CLASSLIB), $(EXTERNAL)) -lm

# 
# test_loops: $(TOOLS) $(SOURCE) $(EXTERNAL) $(OUTPUT) $(TEST_LOOPS)
#	$(CC) $(OPTFLAG) $(OMPFLAG) $(LDFLAG) -o $@ $(addprefix build/,$(notdir $^)) -lm
# 
# test_stephane: $(TOOLS) $(SOURCE) $(EXTERNAL) $(OUTPUT) $(TEST_STEPHANE)
#	$(CC) $(OPTFLAG) $(OMPFLAG) $(LDFLAG) -o $@ $(addprefix build/,$(notdir $^)) -lm
# 
# test_degeneracy: $(TOOLS) $(SOURCE) $(EXTERNAL) $(OUTPUT) $(TEST_DEGENERACY)
#	$(CC) $(OPTFLAG) $(OMPFLAG) $(LDFLAG) -o $@ $(addprefix build/,$(notdir $^)) -lm
# 
# test_transfer: $(TOOLS) $(SOURCE) $(EXTERNAL) $(TEST_TRANSFER)
#	$(CC) $(OPTFLAG) $(OMPFLAG) $(LDFLAG) -o  $@ $(addprefix build/,$(notdir $^)) -lm
# 
# test_nonlinear: $(TOOLS) $(SOURCE) $(EXTERNAL) $(TEST_NONLINEAR)
#	$(CC) $(OPTFLAG) $(OMPFLAG) $(LDFLAG) -o  $@ $(addprefix build/,$(notdir $^)) -lm
# 
# test_perturbations: $(TOOLS) $(SOURCE) $(EXTERNAL) $(TEST_PERTURBATIONS)
#	$(CC) $(OPTFLAG) $(OMPFLAG) $(LDFLAG) -o  $@ $(addprefix build/,$(notdir $^)) -lm
# 
# test_thermodynamics: $(TOOLS) $(SOURCE) $(EXTERNAL) $(TEST_THERMODYNAMICS)
#	$(CC) $(OPTFLAG) $(OMPFLAG) $(LDFLAG) -o  $@ $(addprefix build/,$(notdir $^)) -lm
# 
# test_background: $(TOOLS) $(SOURCE) $(EXTERNAL) $(TEST_BACKGROUND)
#	$(CC) $(OPTFLAG) $(OMPFLAG) $(LDFLAG) -o  $@ $(addprefix build/,$(notdir $^)) -lm
# 
# test_hyperspherical: $(TOOLS) $(TEST_HYPERSPHERICAL)
#	$(CC) $(OPTFLAG) $(OMPFLAG) $(LDFLAG) -o test_hyperspherical $(addprefix build/,$(notdir $^)) -lm
# 
# 
# tar: $(C_ALL) $(C_TEST) $(H_ALL) $(PRE_ALL) $(INI_ALL) $(MISC_FILES) $(HYREC) $(PYTHON_FILES)
#	tar czvf class.tar.gz $(C_ALL) $(H_ALL) $(PRE_ALL) $(INI_ALL) $(MISC_FILES) $(HYREC) $(PYTHON_FILES)
# 
# classy: libclass.a python/classy.pyx python/cclassy.pxd
#	cd python; $(PYTHON) setup.py install --user

clean: .base
	rm -rf $(WRKDIR);
	rm -f libgcl.so
	# rm -f $(MDIR)/python/classy.c
	# rm -rf $(MDIR)/python/build
