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
OPTFLAG = -O4 -ffast-math -march=native
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
#-I/opt/local/include/

# libraries
LIBS = 

#-L/opt/local/lib/ -lgsl

# automatically add external programs if needed. First, initialize to blank.
EXTERNAL =

# eventually update flags for including HyRec
vpath %.o $(CLASS)/build
INCLUDES += -I$(CLASS)/include
EXTERNAL += arrays.o background.o common.o dei_rkck.o evolver_ndf15.o evolver_rkck.o growTable.o helium.o history.o hydrogen.o hyperspherical.o hyrectools.o input.o lensing.o nonlinear.o output.o parser.o perturbations.o primordial.o quadrature.o sparse.o spectra.o thermodynamics.o transfer.o 


%.o:  %.cpp .base
	cd $(WRKDIR);$(CC) $(OPTFLAG) $(OMPFLAG) $(CCFLAG) $(INCLUDES) $(LIBS) -c ../$< -o $*.o

TOOLS = Spline.o SpecialFunctions.o pstring.o Datafile.o parray.o Quadrature.o Common.o Timer.o

SOURCE = Engine.o ClassParams.o ClassCosmology.o Cosmology.o PowerSpectrum.o  LinearPS.o Imn.o Jmn.o Kmn.o OneLoopPS.o ImnOneLoop.o

TEST =  test_linearPS.o test_Imn.o test_Jmn.o test_Kmn.o test_oneloopPS.o test_ImnOneLoop.o

C_TOOLS =  $(addprefix src/, $(addsuffix .cpp,$(basename $(TOOLS))))
C_SOURCE = $(addprefix src/, $(addsuffix .cpp,$(basename $(SOURCE))))
C_TEST = $(addprefix test/, $(addsuffix .cpp,$(basename $(TEST) )))


all: libgcl.so

libgcl.so: $(TOOLS) $(SOURCE) $(EXTERNAL)
	$(SO) $(OPTFLAG) $(OMPFLAG) $(LIBS) $(addprefix build/, $(TOOLS) $(SOURCE)) $(addprefix $(CLASSLIB), $(EXTERNAL)) -o $@ 

test_linearPS: $(TOOLS) $(SOURCE) $(EXTERNAL) $(TEST)
	$(CC) $(OPTFLAG) $(OMPFLAG) $(LDFLAG) $(LIBS) -o test/$@ $(addprefix build/, $(addsuffix .o, $@)) $(addprefix build/, $(TOOLS) $(SOURCE)) $(addprefix $(CLASSLIB), $(EXTERNAL)) -lm

test_Imn: $(TOOLS) $(SOURCE) $(EXTERNAL) $(TEST)
	$(CC) $(OPTFLAG) $(OMPFLAG) $(LDFLAG)  $(LIBS) -o test/$@ $(addprefix build/, $(addsuffix .o, $@)) $(addprefix build/, $(TOOLS) $(SOURCE)) $(addprefix $(CLASSLIB), $(EXTERNAL)) -lm

test_Jmn: $(TOOLS) $(SOURCE) $(EXTERNAL) $(TEST)
	$(CC) $(OPTFLAG) $(OMPFLAG) $(LDFLAG)  $(LIBS) -o test/$@ $(addprefix build/, $(addsuffix .o, $@)) $(addprefix build/, $(TOOLS) $(SOURCE)) $(addprefix $(CLASSLIB), $(EXTERNAL)) -lm

test_Kmn: $(TOOLS) $(SOURCE) $(EXTERNAL) $(TEST)
	$(CC) $(OPTFLAG) $(OMPFLAG) $(LDFLAG)  $(LIBS) -o test/$@ $(addprefix build/, $(addsuffix .o, $@)) $(addprefix build/, $(TOOLS) $(SOURCE)) $(addprefix $(CLASSLIB), $(EXTERNAL)) -lm

test_oneloopPS: $(TOOLS) $(SOURCE) $(EXTERNAL) $(TEST)
	$(CC) $(OPTFLAG) $(OMPFLAG) $(LDFLAG)  $(LIBS) -o test/$@ $(addprefix build/, $(addsuffix .o, $@)) $(addprefix build/, $(TOOLS) $(SOURCE)) $(addprefix $(CLASSLIB), $(EXTERNAL)) -lm

test_ImnOneLoop: $(TOOLS) $(SOURCE) $(EXTERNAL) $(TEST)
	$(CC) $(OPTFLAG) $(OMPFLAG) $(LDFLAG)  $(LIBS) -o test/$@ $(addprefix build/, $(addsuffix .o, $@)) $(addprefix build/, $(TOOLS) $(SOURCE)) $(addprefix $(CLASSLIB), $(EXTERNAL)) -lm

pygcl: libgcl.so python/*.i	
	cd python; $(PYTHON) setup.py build_ext --inplace -I $(CLASS)/include

clean: .base
	rm -rf $(WRKDIR);
	rm -f libgcl.so
	# rm -f $(MDIR)/python/classy.c
	# rm -rf $(MDIR)/python/build
