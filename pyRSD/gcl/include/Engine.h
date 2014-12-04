//--------------------------------------------------------------------------
//
// Description:
// 	class Engine :
//base class for Boltzmann code
//
//
// Author List:
//	Stephane Plaszczynski (plaszczy@lal.in2p3.fr)
//
// History (add to end):
//	creation:   Tue Mar 13 15:28:50 CET 2012 
//
//------------------------------------------------------------------------

#ifndef ENGINE_H
#define ENGINE_H

#include <vector>
#include <ostream>
#include "parray.h"

class Engine {

public:

  enum cltype {TT=0,EE,TE,BB,PP,TP,EP}; //P stands for phi (lensing potential)
  enum pktype {Pklin=0, Pknl, Tk};
  
  //constructors
  Engine();
  
  // destructor
  virtual ~Engine();

  // units = (micro-K)^2
  virtual int GetCls(const std::vector<unsigned>&, parray&, parray&, parray&, parray&);
  virtual int GetLensing(const std::vector<unsigned>&, parray&, parray&, parray&);
  virtual int GetPklin(double, const parray&, parray&);
  virtual int GetPknl(double, const parray&, parray&);
  virtual int GetTk(double, const parray&, parray&);

  // the output functions
  virtual void WriteCls(std::ostream &o);
  virtual void WritePklin(std::ostream &of, double z);
  virtual void WritePknl(std::ostream &of, double z);
  virtual void WriteTk(std::ostream &of, double z);
  
  int lmax() {return _lmax;}

protected:
  int _lmax;
  
  virtual void WriteMatterSpectrum(std::ostream &of, double z, pktype t);

};

#endif

