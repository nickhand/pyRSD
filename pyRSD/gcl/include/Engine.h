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

class Engine
{

public:

  enum cltype {TT=0,EE,TE,BB,PP,TP,EP}; //P stands for phi (lensing potential)
  enum pktype {Pklin=0, Pknl, Tk};
  
  //constructors
  Engine();
  
  // destructor
  virtual ~Engine(){};

  // units = (micro-K)^2
  virtual int GetCls(const std::vector<unsigned>& lVec, //input 
		             parray& cltt, 
		             parray& clte, 
		             parray& clee, 
		             parray& clbb)=0;
  
                 
  virtual int GetLensing(const std::vector<unsigned>& lVec, //input 
			             parray& clpp, 
			             parray& cltp, 
			             parray& clep)=0;
			  
  virtual int GetPklin(double z, const parray& k, parray& Pk)=0;
  virtual int GetPknl(double z, const parray& k, parray& Pk)=0;
  virtual int GetTk(double z, const parray& k, parray& Tk)=0;

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

