%{
#include "Engine.h"
%}

class Engine {
public:
  enum cltype {TT,EE,TE,BB,PP,TP,EP};

  Engine();
  virtual ~Engine();

  int GetCls(const std::vector<unsigned>& lVec, parray& cltt, parray& clte, parray& clee, parray& clbb);          
  int GetLensing(const std::vector<unsigned>& lVec, parray& clpp, parray& cltp, parray& clep);
			  
  double GetPklin(double z, double);
  double GetPknl(double z, double k);
  int GetTk(double z, const parray& k, parray& Tk);

  // the output functions
  void WriteCls(std::ostream &o);
  void WriteTk(std::ostream &of, double z);
  
  int lmax();
  
};