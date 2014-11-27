%{
#include "Engine.h"
%}

class Engine {
public:
  enum cltype {TT,EE,TE,BB,PP,TP,EP};
  enum pktype {Pklin, Pknl, Tk};
  
  Engine();
  virtual ~Engine();

  int GetCls(const std::vector<unsigned>& lVec, parray& cltt, parray& clte, parray& clee, parray& clbb);          
  int GetLensing(const std::vector<unsigned>& lVec, parray& clpp, parray& cltp, parray& clep);
			  
  int GetPklin(double z, const parray& k, parray& Pk);
  int GetPknl(double z, const parray& k, parray& Pk);
  int GetTk(double z, const parray& k, parray& Tk);

  // the output functions
  void WriteCls(std::ostream &o);
  void WritePklin(std::ostream &of, double z);
  void WritePknl(std::ostream &of, double z);
  void WriteTk(std::ostream &of, double z);
  
  int lmax();
  
};