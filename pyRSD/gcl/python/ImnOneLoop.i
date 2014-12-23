%{
#include "ImnOneLoop.h"
%}

class ImnOneLoop {
public: 

    ImnOneLoop(const OneLoopPS& P_1, double epsrel = 1e-4);
    ImnOneLoop(const OneLoopPS& P_1, const OneLoopPS& P_2, double epsrel = 1e-4);

    double EvaluateLinear(double k, int m, int n) const;
    parray EvaluateLinear(const parray& k, int m, int n) const;

    double EvaluateCross(double k, int m, int n) const;
    parray EvaluateCross(const parray& k, int m, int n) const;

    double EvaluateOneLoop(double k, int m, int n) const;
    parray EvaluateOneLoop(const parray& k, int m, int n) const;
    
    const OneLoopPS& GetOneLoopPS1() const;
    const OneLoopPS& GetOneLoopPS2() const;
    const double& GetEpsrel() const;
    const bool GetEqual() const;
};

%extend ImnOneLoop {
%pythoncode {
    def __reduce__(self):
        if (self.GetEqual()):
            args = self.GetOneLoopPS1(), self.GetEpsrel()
        else:
            args = self.GetOneLoopPS1(), self.GetOneLoopPS2(), self.GetEpsrel()
        return self.__class__, args
}
}

