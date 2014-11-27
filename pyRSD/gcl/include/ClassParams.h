#ifndef CLASS_PARAMS_H
#define CLASS_PARAMS_H

// CLASS
#include "class.h"

#include <string>
#include <utility>
#include <stdexcept>
#include <map>
#include <iostream>

/*----------------------------------------------------------------------------*/
/* class to encapsulate CLASS parameters from any type (numerical or string)  */
/*----------------------------------------------------------------------------*/
class ClassParams {

public:
    
    typedef std::map<std::string, std::string> param_vector;
    typedef param_vector::iterator iterator;
    typedef param_vector::const_iterator const_iterator;
    

    ClassParams(const std::string& param_file); 

    // use this to add a CLASS variable
    int Update(const std::string& key, const int& val); 
    int Update(const std::string& key, const float& val); 
    int Update(const std::string& key, const double& val); 
    int Update(const std::string& key, const bool& val); 
    int Update(const std::string& key, const std::string& val);
    int Update(const std::string& key, const char* val); 

    void Print(); 
  
    // accesors
    inline unsigned size() const {return pars.size();}
    inline const std::string& value(const std::string& key) const {return pars.at(key);}
  
    // iterate over the pars variable
    const_iterator begin() const { return pars.begin(); }
    const_iterator end() const { return pars.end(); }
  
    // overload the [] operator to return const reference
    const std::string& operator[](const std::string& key) const { return this->value(key); }

private:
    param_vector pars;
};

#endif