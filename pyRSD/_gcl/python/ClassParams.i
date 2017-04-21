%{
#include "Cosmology.h"
%}

class ClassParams {
public:    
    typedef map<std::string, std::string> param_vector;
    typedef param_vector::iterator iterator;
    typedef param_vector::const_iterator const_iterator;
    
  
    ClassParams(const std::string& param_file);

    int Update(const std::string& key, const int& val); 
    int Update(const std::string& key, const float& val); 
    int Update(const std::string& key, const double& val); 
    int Update(const std::string& key, const bool& val); 
    int Update(const std::string& key, const std::string& val);
    int Update(const std::string& key, const char* val);
  
    void Print();


    unsigned size() const;
    const std::string& value(const std::string& key) const;
    
    %extend {
        std::string __getitem__(const std::string& key) const { return $self->value(key); }
    };
  
};
