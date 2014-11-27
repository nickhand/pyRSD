#include "ClassParams.h" 

#include <sstream>
#include <numeric>
#include <iomanip>
 
using namespace std;


ClassParams::ClassParams(const string& param_file) {
      
      // for error messages
      ErrorMsg _errmsg; 
      
      // initialize an empty file_content to read the params file
      struct file_content fc_input;
      fc_input.size = 0;
      fc_input.filename=new char[1];
      
      // read the param file
      if (parser_read_file(const_cast<char*>(param_file.c_str()), &fc_input, _errmsg) == _FAILURE_){
          throw invalid_argument(_errmsg);
      }
      
      // set the pars
      for (int i=0; i < fc_input.size; i++) {
          this->Update(fc_input.name[i], fc_input.value[i]);
      }
      
      // free the input
      parser_free(&fc_input);
      
}

void ClassParams::Print() {
    
    for (const_iterator iter = pars.begin(); iter != pars.end(); iter++)
        cout << iter->first << " = " << iter->second << endl;
}

int ClassParams::Update(const string& key, const int& val) { 
    
    ostringstream os;
    os << val;    
    pars[key] = os.str(); 
    return pars.size(); 
}

int ClassParams::Update(const string& key, const float& val) { 
    
    ostringstream os;
    os << setprecision(8) << val;    
    pars[key] = os.str(); 
    return pars.size(); 
}


int ClassParams::Update(const string& key, const double& val) { 
    
    ostringstream os;
    os << setprecision(16) << val;    
    pars[key] = os.str(); 
    return pars.size(); 
}

int ClassParams::Update(const string& key, const bool& val) { 
       
    pars[key] = val ? "yes" : "no"; 
    return pars.size(); 
}

int ClassParams::Update(const string& key, const string& val) { 
    
    pars[key] = val; 
    return pars.size(); 
}

int ClassParams::Update(const string& key, const char * val) { 
    
    pars[key] = string(val); 
    return pars.size(); 
}

