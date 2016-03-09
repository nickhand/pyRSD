#include "ClassCosmology.h"

#include <sstream>
#include <string>
#include <cmath>
#include <cstdio>

using namespace std;
using namespace Common;


// Constructors
ClassCosmology::ClassCosmology() : cl(0), dofree(false) {}

ClassCosmology::ClassCosmology(const ClassParams& pars, const string & precision_file)
: cl(0), dofree(true)
{
    // initialize
    Initialize(pars, precision_file);
    
}
/*----------------------------------------------------------------------------*/
ClassCosmology::ClassCosmology(const string & param_file, const string & precision_file)
: cl(0), dofree(true)
{
    
    string fname(param_file);
    ClassParams parameters(fname);
    verbose("Reading CLASS parameters from %s\n", fname.c_str());

    // and initialize
    Initialize(parameters, precision_file);

}

/*----------------------------------------------------------------------------*/
void ClassCosmology::Initialize(const ClassParams& pars, const string & precision_file)
{
    // set lmax value
    _lmax = 0;
    
    if (precision_file != "")
        verbose("Reading CLASS precision file from %s\n", precision_file.c_str());

    // setup the file contents
    struct file_content fc_input;       /* a temporary structure with all input parameters */
    struct file_content fc_precision;   /* a temporary structure with all precision parameters */
    struct file_content *pfc_input;     /* a pointer to either fc or fc_input */
          
    // initialize the file contents 
    fc.size = 0;
    fc_input.size = 0;
    fc_precision.size = 0;

    // determine the right input pointer
    if (precision_file != "")
        pfc_input = &fc_input;
    else
        pfc_input = &fc;
        
    // read the precision file
    if (precision_file != "")
        if (parser_read_file(const_cast<char*>(precision_file.c_str()), &fc_precision, _errmsg) == _FAILURE_){
            throw invalid_argument(_errmsg);
        }
    
    // prepare fp structure
    size_t n = pars.size();
    char file_not_needed[] = "not_needed_file.dat";
    parser_init(pfc_input, n, file_not_needed, _errmsg);
  
    // set up the input 
    int i = 0;
    for (ClassParams::const_iterator iter = pars.begin(); iter != pars.end(); iter++) {        
        string key(iter->first);
        string val(iter->second);
        strcpy(pfc_input->name[i], key.c_str());
        strcpy(pfc_input->value[i], val.c_str());
    
        // store the parameter name
        parNames.push_back(key);
    
        // identify lmax
        if (key == "l_max_scalars") {
            istringstream strstrm(val);
            strstrm >> _lmax;
        }
        i++;
    }
    verbose("%s :  : using lmax = %d\n", __FILE__ , _lmax);

    // concantanate precision
    if (precision_file != "") {
        //concatenate both
        if (parser_cat(pfc_input, &fc_precision, &fc, _errmsg) == _FAILURE_) throw invalid_argument(_errmsg);
        parser_free(&fc_precision);
    }

    // initialize the input
    if (input_init(pfc_input, &pr, &ba, &th, &pt, &tr, &pm, &sp, &nl, &le, &op, _errmsg) == _FAILURE_) 
        throw invalid_argument(_errmsg);
        
    // compute the Cls
    int status = Run();
    if (status)
        error("Error running CLASS init\n");
    
    // initialize the cl parray with the right size
    cl = new double[sp.ct_size];
    
    // free the temporary fc input
    parser_free(&fc_input);
    
    // store Omega0_m and Omegar_0 at z = 0 so we don't keep computing it
    Omega0_m_ = BackgroundValue(0., ba.index_bg_Omega_m);
    Omega0_r_ = BackgroundValue(0., ba.index_bg_Omega_r);
    
    dofree = true;
    

}
/*----------------------------------------------------------------------------*/

// destructor
ClassCosmology::~ClassCosmology()
{
    dofree && FreeStructs();
    delete [] cl;
}
/*----------------------------------------------------------------------------*/

// print content of file_content
void ClassCosmology::PrintFC() const {
    
    printf("FILE_CONTENT SIZE=%d\n",fc.size);
    for (int i=0; i<fc.size; i++) 
        printf("%d : %s = %s\n",i,fc.name[i],fc.value[i]);
}
/*----------------------------------------------------------------------------*/

// main CLASS function from main.c
int ClassCosmology::class_main(struct file_content *pfc,
			                   struct precision * ppr,
			                   struct background * pba,
			                   struct thermo * pth,
			                   struct perturbs * ppt,
			                   struct transfers * ptr,
			                    struct primordial * ppm,
			                    struct spectra * psp,
			                    struct nonlinear * pnl,
			                    struct lensing * ple,
			                    struct output * pop,
			                    ErrorMsg errmsg) {
    

    if (input_init(pfc,ppr,pba,pth,ppt,ptr,ppm,psp,pnl,ple,pop,errmsg) == _FAILURE_) {
        info("\n\nError running input_init_from_arguments \n=>%s\n",errmsg);
        dofree = false;
        return _FAILURE_;
    }

    if (background_init(ppr,pba) == _FAILURE_) {
        info("\n\nError running background_init \n=>%s\n",pba->error_message);
        dofree = false;
        return _FAILURE_;
    }

    if (thermodynamics_init(ppr,pba,pth) == _FAILURE_) {
        info("\n\nError in thermodynamics_init \n=>%s\n",pth->error_message);
        background_free(&ba);
        dofree = false;
        return _FAILURE_;
    }

    if (perturb_init(ppr,pba,pth,ppt) == _FAILURE_) {
        info("\n\nError in perturb_init \n=>%s\n",ppt->error_message);
        thermodynamics_free(&th);
        background_free(&ba);
        dofree = false;
        return _FAILURE_;
    }

    if (primordial_init(ppr,ppt,ppm) == _FAILURE_) {
        info("\n\nError in primordial_init \n=>%s\n",ppm->error_message);
        perturb_free(&pt);
        thermodynamics_free(&th);
        background_free(&ba);
        dofree = false;
        return _FAILURE_;
    }

    if (nonlinear_init(ppr,pba,pth,ppt,ppm,pnl) == _FAILURE_)  {
        info("\n\nError in nonlinear_init \n=>%s\n",pnl->error_message);
        primordial_free(&pm);
        perturb_free(&pt);
        thermodynamics_free(&th);
        background_free(&ba);
        dofree = false;
        return _FAILURE_;
    }

    if (transfer_init(ppr,pba,pth,ppt,pnl,ptr) == _FAILURE_) {
        info("\n\nError in transfer_init \n=>%s\n",ptr->error_message);
        nonlinear_free(&nl);
        primordial_free(&pm);
        perturb_free(&pt);
        thermodynamics_free(&th);
        background_free(&ba);
        dofree = false;
        return _FAILURE_;
    }

    if (spectra_init(ppr,pba,ppt,ppm,pnl,ptr,psp) == _FAILURE_) {
        info("\n\nError in spectra_init \n=>%s\n",psp->error_message);
        transfer_free(&tr);
        nonlinear_free(&nl);
        primordial_free(&pm);
        perturb_free(&pt);
        thermodynamics_free(&th);
        background_free(&ba);
        dofree = false;
        return _FAILURE_;
    }

    if (lensing_init(ppr,ppt,psp,pnl,ple) == _FAILURE_) {
        info("\n\nError in lensing_init \n=>%s\n",ple->error_message);
        info("Calling spectra free #1\n");
        spectra_free(&sp);
        transfer_free(&tr);
        nonlinear_free(&nl);
        primordial_free(&pm);
        perturb_free(&pt);
        thermodynamics_free(&th);
        background_free(&ba);
        dofree = false;
        return _FAILURE_;
    }


    dofree = true;
    return _SUCCESS_;
}
/*----------------------------------------------------------------------------*/

// run CLASS to do the desired work
int ClassCosmology::Run() {

    int status = this->class_main(&fc, &pr, &ba, &th, &pt, &tr, &pm, &sp, &nl, &le, &op, _errmsg);
    return status;
}
/*----------------------------------------------------------------------------*/

int ClassCosmology::FreeStructs() {
  
  
    if (lensing_free(&le) == _FAILURE_) {
        printf("\n\nError in lensing_free \n=>%s\n",le.error_message);
        return _FAILURE_;
    }
  
    if (nonlinear_free(&nl) == _FAILURE_) {
        printf("\n\nError in nonlinear_free \n=>%s\n",nl.error_message);
        return _FAILURE_;
    }
  
    if (spectra_free(&sp) == _FAILURE_) {
        printf("\n\nError in spectra_free \n=>%s\n",sp.error_message);
        return _FAILURE_;
    }
    
    if (primordial_free(&pm) == _FAILURE_) {
        printf("\n\nError in primordial_free \n=>%s\n",pm.error_message);
        return _FAILURE_;
    }
    
    if (transfer_free(&tr) == _FAILURE_) {
        printf("\n\nError in transfer_free \n=>%s\n",tr.error_message);
        return _FAILURE_;
    }

    if (perturb_free(&pt) == _FAILURE_) {
        printf("\n\nError in perturb_free \n=>%s\n",pt.error_message);
        return _FAILURE_;
    }

    if (thermodynamics_free(&th) == _FAILURE_) {
        printf("\n\nError in thermodynamics_free \n=>%s\n",th.error_message);
        return _FAILURE_;
    } 

    if (background_free(&ba) == _FAILURE_) {
        printf("\n\nError in background_free \n=>%s\n",ba.error_message);
        return _FAILURE_;
    }
    return _SUCCESS_;
}
/*----------------------------------------------------------------------------*/
/* Functions for getting spectra */
/*----------------------------------------------------------------------------*/

// returns Cl at a specific ell value
double ClassCosmology::GetCl(Engine::cltype t, const long &l) {

    if (!dofree) throw out_of_range("No Cls available because CLASS failed");
    if (pt.has_cls == 0) throw out_of_range("No Cls requested as output");

    if (output_total_cl_at_l(&sp,&le,&op,static_cast<double>(l),cl) == _FAILURE_){
        cerr << ">>>fail getting Cl type=" << (int)t << " @l=" << l <<endl; 
        throw out_of_range(sp.error_message);
    }
    double zecl = -1;
    double tomuk = 1e6*Tcmb();
    double tomuk2 = tomuk*tomuk;

    switch(t) {
        case TT:
            (sp.has_tt==_TRUE_) ? zecl=tomuk2*cl[sp.index_ct_tt] : throw invalid_argument("no ClTT available");
            break;
        case TE:
            (sp.has_te==_TRUE_) ? zecl=tomuk2*cl[sp.index_ct_te] : throw invalid_argument("no ClTE available");
            break; 
        case EE:
            (sp.has_ee==_TRUE_) ? zecl=tomuk2*cl[sp.index_ct_ee] : throw invalid_argument("no ClEE available");
            break;
        case BB:
            (sp.has_bb==_TRUE_) ? zecl=tomuk2*cl[sp.index_ct_bb] : throw invalid_argument("no ClBB available");
            break;
        case PP:
            (sp.has_pp==_TRUE_) ? zecl=cl[sp.index_ct_pp] : throw invalid_argument("no ClPhi-Phi available");
            break;
        case TP:
            (sp.has_tp==_TRUE_) ? zecl=tomuk*cl[sp.index_ct_tp] : throw invalid_argument("no ClT-Phi available");
            break;
        case EP:
            (sp.has_ep==_TRUE_) ? zecl=tomuk*cl[sp.index_ct_ep] : throw invalid_argument("no ClE-Phi available");
            break;
    } 
    return zecl;
}
/*----------------------------------------------------------------------------*/

// return the computed Cls at the desired ell values
int  ClassCosmology::GetCls(const vector<unsigned>& lvec, // input 
		                    parray& cltt, 
		                    parray& clte, 
		                    parray& clee, 
		                    parray& clbb) {
    
    if (pt.has_cls != _TRUE_)
        return _FAILURE_;
        
    cltt.resize(lvec.size());
    clte.resize(lvec.size());
    clee.resize(lvec.size());
    clbb.resize(lvec.size());
  
    for (size_t i=0; i < lvec.size(); i++){
        try {
            cltt[i] = GetCl(TT, lvec[i]);
            clte[i] = GetCl(TE, lvec[i]);
            clee[i] = GetCl(EE, lvec[i]);
            clbb[i] = GetCl(BB, lvec[i]);
        }
        catch(exception &e){
            throw e;
        }
    }
    return _SUCCESS_;
}
/*----------------------------------------------------------------------------*/
 
// return the computed lensing spectra the desired ell values
int ClassCosmology::GetLensing(const vector<unsigned>& lvec, // input 
		                    parray& clpp, 
		                    parray& cltp, 
		                    parray& clep  ) {
 
    if (pt.has_cl_cmb_lensing_potential != _TRUE_)
        return _FAILURE_;
        
    clpp.resize(lvec.size());
    cltp.resize(lvec.size());
    clep.resize(lvec.size());
  
    for (size_t i=0; i < lvec.size(); i++) {
        try {
            clpp[i] = GetCl(PP, lvec[i]);
            cltp[i] = GetCl(TP, lvec[i]);
            clep[i] = GetCl(EP, lvec[i]);
        }
        catch(exception &e) {
            cout << "Lensing computation failure" << endl;
            cout << __FILE__ << e.what() << endl;
            return _FAILURE_;
        }
    }
    return _SUCCESS_;
}
/*----------------------------------------------------------------------------*/

// generic private class for computing either P_lin(k) or P_nl(k)
double ClassCosmology::GetPk(double z, double k, Pktype method) {
    
    int index_md = sp.index_md_scalars;
    if ((sp.ic_size[index_md] > 1) && (method == Pk_linear)) 
        throw out_of_range("Cannot currently deal with mutiple initial conditions spectra, try only specifying one.");
        
    double pk_ic = 0;
    int status;
    double thisPk;
    double thisk = k*h();

    // check for k > kmax
    if (thisk > k_max())
        warning("Computing P(k) for k > kmax; 0 will be returned\n");
    
    // make sure to put k from h/Mpc to 1/Mpc
    if (method == Pk_linear)
        status = spectra_pk_at_k_and_z(&ba, &pm, &sp, thisk, z, &thisPk, &pk_ic);
    else
        status = spectra_pk_nl_at_k_and_z(&ba, &pm, &sp, thisk, z, &thisPk);
    
    if ((status == _FAILURE_) && (method == Pk_linear))
        error("`spectra_pk_at_k_and_z error: %s\n", sp.error_message);
    if ((status == _FAILURE_) && (method == Pk_nonlinear))
        error("`spectra_pk_nl_at_k_and_z error: %s\n", sp.error_message);
        
    return thisPk*pow3(h()); // put into units of (Mpc/h^3)
}

/*----------------------------------------------------------------------------*/

// compute the k, linear Pk in units of h/Mpc, (Mpc/h)^3
double ClassCosmology::GetPklin(double z, double k) {
    
    if (pt.has_pk_matter == _FALSE_)
        error("Cannot compute linear power if matter P(k) not requested\n");
    return GetPk(z, k, Pk_linear);
}

/*----------------------------------------------------------------------------*/

// compute the k, nonlinear Pk in units of h/Mpc, (Mpc/h)^3
double ClassCosmology::GetPknl(double z, double k) {
        
    if (nl.method == nl_none)
        error("Cannot compute nonlinear P(k) if `non linear` parameter is not set to `halofit`\n");
    return GetPk(z, k, Pk_nonlinear);
}

/*----------------------------------------------------------------------------*/

// return the k, transfer function (in CAMB format) in units of h/Mpc, unitless
int ClassCosmology::GetTk(double z, const parray& k, parray& Tk) {
    
    if (pt.has_density_transfers != _TRUE_)
        return _FAILURE_;
    
    int index_md = sp.index_md_scalars;
        
    // resize the output array
    Tk.resize(k.size());
    
    for (size_t i = 0; i < k.size(); i++) {

        // compute the linear power spectrum
        Tk[i] = GetPklin(z, k[i]);
    
        // scale by the primodial power spectrum
        double pk_primordial_k;
        double thisk = k[i]*h();
        primordial_spectrum_at_k(&pm, index_md, linear, thisk, &pk_primordial_k);
        Tk[i] /= (pk_primordial_k*k[i]*pow3(h()));
        Tk[i] = sqrt(Tk[i]);
    }
    return _SUCCESS_;
}
/*----------------------------------------------------------------------------*/
/* Functions for computing cosmological quantities as a function of z */
/*----------------------------------------------------------------------------*/
double ClassCosmology::BackgroundValue(double z, int index) const {
    
    double tau;
    double toret;
    int index2;
    parray pvecback(ba.bg_size);
    
    // find the conformal time for this redshift
    background_tau_of_z(&ba, z, &tau);

    // call to fill pvecback
    background_at_tau(&ba, tau, ba.long_info, ba.inter_normal, &index2, pvecback);
    if (index > 0) 
        toret = pvecback[index];
    else
        toret = -1;
    return toret;
}

/*----------------------------------------------------------------------------*/

// the power spectrum normalization, sigma8
double ClassCosmology::Sigma8_z(double z) const {
    

    double sigma8 = 0.;
    BackgroundValue(z, -1);
    spectra_sigma(&ba, &pm, &sp, 8./ba.h, z, &sigma8);

    return sigma8;
}

/*----------------------------------------------------------------------------*/
double ClassCosmology::rho_crit(bool cgs) const {
    
    double H = 100. * (Constants::km / Constants::second / Constants::Mpc);
    double rho_crit_cgs = 3*pow(H, 2)/(8*M_PI*Constants::G);
    
    if (cgs)
        return rho_crit_cgs;
    else
        return rho_crit_cgs / (Constants::M_sun / pow(Constants::Mpc, 3));
}

/*----------------------------------------------------------------------------*/
double ClassCosmology::rho_crit_z(double z, bool cgs) const {
    
    double Ez = H_z(z) / H0();
    return rho_crit(cgs) * pow(Ez, 2);
}


/*----------------------------------------------------------------------------*/
double ClassCosmology::Omega_m_z(double z) const {
    
    double Ez = H_z(z) / H0();
    return Omega0_m() * pow(1.+z, 3) / pow(Ez, 2);
}

/*----------------------------------------------------------------------------*/
double ClassCosmology::rho_bar_z(double z, bool cgs) const {
    
    return Omega_m_z(z) * rho_crit_z(z, cgs);
}

/*----------------------------------------------------------------------------*/
double ClassCosmology::Dm_z(double z) const {
    double Dc = Dc_z(z);
    double toret(0.);
    if (ba.sgnK == 0)
        toret = Dc;
    else if (ba.sgnK == 1)
        toret = sin(sqrt(ba.K)*Dc)/sqrt(ba.K);
    else if (ba.sgnK == -1)
        toret = sinh(sqrt(-ba.K)*Dc)/sqrt(-ba.K);
    return toret;
}

parray ClassCosmology::Dm_z(const parray& z) const {
    double (ClassCosmology::*pf)(double) const = &ClassCosmology::Dm_z;
    return EvaluateMany(bind(pf, this, placeholders::_1), z);
}

parray ClassCosmology::f_z(const parray& z) const {
    double (ClassCosmology::*pf)(double) const = &ClassCosmology::f_z;
    return EvaluateMany(bind(pf, this, placeholders::_1), z);
}

parray ClassCosmology::H_z(const parray& z) const {
    double (ClassCosmology::*pf)(double) const = &ClassCosmology::H_z;
    return EvaluateMany(bind(pf, this, placeholders::_1), z);
}

parray ClassCosmology::Da_z(const parray& z) const {
    double (ClassCosmology::*pf)(double) const = &ClassCosmology::Da_z;
    return EvaluateMany(bind(pf, this, placeholders::_1), z);
}

parray ClassCosmology::Dc_z(const parray& z) const {
    double (ClassCosmology::*pf)(double) const = &ClassCosmology::Dc_z;
    return EvaluateMany(bind(pf, this, placeholders::_1), z);
}

parray ClassCosmology::D_z(const parray& z) const {
    double (ClassCosmology::*pf)(double) const = &ClassCosmology::D_z;
    return EvaluateMany(bind(pf, this, placeholders::_1), z);
}

parray ClassCosmology::Sigma8_z(const parray& z) const {
    double (ClassCosmology::*pf)(double) const = &ClassCosmology::Sigma8_z;
    return EvaluateMany(bind(pf, this, placeholders::_1), z);
}

parray ClassCosmology::Omega_m_z(const parray& z) const {
    double (ClassCosmology::*pf)(double) const = &ClassCosmology::Omega_m_z;
    return EvaluateMany(bind(pf, this, placeholders::_1), z);
}

parray ClassCosmology::rho_bar_z(const parray& z, bool cgs) const {
    double (ClassCosmology::*pf)(double, bool) const = &ClassCosmology::rho_bar_z;
    return EvaluateMany(bind(pf, this, placeholders::_1, cgs), z);
}

parray ClassCosmology::rho_crit_z(const parray& z, bool cgs) const {
    double (ClassCosmology::*pf)(double, bool) const = &ClassCosmology::rho_crit_z;;
    return EvaluateMany(bind(pf, this, placeholders::_1, cgs), z);
}

double ClassCosmology::dV(double z) const {
    double c_kms = Constants::c_light/(Constants::km/Constants::second);
    return c_kms * pow2((1+z)*Da_z(z)) / H_z(z) / pow3(1e3);  
}

parray ClassCosmology::dV(const parray& z) const {
    double (ClassCosmology::*pf)(double) const = &ClassCosmology::dV;
    return EvaluateMany(bind(pf, this, placeholders::_1), z);
}

double ClassCosmology::V(double zmin, double zmax, int Nz) const {
    parray z = parray::linspace(zmin, zmax, Nz);
    parray integrand = dV(z);
    return 4*M_PI*SimpsIntegrate(z, integrand);
}

parray ClassCosmology::V(const parray& zmin, const parray& zmax, int Nz) const {
    
    int N = (int) zmin.size();
    parray toret(N);
    #pragma omp parallel for
    for(int i = 0; i < N; i++)
        toret[i] = V(zmin[i], zmax[i], Nz);
    return toret;
}


