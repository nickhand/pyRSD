cdef extern from "gsl/gsl_errno.h":
    ctypedef char const_char "const char"
    
    ctypedef void gsl_error_handler_t(const_char *reason,
                                      const_char *file,
                                      int line,
                                      int gsl_errno)
    gsl_error_handler_t *gsl_set_error_handler(gsl_error_handler_t *new_handler)
