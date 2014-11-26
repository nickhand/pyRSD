#include <cmath>
#include <cstdarg>
#include <cstdlib>

#include "Common.h"

void write(FILE* stream, const char* format, ...) {
    va_list args;
    char dest[6000];
    va_start(args, format);
    vsnprintf(dest, 2048, format, args);
    va_end(args);
    fprintf(stream, dest);
}


void info(const char* format, ...) {
    va_list ap;
    va_start(ap, format);
    vfprintf(stdout, format, ap);
    va_end(ap);
    fflush(stdout);
}

void verbose(const char* format, ...) {
#ifdef VERBOSE
    va_list ap;
    va_start(ap, format);
    vfprintf(stdout, format, ap);
    va_end(ap);
    fflush(stdout);
#endif
}

void debug(const char* format, ...) {
#ifdef DEBUG
    va_list ap;
    va_start(ap, format);
    vfprintf(stdout, format, ap);
    va_end(ap);
    fflush(stdout);
#endif
}

void warning(const char* format, ...) {
    va_list ap;
    va_start(ap, format);
    vfprintf(stderr, format, ap);
    va_end(ap);
    fflush(stderr);
}

void error(const char* format, ...) {
    va_list ap;
    va_start(ap, format);
    vfprintf(stderr, format, ap);
    va_end(ap);
    fflush(stderr);
    abort();
}
