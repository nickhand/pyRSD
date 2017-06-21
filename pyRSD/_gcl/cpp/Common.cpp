#include <cmath>
#include <cstdarg>
#include <cstdlib>
#include <stdexcept>
#include <iostream>
#include "Common.h"

void Common::throw_error(const char* msg, std::string file, int lineno)
{
    std::string emsg(msg);
    std::string out = emsg + ", file: " + file + ", line: " + std::to_string(lineno);
    throw std::runtime_error(out);
}

void Common::throw_error(std::string msg, std::string file, int lineno)
{
    std::string out = msg + " (file: " + file + ", line: " + std::to_string(lineno) + ")";
    throw std::runtime_error(out);
}

void Common::write(FILE* stream, const char* format, ...) {
    va_list args;
    char dest[6000];
    va_start(args, format);
    vsnprintf(dest, 2048, format, args);
    va_end(args);
    fprintf(stream, dest);
}


void Common::info(const char* format, ...) {
    va_list ap;
    va_start(ap, format);
    vfprintf(stdout, format, ap);
    va_end(ap);
    fflush(stdout);
}

void Common::verbose(const char* format, ...) {
    va_list ap;
    va_start(ap, format);
#ifdef VERBOSE
    vfprintf(stdout, format, ap);
    va_end(ap);
    fflush(stdout);
#endif
}

void Common::debug(const char* format, ...) {
    va_list ap;
    va_start(ap, format);
#ifdef DEBUG
    vfprintf(stdout, format, ap);
    va_end(ap);
    fflush(stdout);
#endif
}

void Common::warning(const char* format, ...) {
    va_list ap;
    va_start(ap, format);
    vfprintf(stderr, format, ap);
    va_end(ap);
    fflush(stderr);
}

void Common::error(const char* format, ...) {
    va_list ap;
    va_start(ap, format);
    vfprintf(stderr, format, ap);
    va_end(ap);
    fflush(stderr);
    abort();
}
