#ifndef ASSERTM_H
#define ASSERTM_H

#include <iostream>

#ifndef NDEBUG
#   define assertM(Expr, Msg) \
    __assertM(#Expr, Expr, __FILE__, __LINE__, Msg)
#else
#   define assertM(Expr, Msg) ;
#endif

void __assertM(const char* exprString, bool expr, const char* file, int line, const char* message) {
    if (!expr) {
        std::cerr << "Assertion failed:\t" << message << "\n"
                  << "Expected:\t" << exprString << "\n"
                  << "Source:\t\t" << file << ", line " << line << "\n";
        abort();
    }
}

#endif //ASSERTM_H
