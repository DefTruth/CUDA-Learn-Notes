#include "libc/funcc.h"
#include "liba/func.h"
int AddFuncC(int a, int b) {return AddFuncA(a, a) + AddFuncA(b, b);}
