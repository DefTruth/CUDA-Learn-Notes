#include "liba/func.h"
#include "libb/func.h"
#include <iostream>

class ACls;
extern ACls* a_inst;

int main(int argc, char* argv[]) {
  std::cout << "a_inst: " << a_inst << std::endl;
  std::cout << "AddFuncA(2, 3): " << AddFuncA(2, 3) << std::endl;
  std::cout << "AddFuncB(2.0f, 3.0f): " << AddFuncB(2.0f, 3.0f) << std::endl; 
}
