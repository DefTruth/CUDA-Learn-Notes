#include <iostream>

class ACls {
public:
	ACls() {
		std::cout << "Create an ACls instance and do some things!" << std::endl;
	}
};

// create a global instance.
ACls* a_inst = new ACls();  