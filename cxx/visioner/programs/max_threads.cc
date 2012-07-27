#include <iostream>
#include <boost/thread.hpp>

int main(int, char* []) {	
  std::cout << boost::thread::hardware_concurrency() << "\n";
  return EXIT_SUCCESS;
}
