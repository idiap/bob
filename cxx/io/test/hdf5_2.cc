#include <stdexcept>
#include <boost/filesystem.hpp>
#include <boost/make_shared.hpp>
#include "io/HDF5File.h"

int main() {
  //const std::string filename = temp_file();
  const std::string filename("crashtest.hdf5");
  boost::shared_ptr<bob::io::HDF5File> config = 
    boost::make_shared<bob::io::HDF5File>(filename, bob::io::HDF5File::trunc);
  config->set("integer", 3);
  config.reset();

  // Try to write on a read-only version opened
  config = boost::make_shared<bob::io::HDF5File>(filename, bob::io::HDF5File::in);

  // This should raise an exception and that is it, the program would segfault
  try {
    config->set("float", 3.14);
  }
  catch (std::exception& e) {
    std::cout << "Got exception: " << e.what() << std::endl;
    //config.reset(); //try to cleanup
  }

  // Clean-up
  //boost::filesystem::remove(filename);
}
