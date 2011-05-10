project(machine-python)
cmake_minimum_required(VERSION 2.6)

set(pysrc
   "src/main.cc"
   )

include(../../cmake/macros.cmake)

torch_python_package_bindings(machine "${pysrc}")
add_dependencies(pytorch_machine core-array-python_header_install machine-cxx_header_install)
torch_python_install(machine)
  
# Our test units
torch_python_add_test(python-machine-kmeans-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_machine.py)

# Installs all python scripts
#file(COPY script/ DESTINATION ${CMAKE_INSTALL_PREFIX}/bin FILES_MATCHING PATTERN "*.py")

# Our examples (run and install)
#torch_example_install("python" "machine" example/optflowHS.py)
