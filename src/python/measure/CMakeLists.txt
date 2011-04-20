project(measure-python)
cmake_minimum_required(VERSION 2.6)

set(pysrc
   "src/error.cc"
   "src/main.cc"
   )

include(../../cmake/macros.cmake)

torch_python_package_bindings(measure "${pysrc}")
#add_dependencies(pytorch_measure core-array-python_header_install)
torch_python_install(measure)
  
# Our test units
#torch_python_add_test(python-measure-basic ${CMAKE_CURRENT_SOURCE_DIR}/test/test_measure.py)

# Installs all python scripts
#file(COPY script/ DESTINATION ${CMAKE_INSTALL_PREFIX}/bin FILES_MATCHING PATTERN "*.py")

# Our examples (run and install)
#torch_python_add_test(python-measure-load-example
#  ${CMAKE_CURRENT_SOURCE_DIR}/example/loadConfig.py --test)
#torch_example_install("python" "measure" example/loadConfig.py)
