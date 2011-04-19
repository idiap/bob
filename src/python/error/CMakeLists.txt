project(error-python)
cmake_minimum_required(VERSION 2.6)

set(pysrc
   "src/eval.cc"
   "src/main.cc"
   )

include(../../cmake/macros.cmake)

torch_python_package_bindings(error "${pysrc}")
#add_dependencies(pytorch_error core-array-python_header_install)
torch_python_install(error)
  
# Our test units
#torch_python_add_test(python-error-basic ${CMAKE_CURRENT_SOURCE_DIR}/test/test_error.py)

# Installs all python scripts
#file(COPY script/ DESTINATION ${CMAKE_INSTALL_PREFIX}/bin FILES_MATCHING PATTERN "*.py")

# Our examples (run and install)
#torch_python_add_test(python-error-load-example
#  ${CMAKE_CURRENT_SOURCE_DIR}/example/loadConfig.py --test)
#torch_example_install("python" "error" example/loadConfig.py)
