project(config-python)
cmake_minimum_required(VERSION 2.6)

set(pysrc
   "src/configuration.cc"
   "src/exception.cc"
   "src/main.cc"
   )

include(../../cmake/macros.cmake)

torch_python_package_bindings(config "${pysrc}")
add_dependencies(pytorch_config core-array-python_header_install)
torch_python_install(config)
  
# Our test units
torch_python_add_test(python-config-basic ${CMAKE_CURRENT_SOURCE_DIR}/test/test_config.py)

# Installs all python scripts
#file(COPY script/ DESTINATION ${CMAKE_INSTALL_PREFIX}/bin FILES_MATCHING PATTERN "*.py")

# Our examples (run and install)
#torch_python_add_test(python-config-load-example
#  ${CMAKE_CURRENT_SOURCE_DIR}/example/loadConfig.py --test)
#torch_example_install("python" "config" example/loadConfig.py)
