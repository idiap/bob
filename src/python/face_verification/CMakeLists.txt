project(face_verification-python)
cmake_minimum_required(VERSION 2.6)
 
include(../../cmake/macros.cmake)

# Our test units
torch_python_add_test(python-face_verification-bancasmall-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_bancasmall.py)

# Installs all python scripts
file(COPY script/ DESTINATION ${CMAKE_INSTALL_PREFIX}/bin FILES_MATCHING PATTERN "*.py")

