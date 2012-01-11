project(face_verification-python)
cmake_minimum_required(VERSION 2.6)
 
include(../../cmake/macros.cmake)

# Our test units
bob_python_add_test(python-face_verification-bancasmall-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_bancasmall.py)

# Installs all python scripts
if((${CMAKE_SYSTEM_NAME} MATCHES "Darwin"))
  execute_process(COMMAND mktemp -d face_verification.XXXXXXXXXX OUTPUT_VARIABLE FACE_VERIF_TEMP_DIRECTORY OUTPUT_STRIP_TRAILING_WHITESPACE)
else((${CMAKE_SYSTEM_NAME} MATCHES "Darwin"))
  execute_process(COMMAND mktemp -d OUTPUT_VARIABLE FACE_VERIF_TEMP_DIRECTORY OUTPUT_STRIP_TRAILING_WHITESPACE)
endif((${CMAKE_SYSTEM_NAME} MATCHES "Darwin"))
execute_process(COMMAND rmdir ${FACE_VERIF_TEMP_DIRECTORY})

set_property(TEST python-face_verification-bancasmall-tests APPEND PROPERTY ENVIRONMENT
  "BOB_FACE_VERIF_TEMP_DIRECTORY=${FACE_VERIF_TEMP_DIRECTORY}")
file(COPY script/ DESTINATION ${CMAKE_INSTALL_PREFIX}/bin FILES_MATCHING PATTERN "*.py")
