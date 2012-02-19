project(face_verification-python)
cmake_minimum_required(VERSION 2.6)
 
set(pysrc
   "lib/__init__.py"
   "lib/test/__init__.py"
   "lib/test/bancasmall.py"
   )

bob_python_package_bindings(face_verification "" "${pysrc}" "")

# Our test units
bob_python_add_unittest(face_verification lib/test/bancasmall.py main
  ${CMAKE_CURRENT_SOURCE_DIR}/data)

# Installs all python scripts
if((${CMAKE_SYSTEM_NAME} MATCHES "Darwin"))
  execute_process(COMMAND mktemp -d face_verification.XXXXXXXXXX OUTPUT_VARIABLE FACE_VERIF_TEMP_DIRECTORY OUTPUT_STRIP_TRAILING_WHITESPACE)
else((${CMAKE_SYSTEM_NAME} MATCHES "Darwin"))
  execute_process(COMMAND mktemp -d OUTPUT_VARIABLE FACE_VERIF_TEMP_DIRECTORY OUTPUT_STRIP_TRAILING_WHITESPACE)
endif((${CMAKE_SYSTEM_NAME} MATCHES "Darwin"))
execute_process(COMMAND rmdir ${FACE_VERIF_TEMP_DIRECTORY})

set_property(TEST python-face_verification-bancasmall-main APPEND PROPERTY ENVIRONMENT "BOB_FACE_VERIF_TEMP_DIRECTORY=${FACE_VERIF_TEMP_DIRECTORY}")
