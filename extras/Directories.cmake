# This macro just defines a few standard directories
execute_process(COMMAND uname -s OUTPUT_VARIABLE unames)
execute_process(COMMAND uname -m OUTPUT_VARIABLE unamem)
execute_process(COMMAND pwd OUTPUT_VARIABLE pwd)
set(INSTALL_DIR "${pwd}/build_${unames}_${unamem}")
string(REGEX REPLACE "\n" "" INSTALL_DIR ${INSTALL_DIR})
set(INCLUDE_DIR "${pwd}/include")
string(REGEX REPLACE "\n" "" INCLUDE_DIR ${INCLUDE_DIR})
