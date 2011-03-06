# Tries to find a local version of Python installed
# Andre Anjos - 09.july.2010

include(FindPythonLibs)
include(FindPythonInterp)

# This will calculate the current version of python found
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import sys; print '%d.%d' % (sys.version_info[0], sys.version_info[1])" OUTPUT_VARIABLE PYTHON_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
set(PYTHON_VERSION ${PYTHON_VERSION} CACHE INTERNAL "python")

# This will calculate the include path for numpy
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import numpy; print numpy.get_include()" OUTPUT_VARIABLE PYTHON_NUMPY_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)

set(python_INCLUDE_DIRS "${PYTHON_INCLUDE_DIR};${PYTHON_NUMPY_INCLUDE_DIR}" CACHE INTERNAL "incdirs")
get_filename_component(python_LIBRARY_DIRS ${PYTHON_LIBRARY} PATH CACHE)
  
set(PYTHON_INSTALL_DIRECTORY ${CMAKE_INSTALL_PREFIX}/lib/python${PYTHON_VERSION}
  CACHE INTERNAL "python")

include_directories(SYSTEM ${python_INCLUDE_DIRS})
