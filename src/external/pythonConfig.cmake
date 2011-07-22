# Tries to find a local version of Python installed
# Andre Anjos - 09.july.2010

# We pre-calculate the default python version
execute_process(COMMAND python -c "import sys; print '%d.%d' % (sys.version_info[0], sys.version_info[1])" OUTPUT_VARIABLE PYTHON_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)

# Cache this variable so it stays
set(PYTHON_VERSION ${PYTHON_VERSION} CACHE INTERNAL "python")

# We then set the preference to use that
set(Python_ADDITIONAL_VERSIONS ${PYTHON_VERSION})

include(FindPythonLibs)
include(FindPythonInterp)

# This calculates the correct python installation prefix for the current
# interpreter
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import sys; print sys.prefix" OUTPUT_VARIABLE PYTHON_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)

# This will calculate the include path for numpy
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import numpy; print numpy.get_include()" OUTPUT_VARIABLE PYTHON_NUMPY_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)

# Do not use the include dir path found by FindPythonLibs as it does not
# work properly on OSX (we end up getting the system path if another python
# version is selected). This may cause compilation problems.
set(python_INCLUDE_DIRS "${PYTHON_PREFIX}/include/python${PYTHON_VERSION};${PYTHON_NUMPY_INCLUDE_DIR}" CACHE INTERNAL "incdirs")
get_filename_component(python_LIBRARY_DIRS ${PYTHON_LIBRARY} PATH CACHE)
  
set(PYTHON_INSTALL_DIRECTORY ${CMAKE_INSTALL_PREFIX}/lib/python${PYTHON_VERSION}
  CACHE INTERNAL "python")

include_directories(SYSTEM ${python_INCLUDE_DIRS})
