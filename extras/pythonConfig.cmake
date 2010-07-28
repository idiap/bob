# Tries to find a local version of Python installed
# Andre Anjos - 09.july.2010

# This includes in OSX, the MacPorts installation path, comment if you want to
# use the stock python distribution.
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} /opt/local/lib)

include(FindPythonLibs)
include(FindPythonInterp)

# This will calculate the current version of python found
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import sys; print '%d.%d' % (sys.version_info[0], sys.version_info[1])" OUTPUT_VARIABLE PYTHON_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
set(PYTHON_VERSION ${PYTHON_VERSION} CACHE INTERNAL "python")
