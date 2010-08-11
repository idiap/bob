project(src)

cmake_minimum_required(VERSION 2.6)

# This makes sure we have all headers within reach before starting to compile
include_directories(${CMAKE_INSTALL_PREFIX}/include/torch)

foreach(proj oourafft lbfgs th core sp ap ip measurer machine scanning trainer)
	add_subdirectory(${proj})
endforeach(proj)

# Installs python top-level wrapper
file(COPY torch DESTINATION ${INSTALL_DIR}/lib/python${PYTHON_VERSION} FILES_MATCHING PATTERN "*.py")

# Compiles all installed python files
add_custom_target(python-compilation ALL ${PYTHON_EXECUTABLE} -m compileall .
  WORKING_DIRECTORY ${INSTALL_DIR}/lib/python${PYTHON_VERSION} 
  COMMENT "Compiling python files")

install(EXPORT torch DESTINATION share/cmake)
