project(src)

cmake_minimum_required(VERSION 2.6)

# This makes sure we have all headers within reach before starting to compile
include_directories(${CMAKE_INSTALL_PREFIX}/include/torch)

# Installs python top-level wrapper
if (PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND AND Boost_FOUND)
  set(pydir ${CMAKE_INSTALL_PREFIX}/lib/python${PYTHON_VERSION})
  add_custom_target(torch-python-install ALL mkdir -p ${pydir} COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR}/python/torch ${pydir} COMMENT "Installing torch python files...")
endif (PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND AND Boost_FOUND)

# Compiles all installed python files
add_custom_target(python-compilation ALL ${PYTHON_EXECUTABLE} -m compileall .
  WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}/lib/python${PYTHON_VERSION} 
  COMMENT "Compiling python files")

foreach(proj oourafft lbfgs th core sp ap ip measurer machine scanning trainer)
	add_subdirectory(${proj})
endforeach(proj)

install(EXPORT torch DESTINATION share/cmake)
