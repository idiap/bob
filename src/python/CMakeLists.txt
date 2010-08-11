project(python)

cmake_minimum_required(VERSION 2.6)

# Installs python top-level wrapper
if (PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND AND Boost_FOUND)
  set(pydir ${CMAKE_INSTALL_PREFIX}/lib/python${PYTHON_VERSION})
  add_custom_target(torch-python-install ALL mkdir -p ${pydir} COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR}/torch ${pydir} COMMENT "Installing torch python files...")
endif (PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND AND Boost_FOUND)

# Compiles all installed python files
add_custom_target(python-compilation ALL ${PYTHON_EXECUTABLE} -m compileall .
  WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}/lib/python${PYTHON_VERSION} 
  COMMENT "Compiling python files")

# Installs all python scripts
file(COPY script/ DESTINATION ${CMAKE_INSTALL_PREFIX}/bin FILES_MATCHING PATTERN "*.py")

# Finally add all python tests
add_test(python-video-tests ${CMAKE_CURRENT_SOURCE_DIR}/test/test_video.py)
add_test(python-facefinder-tests ${CMAKE_CURRENT_SOURCE_DIR}/test/test_facefinder.py)
add_test(python-cropping-tests ${CMAKE_CURRENT_SOURCE_DIR}/test/test_cropping.py)
add_test(python-videotensor-tests ${CMAKE_CURRENT_SOURCE_DIR}/test/test_videotensor.py)
