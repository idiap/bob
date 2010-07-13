# Builds all python code in one shot
# Andre Anjos - 12.july.2010

project(pytorch)
cmake_minimum_required(VERSION 2.6)

add_subdirectory(src)

# Installs all python complements to use torch in a nice way
file(COPY torch DESTINATION ${INSTALL_DIR}/lib/python${PYTHON_VERSION} FILES_MATCHING PATTERN "*.py")

# Compiles all installed python files
add_custom_target(python-compilation ALL ${PYTHON_EXECUTABLE} -m compileall .
  WORKING_DIRECTORY ${INSTALL_DIR}/lib/python${PYTHON_VERSION} 
  COMMENT "Compiling python files")

add_subdirectory(tests)
