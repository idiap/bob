# Builds all python code in one shot
# Andre Anjos - 12.july.2010

project(examples)
cmake_minimum_required(VERSION 2.6)

add_subdirectory(src)

# Installs all python complements to use torch in a nice way
execute_process(python -c "import sys; print \'%d.%d\' % (sys.version_info[0], sys.version_info[1]" OUTPUT_VARIABLE PYTHON_VERSION)
file(COPY torch DESTINATION ${INSTALL_DIR}/lib/python${PYTHON_VERSION}/site-packages FILES_MATCHING PATTERN "*.py")
