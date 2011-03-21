project(python)

cmake_minimum_required(VERSION 2.6)

# Installs python top-level wrapper
if (PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND AND Boost_FOUND)
 
  # We need first to create the basic python targets before including the
  # directories containing the python modules
  add_custom_target(torch-python-install ALL mkdir -p ${PYTHON_INSTALL_DIRECTORY} COMMAND rsync -a ${CMAKE_CURRENT_SOURCE_DIR}/lib/ ${PYTHON_INSTALL_DIRECTORY}/torch/ COMMENT "Installing root torch python files...")
  # Installs all python scripts
  file(COPY script/ DESTINATION ${CMAKE_INSTALL_PREFIX}/bin FILES_MATCHING PATTERN "*.py")

  # Compiles all installed python files
  add_custom_target(python-compilation ALL ${PYTHON_EXECUTABLE} -m compileall . 
    WORKING_DIRECTORY ${PYTHON_INSTALL_DIRECTORY} COMMENT "Compiling python files")

  add_subdirectory(core)
  add_subdirectory(database)
  add_subdirectory(ip)
  add_subdirectory(machine)
  add_subdirectory(math)
  add_subdirectory(scanning)
  add_subdirectory(sp)

endif (PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND AND Boost_FOUND)


