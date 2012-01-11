project(python)

cmake_minimum_required(VERSION 2.6)

# Installs python top-level wrapper
if (PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND AND Boost_FOUND)
 
  # We need first to create the basic python targets before including the
  # directories containing the python modules
  add_custom_target(bob-python-install ALL mkdir -p ${PYTHON_INSTALL_DIRECTORY} COMMAND rsync -a ${CMAKE_CURRENT_SOURCE_DIR}/lib/ ${PYTHON_INSTALL_DIRECTORY}/bob/ COMMENT "Installing root bob python files...")

  # Compiles all installed python files
  add_custom_target(python-compilation ALL ${PYTHON_EXECUTABLE} -m compileall . 
    WORKING_DIRECTORY ${PYTHON_INSTALL_DIRECTORY} COMMENT "Compiling python files")

  add_subdirectory(core)
  add_subdirectory(io)
  add_subdirectory(math)
  add_subdirectory(measure)
  add_subdirectory(sp)
  add_subdirectory(ip)
  add_subdirectory(db)
  add_subdirectory(machine)
  add_subdirectory(trainer)
  add_subdirectory(face_verification)

  if (NOT QT4_FOUND)
    message(STATUS "Qt4 NOT FOUND, supressing visioner-python from build")
  else (NOT QT4_FOUND)
    add_subdirectory(visioner)
  endif (NOT QT4_FOUND)

endif (PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND AND Boost_FOUND)
