project(python)

cmake_minimum_required(VERSION 2.6)

# Installs python top-level wrapper
if (PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND AND Boost_FOUND)
  add_custom_target(pybob_root ALL
                    COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/lib" "${CMAKE_BINARY_DIR}/lib/python${PYTHON_VERSION}/bob/")

  # Adds specific python sources
  set(pysrc "lib/__init__.py")
  bob_python_module(root ${pysrc})

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
    add_subdirectory(daq)
  endif (NOT QT4_FOUND)

  # TODO Problem: this command is executed *before* the add_subdirectory
  # Compiles all installed python files
  install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} -m compileall . WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}/lib)")
endif (PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND AND Boost_FOUND)
