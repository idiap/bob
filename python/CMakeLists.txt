project(python)

cmake_minimum_required(VERSION 2.6)

# Installs python top-level wrapper
if (PYTHON_EXECUTABLE AND PYTHONLIBS_FOUND AND Boost_FOUND)

  # Sets up a local version of the interpreter
  set(BOB_PYTHONPATH ${CMAKE_BINARY_DIR}/lib/python${PYTHON_VERSION})
  if(BOB_INSTALL_PYTHON_INTERPRETER)
    bob_configure_bobpython(bin/python.in bin/python bin)
  else()
    # just configure it locally for build-time usage
    configure_file(bin/python.in ${CMAKE_BINARY_DIR}/bin/python @ONLY)
  endif()

  # An easy handler to debug python stuff
  set(BOB_PYTHONPATH ${CMAKE_BINARY_DIR}/lib/python${PYTHON_VERSION})
  configure_file(bin/gdbpy.in ${CMAKE_BINARY_DIR}/bin/gdbpy @ONLY)

  bob_python_bindings(root root "" "")
  
  # Installs and tests our scripts
  bob_python_script(root bob_config.py lib/script/config.py)
  bob_python_add_test(root bob_config.py)

  add_subdirectory(core)
  add_subdirectory(io)
  add_subdirectory(math)
  add_subdirectory(measure)
  add_subdirectory(db)
  add_subdirectory(sp)
  add_subdirectory(ip)
  add_subdirectory(machine)
  add_subdirectory(trainer)

  if(QT4_FOUND)
    if (BOB_VISIONER_ENABLED)
      add_subdirectory(visioner)
    endif()

    add_subdirectory(daq)
  endif()

  add_subdirectory(post_install)

endif ()
