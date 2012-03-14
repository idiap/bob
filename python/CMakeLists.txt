project(python)

cmake_minimum_required(VERSION 2.6)

# Installs python top-level wrapper
if (PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND AND Boost_FOUND)

  # Sets up a local version of the interpreter
  set(BOB_PYTHONPATH ${CMAKE_BINARY_DIR}/lib/python${PYTHON_VERSION})
  configure_file(bin/python.in ${CMAKE_BINARY_DIR}/bin/python @ONLY)

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
  add_subdirectory(face_verification)

  if(QT4_FOUND)
    if (BOB_VISIONER_ENABLED)
      add_subdirectory(visioner)
    endif()

    add_subdirectory(daq)
  endif()

  add_subdirectory(post_install)
endif (PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND AND Boost_FOUND)
