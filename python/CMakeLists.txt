project(python)

cmake_minimum_required(VERSION 2.6)

# Installs python top-level wrapper
if (PYTHON_EXECUTABLE AND PYTHONLIBS_FOUND AND Boost_FOUND)

  # Sets up a local version of the interpreter
  if(BOB_INSTALL_PYTHON_INTERPRETER)
    bob_configure_bobpython(bin/python.in bin/python "ON")
    if(IPYTHON_EXECUTABLE)
      bob_configure_bobpython(bin/ipython.in bin/ipython "ON")
    endif()
  else()
    # just configure it locally for build-time usage
    set(BOB_PYTHONPATH "${CMAKE_BINARY_DIR}/${PYTHON_SITE_PACKAGES}")
    configure_file(bin/python.in ${CMAKE_BINARY_DIR}/bin/python @ONLY)
    if(IPYTHON_EXECUTABLE)
      configure_file(bin/ipython.in ${CMAKE_BINARY_DIR}/bin/ipython @ONLY)
    endif()
  endif()

  # special version of python for setuptools usage on installation area
  if(IS_ABSOLUTE ${CMAKE_INSTALL_PREFIX})
    set(ABSOLUTE_install_prefix "${CMAKE_INSTALL_PREFIX}")
  else()
    set(ABSOLUTE_install_prefix "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_PREFIX}")
  endif()
  set(BOB_PYTHONPATH "${CMAKE_BINARY_DIR}/${PYTHON_SITE_PACKAGES}")
  configure_file(bin/egg-python.in ${CMAKE_BINARY_DIR}/bin/egg-build @ONLY)
  set(BOB_PYTHONPATH "${ABSOLUTE_install_prefix}/${PYTHON_SITE_PACKAGES}")
  configure_file(bin/egg-python.in ${CMAKE_BINARY_DIR}/bin/egg-install @ONLY)

  # An easy handler to debug python stuff
  configure_file(bin/gdbpy.in ${CMAKE_BINARY_DIR}/bin/gdbpy @ONLY)

  bob_python_package(root root "" "")
  
  # Installs and tests our scripts
  # TTT bob_python_add_test(root bob_config.py)

  get_property(BOB_PYTHON_COMPILE_DEFINITIONS DIRECTORY PROPERTY COMPILE_DEFINITIONS)
  get_property(BOB_PYTHON_INCLUDE_DIRECTORIES DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
  get_property(BOB_PYTHON_LINK_DIRECTORIES DIRECTORY PROPERTY LINK_DIRECTORIES)

  set(BOB_PYTHON_LIBRARY_PREFIX ${CMAKE_BINARY_DIR})
  configure_file(setup.py.in ${CMAKE_BINARY_DIR}/egg/build/setup.py @ONLY)

  add_subdirectory(integration)
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
    add_subdirectory(visioner)
    add_subdirectory(daq)
  endif()

  # Always keep this one last
  add_subdirectory(post_install)

endif ()
