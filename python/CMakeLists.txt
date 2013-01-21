project(python)

# Configures the wrapper to the python binary that automatically sets the
# correct environment.
#
# file_input: path to the script to configure
# script_name: the destination name for the script
#
# Example: bob_configure_bobpython(bin/python.in bin/python)
macro(bob_configure_bobpython file_input script_name)

  get_filename_component(install_dir ${script_name} PATH)
  get_filename_component(install_name ${script_name} NAME)

  if ("${install_name}" STREQUAL "") 
    set(install_name ".")
  endif()

  # configures and installs the build directory version of the script
  set(ABSOLUTE_build_dir ${CMAKE_BINARY_DIR})
  set(BOB_PYTHONPATH ${CMAKE_BINARY_DIR}/${PYTHON_SITE_PACKAGES})
  configure_file(${file_input} ${ABSOLUTE_build_dir}/${script_name} @ONLY)

  # gets the absolute installation prefix
  if(IS_ABSOLUTE ${CMAKE_INSTALL_PREFIX})
    set(ABSOLUTE_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
  else()
    set(ABSOLUTE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_PREFIX})
  endif()

  # configures and installs the installation directory version of the script
  set(ABSOLUTE_install_dir ${ABSOLUTE_INSTALL_PREFIX}/${install_dir})
  set(BOB_PYTHONPATH ${ABSOLUTE_INSTALL_PREFIX}/${PYTHON_SITE_PACKAGES})
  configure_file(${file_input} ${CMAKE_BINARY_DIR}/to_install/${install_name} @ONLY)
  install(PROGRAMS ${CMAKE_BINARY_DIR}/to_install/${install_name} DESTINATION ${install_dir} RENAME ${install_name})

endmacro()

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
set(BOB_PKG_CONFIG_PATH "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/pkgconfig")
configure_file(bin/egg-build.in ${CMAKE_BINARY_DIR}/bin/egg-build @ONLY)

set(BOB_PKG_CONFIG_PATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/pkgconfig")
set(BOB_PYTHONPATH "${ABSOLUTE_install_prefix}/${PYTHON_SITE_PACKAGES}")
configure_file(bin/egg-install.in ${CMAKE_BINARY_DIR}/bin/egg-install @ONLY)

# An easy handler to debug python stuff
set(BOB_PYTHONPATH "${CMAKE_BINARY_DIR}/${PYTHON_SITE_PACKAGES}")
configure_file(bin/gdbpy.in ${CMAKE_BINARY_DIR}/bin/gdbpy @ONLY)

set(pybin "${CMAKE_BINARY_DIR}/bin/egg-build")
set(pysetup "${CMAKE_SOURCE_DIR}/python/setup.py")
get_filename_component(basedir ${pysetup} PATH)
add_custom_target(egg ALL mkdir -pv ${CMAKE_BINARY_DIR}/${PYTHON_SITE_PACKAGES}
  COMMAND ${pybin} ${pysetup} build --compiler=cygwin --force --build-base=${CMAKE_BINARY_DIR}/egg-build install --force --prefix=${CMAKE_BINARY_DIR} --record=${CMAKE_BINARY_DIR}/egg-build-record.txt --single-version-externally-managed WORKING_DIRECTORY ${basedir} DEPENDS ${ENABLED_PACKAGES})
add_custom_target(egg-scripts ALL ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/bin/fix_scripts.py ${CMAKE_BINARY_DIR}/bin)
add_dependencies(egg-scripts egg)

# Testing
# An easy handler for nose tests
find_program(NOSETESTS_EXECUTABLE NAMES nosetests-${PYTHON_VERSION} nosetests
  HINTS $ENV{SPHINX_DIR} PATH_SUFFIXES bin
  DOC "Nose python test scanner and runner"
  )
set(BOB_PYTHONPATH "${CMAKE_BINARY_DIR}/${PYTHON_SITE_PACKAGES}")
configure_file(bin/nosetests.in ${CMAKE_BINARY_DIR}/bin/nosetests @ONLY)
add_custom_target(nosetests ${CMAKE_BINARY_DIR}/bin/nosetests --with-insulate -v bob)
add_dependencies(nosetests egg-scripts)

# Documentation
add_subdirectory(doc)

# Always keep this one last
add_subdirectory(post_install)
