project(post_install)

if(IS_ABSOLUTE ${CMAKE_INSTALL_PREFIX})
  set(ABSOLUTE_install_prefix "${CMAKE_INSTALL_PREFIX}")
else()
  set(ABSOLUTE_install_prefix "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_PREFIX}")
endif()
  
set(BOB_PYTHON_LIBRARY_PREFIX ${ABSOLUTE_install_prefix})
configure_file(../setup.py.in ${CMAKE_BINARY_DIR}/egg/setup-install.py @ONLY)

# Setuptools/Python Egg installation
set(pybin "${CMAKE_BINARY_DIR}/bin/egg-install")
set(pysetup "${CMAKE_BINARY_DIR}/egg/setup-install.py")
get_filename_component(basedir ${pysetup} PATH)
install(CODE "execute_process(COMMAND mkdir -pv ${ABSOLUTE_install_prefix}/${PYTHON_SITE_PACKAGES} WORKING_DIRECTORY ${basedir})")
install(CODE "execute_process(COMMAND ${pybin} ${pysetup} build --force --build-base=files-install install --prefix=${ABSOLUTE_install_prefix} WORKING_DIRECTORY ${basedir})")
