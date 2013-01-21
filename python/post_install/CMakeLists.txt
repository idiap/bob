project(post_install)

if(IS_ABSOLUTE ${CMAKE_INSTALL_PREFIX})
  set(ABSOLUTE_install_prefix "${CMAKE_INSTALL_PREFIX}")
else()
  set(ABSOLUTE_install_prefix "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_PREFIX}")
endif()
  
# Setuptools/Python Egg installation
set(pybin "${CMAKE_BINARY_DIR}/bin/egg-install")
set(pysetup "${CMAKE_SOURCE_DIR}/python/setup.py")
get_filename_component(basedir ${pysetup} PATH)
install(CODE "execute_process(COMMAND mkdir -pv \$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/${PYTHON_SITE_PACKAGES} WORKING_DIRECTORY ${basedir})")
install(CODE "execute_process(COMMAND ${pybin} ${pysetup} build ${BOB_SETUPTOOLS_COMPILER} --force --build-base=${CMAKE_BINARY_DIR}/egg-install install --force --prefix=\$ENV{DESTDIR}${ABSOLUTE_install_prefix} --record=${CMAKE_BINARY_DIR}/egg-install-record.txt --single-version-externally-managed WORKING_DIRECTORY ${basedir})")
install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/python/bin/fix_scripts.py $ENV{DESTDIR}${ABSOLUTE_install_prefix}/bin \$ENV{DESTDIR})")
