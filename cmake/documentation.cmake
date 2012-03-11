# New targets for building documentation

# add a target to generate API documentation with Doxygen
find_package(Doxygen)

if(DOXYGEN_FOUND)

  configure_file(${CMAKE_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

  add_custom_target(doxygen 
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile 
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Generating C++ API documentation with Doxygen" VERBATIM
  )

  add_custom_target(install-doxygen
      COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${CMAKE_BINARY_DIR}/doxygen" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/doc/bob/doxygen"
      DEPENDS doxygen
    COMMENT "Installing C++ API documentation, DESTDIR = '$ENV{DESTDIR}'" VERBATIM
  )

endif(DOXYGEN_FOUND)

# add a target to generate user documentation with Sphinx
find_program(SPHINX_EXECUTABLE "sphinx-build")
if(NOT SPHINX_EXECUTABLE)
  find_program(SPHINX_EXECUTABLE "sphinx-build-${PYTHON_VERSION}")
endif()

if(SPHINX_EXECUTABLE)

  configure_file(${CMAKE_SOURCE_DIR}/conf.py.in ${CMAKE_CURRENT_BINARY_DIR}/conf.py)

  add_custom_target(sphinx
    COMMAND ${SPHINX_EXECUTABLE} -c ${CMAKE_BINARY_DIR} -b html ${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR}/sphinx/html
    COMMENT "Generating (html) User Guide with Sphinx" VERBATIM
  )

  add_custom_target(sphinx-latex 
    COMMAND ${SPHINX_EXECUTABLE} -c ${CMAKE_BINARY_DIR} -b latex ${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR}/sphinx/latex 
    COMMENT "Generating (LaTeX2e) User Guide with Sphinx" VERBATIM
  )

  add_custom_target(sphinx-coverage 
    COMMAND ${SPHINX_EXECUTABLE} -c ${CMAKE_BINARY_DIR} -b coverage ${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR}/sphinx/coverage 
    COMMENT "Generating (coverage) report with Sphinx" VERBATIM
  )

  add_custom_target(install-sphinx
      COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${CMAKE_BINARY_DIR}/sphinx" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/doc/bob/sphinx"
      DEPENDS sphinx
    COMMENT "Installing (Sphinx) User Guide" VERBATIM
  )

endif(SPHINX_EXECUTABLE)
