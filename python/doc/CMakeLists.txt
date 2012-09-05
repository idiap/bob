include(FindDoxygen)
include(FindLATEX)

# Find Sphinx (python documentation generator)
find_program(SPHINX_EXECUTABLE NAMES sphinx-build-${PYTHON_VERSION} sphinx-build
  HINTS $ENV{SPHINX_DIR} PATH_SUFFIXES bin
  DOC "Sphinx documentation generator"
  )

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(Sphinx DEFAULT_MSG SPHINX_EXECUTABLE)
mark_as_advanced(SPHINX_EXECUTABLE)

if(DOXYGEN_FOUND AND PDFLATEX_COMPILER)

  configure_file(Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

  add_custom_target(doxygen 
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile 
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Generating C++ API documentation with Doxygen" VERBATIM
  )

  add_custom_target(install-doxygen
      COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${CMAKE_BINARY_DIR}/doxygen" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/doc/bob/doxygen"
      DEPENDS doxygen
    COMMENT "Installing C++ API documentation" VERBATIM
  )

endif()

if(SPHINX_FOUND)

  add_custom_target(sphinx
    COMMAND ${CMAKE_BINARY_DIR}/bin/python ${SPHINX_EXECUTABLE} -c ${CMAKE_SOURCE_DIR} -b html -E ${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR}/sphinx/html
    COMMENT "Generating (html) User Guide with Sphinx" VERBATIM
    )

  add_custom_target(sphinx-doctest
    COMMAND ${CMAKE_BINARY_DIR}/bin/python ${SPHINX_EXECUTABLE} -c ${CMAKE_SOURCE_DIR} -b doctest -E ${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR}/sphinx/html
    COMMENT "Running doctests with Sphinx" VERBATIM
    )

  if(PDFLATEX_COMPILER)
    add_custom_target(sphinx-latex 
      COMMAND ${CMAKE_BINARY_DIR}/bin/python ${SPHINX_EXECUTABLE} -c ${CMAKE_SOURCE_DIR} -b latex -E ${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR}/sphinx/latex 
      COMMAND make -C ${CMAKE_BINARY_DIR}/sphinx/latex
      COMMENT "Generating (LaTeX2e) User Guide with Sphinx + PDF" VERBATIM
      )
  elseif(LATEX_COMPILER)
    add_custom_target(sphinx-latex 
      COMMAND ${CMAKE_BINARY_DIR}/bin/python ${SPHINX_EXECUTABLE} -c ${CMAKE_SOURCE_DIR} -b latex -E ${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR}/sphinx/latex 
      COMMENT "Generating (LaTeX2e) User Guide with Sphinx" VERBATIM
      )
  endif()

  add_custom_target(sphinx-coverage
    COMMAND ${CMAKE_BINARY_DIR}/bin/python ${SPHINX_EXECUTABLE} -c ${CMAKE_SOURCE_DIR} -b coverage -E ${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR}/sphinx/coverage 
    COMMENT "Generating (coverage) report with Sphinx" VERBATIM
  )

  add_custom_target(install-sphinx
      COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${CMAKE_BINARY_DIR}/sphinx" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/doc/bob/sphinx"
      DEPENDS sphinx
    COMMENT "Installing (Sphinx) User Guide" VERBATIM
  )

endif()
