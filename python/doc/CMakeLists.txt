find_program(SPHINX_EXECUTABLE NAMES sphinx-build-${PYTHON_VERSION} sphinx-build
  HINTS $ENV{SPHINX_DIR} PATH_SUFFIXES bin
  DOC "Sphinx documentation generator"
  )

if(SPHINX_EXECUTABLE)

  add_custom_target(sphinx
    COMMAND ${CMAKE_BINARY_DIR}/bin/python ${SPHINX_EXECUTABLE} -b html -E ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_BINARY_DIR}/sphinx/html
    COMMENT "Generating (html) User Guide with Sphinx" VERBATIM
    DEPENDS egg
  )

  add_custom_target(sphinx-doctest
    COMMAND ${CMAKE_BINARY_DIR}/bin/python ${SPHINX_EXECUTABLE} -b doctest -E ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_BINARY_DIR}/sphinx/html
    COMMENT "Running doctests with Sphinx" VERBATIM
    DEPENDS egg
  )

  if(PDFLATEX_COMPILER)
    add_custom_target(sphinx-latex 
      COMMAND ${CMAKE_BINARY_DIR}/bin/python ${SPHINX_EXECUTABLE} -b latex -E ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_BINARY_DIR}/sphinx/latex 
      COMMAND make -C ${CMAKE_BINARY_DIR}/sphinx/latex
      COMMENT "Generating (LaTeX2e) User Guide with Sphinx + PDF" VERBATIM
      DEPENDS egg
    )
  else()
    add_custom_target(sphinx-latex 
      COMMAND ${CMAKE_BINARY_DIR}/bin/python ${SPHINX_EXECUTABLE} -b latex -E ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_BINARY_DIR}/sphinx/latex 
      COMMENT "Generating (LaTeX2e) User Guide with Sphinx" VERBATIM
      DEPENDS egg
    )
  endif()

  add_custom_target(sphinx-coverage 
    COMMAND ${CMAKE_BINARY_DIR}/bin/python ${SPHINX_EXECUTABLE} -b coverage -E ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_BINARY_DIR}/sphinx/coverage 
    COMMENT "Generating (coverage) report with Sphinx" VERBATIM
    DEPENDS egg
  )

  add_custom_target(install-sphinx
      COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${CMAKE_BINARY_DIR}/sphinx" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/doc/bob/sphinx"
      DEPENDS sphinx
    COMMENT "Installing (Sphinx) User Guide" VERBATIM
  )

endif()
