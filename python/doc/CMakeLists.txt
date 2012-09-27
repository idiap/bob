find_program(SPHINX_EXECUTABLE NAMES sphinx-build-${PYTHON_VERSION} sphinx-build
  HINTS $ENV{SPHINX_DIR} PATH_SUFFIXES bin
  DOC "Sphinx documentation generator"
  )

if(SPHINX_EXECUTABLE)

  add_custom_target(sphinx
    COMMAND ${CMAKE_BINARY_DIR}/bin/python ${SPHINX_EXECUTABLE} -b html -E ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_BINARY_DIR}/sphinx/html
    COMMENT "Generating (html) User Guide with Sphinx" VERBATIM
  )
  add_dependencies(sphinx egg-scripts)

  add_custom_target(sphinx-doctest
    COMMAND ${CMAKE_BINARY_DIR}/bin/python ${SPHINX_EXECUTABLE} -b doctest -E ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_BINARY_DIR}/sphinx/html
    COMMENT "Running doctests with Sphinx" VERBATIM
  )
  add_dependencies(sphinx-doctest egg-scripts)

  include(FindLATEX)

  if(PDFLATEX_COMPILER)
    add_custom_target(sphinx-latex 
      COMMAND ${CMAKE_BINARY_DIR}/bin/python ${SPHINX_EXECUTABLE} -b latex -E ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_BINARY_DIR}/sphinx/latex 
      COMMAND make -C ${CMAKE_BINARY_DIR}/sphinx/latex
      COMMENT "Generating (LaTeX2e) User Guide with Sphinx + PDF" VERBATIM
    )
    add_dependencies(sphinx-latex egg-scripts)
  elseif(LATEX_COMPILER)
    add_custom_target(sphinx-latex 
      COMMAND ${CMAKE_BINARY_DIR}/bin/python ${SPHINX_EXECUTABLE} -b latex -E ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_BINARY_DIR}/sphinx/latex 
      COMMENT "Generating (LaTeX2e) User Guide with Sphinx" VERBATIM
    )
    add_dependencies(sphinx-latex egg-scripts)
  else()
    message(WARNING "Sphinx PDF output requires either pdflatex or latex and I cannot find neither so 'make sphinx-latex' will only dump LaTeX sources")
  endif()

  add_custom_target(sphinx-coverage 
    COMMAND ${CMAKE_BINARY_DIR}/bin/python ${SPHINX_EXECUTABLE} -b coverage -E ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_BINARY_DIR}/sphinx/coverage 
    COMMENT "Generating (coverage) report with Sphinx" VERBATIM
  )
  add_dependencies(sphinx-coverage egg-scripts)

  add_custom_target(install-sphinx
      COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${CMAKE_BINARY_DIR}/sphinx" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/doc/bob/sphinx"
    COMMENT "Installing (Sphinx) User Guide" VERBATIM
  )
  add_dependencies(install-sphinx sphinx-scripts)

endif()
