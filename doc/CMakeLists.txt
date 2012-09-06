include(FindDoxygen)

if(DOXYGEN_FOUND)

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
