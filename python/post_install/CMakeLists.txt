project(post_install)

if(IS_ABSOLUTE ${CMAKE_INSTALL_PREFIX})
  set(ABSOLUTE_install_prefix "${CMAKE_INSTALL_PREFIX}")
else()
  set(ABSOLUTE_install_prefix "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_PREFIX}")
endif()
install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} -m compileall \"${ABSOLUTE_install_prefix}/lib/python${PYTHON_VERSION}/bob\")")
