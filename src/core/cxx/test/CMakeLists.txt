project(core-test)

# Defines tests for this package
bob_add_test(core logging test/logging.cc)
bob_add_test(core blitzarray test/blitzArray.cc)
bob_add_test(core convertarray test/convert.cc)
bob_add_test(core repmat test/repmat.cc)
bob_add_test(core reshape test/reshape.cc)
if((${CMAKE_SYSTEM_NAME} MATCHES "Darwin"))
  target_link_libraries(bobtest_core_blitzarray "-framework CoreServices")
endif((${CMAKE_SYSTEM_NAME} MATCHES "Darwin"))
