project(bob_core)

# This defines the dependencies of this package
set(bob_deps "")
set(shared ${Boost_SYSTEM_LIBRARY_RELEASE};${Boost_FILESYSTEM_LIBRARY_RELEASE};${Boost_IOSTREAMS_LIBRARY_RELEASE};${Boost_THREAD_LIBRARY_RELEASE})

string(TOLOWER "${CMAKE_BUILD_TYPE}" LOWER_BUILD_TYPE)
# also add a dependency to Blitz in debug mode (needed to enable the -BZ_BLITZ debugging mode)
if("${LOWER_BUILD_TYPE}" STREQUAL "debug")
  set(shared ${shared} ${Blitz_LIBRARIES})
endif()

# This defines the list of source files inside this package.
set(src
    "Random.cc"
    "Exception.cc"
    "convert_exception.cc"
    "logging.cc"
    "array_exception.cc"
    "array_type.cc"
    "array.cc"
    "blitz_array.cc"
    "cast.cc"
    "reshape_exception.cc"
    "repmat_exception.cc"
    )

bob_add_library(${PROJECT_NAME} "${src}")
target_link_libraries(${PROJECT_NAME} ${shared})

# Defines tests for this package
bob_add_test(${PROJECT_NAME} logging test/logging.cc)
bob_add_test(${PROJECT_NAME} blitzarray test/blitzArray.cc)
bob_add_test(${PROJECT_NAME} convertarray test/convert.cc)
bob_add_test(${PROJECT_NAME} repmat test/repmat.cc)
bob_add_test(${PROJECT_NAME} reshape test/reshape.cc)
if((${CMAKE_SYSTEM_NAME} MATCHES "Darwin"))
  target_link_libraries(test_${PROJECT_NAME}_blitzarray "-framework CoreServices")
endif((${CMAKE_SYSTEM_NAME} MATCHES "Darwin"))

# Pkg-Config generator
bob_pkgconfig(${PROJECT_NAME} "${bob_deps}")
