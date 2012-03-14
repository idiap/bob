project(core-cxx)
cmake_minimum_required(VERSION 2.6)

# This defines the dependencies of this package
set(deps "") #other bob subprojects
set(shared ${Boost_SYSTEM_LIBRARY_RELEASE} ${Boost_THREAD_LIBRARY_RELEASE}
           ${Boost_FILESYSTEM_LIBRARY_RELEASE}
           ${Boost_IOSTREAMS_LIBRARY_RELEASE} ${Boost_DATE_TIME_LIBRARY_RELEASE}
           ${Boost_PROGRAM_OPTIONS_LIBRARY_RELEASE})

string(TOLOWER "${CMAKE_BUILD_TYPE}" LOWER_BUILD_TYPE)
# also add a dependency to Blitz in debug mode (needed to enable the -BZ_BLITZ debugging mode)
if("${LOWER_BUILD_TYPE}" STREQUAL "debug")
  set(shared ${shared} ${Blitz_LIBRARIES})
endif()

# If we have google-perftools installed, enable the HAS_GOOGLE_PERFTOOLS flag,
# so profiling extensions can be compiled in.
if(GOOGLE_PERFTOOLS_FOUND)
  set(shared ${shared} ${GOOGLE_PERFTOOLS_LIBRARIES})
endif(GOOGLE_PERFTOOLS_FOUND)

# This defines the list of source files inside this package.
set(src
    "src/Random.cc"
    "src/Exception.cc"
    "src/convert_exception.cc"
    "src/logging.cc"
    "src/array_exception.cc"
    "src/array_type.cc"
    "src/array.cc"
    "src/blitz_array.cc"
    "src/cast.cc"
    "src/reshape_exception.cc"
    "src/repmat_exception.cc"
    )

bob_library(core "${src}" "${deps}" "${shared}")

# Defines tests for this package
bob_test(core logging test/logging.cc)
bob_test(core blitzarray test/blitzArray.cc)
bob_test(core convertarray test/convert.cc)
bob_test(core repmat test/repmat.cc)
bob_test(core reshape test/reshape.cc)
if((${CMAKE_SYSTEM_NAME} MATCHES "Darwin"))
  target_link_libraries(bobtest_core_blitzarray "-framework CoreServices")
endif((${CMAKE_SYSTEM_NAME} MATCHES "Darwin"))

# Defines benchmarks we make available
bob_benchmark(core bigtensor2d "benchmark/bigtensor2d.cc")
bob_example_install("cxx" "core" "benchmark/bigtensor2d.cc")
