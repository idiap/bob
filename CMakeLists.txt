# Configures and builds bob
# Andre Anjos <andre.anjos@idiap.ch>
# Sat  1 Sep 20:36:15 2012 CEST

project(bob)
cmake_minimum_required(VERSION 2.6)

if(${CMAKE_MAJOR_VERSION} GREATER 2 AND ${CMAKE_MINOR_VERSION} GREATER 8 AND ${CMAKE_PATCH_VERSION} GREATER 5)
  include(GNUInstallDirs)
else()
  include(src/BobGNUInstallDirs.cmake)
endif()

# -----------------
# START of options.
# -----------------

if (NOT CMAKE_BUILD_TYPE)
  message(FATAL_ERROR "error: choose either Release or Debug as build type by specifying -DCMAKE_BUILD_TYPE=Release or -DCMAKE_BUILD_TYPE=Debug on your cmake command line.")
endif ()

set(BOB_SYSTEM_PROCESSOR ${CMAKE_SYSTEM_PROCESSOR})
set(BOB_SYSTEM_NAME ${CMAKE_SYSTEM_NAME})
if(APPLE)
  set(BOB_SYSTEM_PROCESSOR "x86_64") #forced, see __LP64__ comment bellow
  set(BOB_SYSTEM_NAME "macosx")
endif()

string(TOLOWER "${BOB_SYSTEM_NAME}-${BOB_SYSTEM_PROCESSOR}-${CMAKE_BUILD_TYPE}" BOB_PLATFORM_STR)
set(BOB_PLATFORM "${BOB_PLATFORM_STR}" CACHE STRING "The name of the platform Bob is being built to")

option(SET_PUBLIC_LIBRARY_PATH "Use `ld -rpath' (Linux) or `install_name_tool' (Apple) when linking public libraries, executables." ON)

# Please note there are two more options that can affect the build:
# BOB_VERSION
# BOB_SOVERSION
# They are used towards the end of the configuration. Look down for details.

# ---------------
# END of options.
# ---------------

# Flag setting

# For both C and C++
set(COMMON_FLAGS "-pedantic -Wall")
if (NOT CYGWIN)
  set(COMMON_FLAGS "${COMMON_FLAGS} -pthread")
endif (NOT CYGWIN)
if (NOT CMAKE_COMPILER_IS_GNUCC)
  # Then, it must be clang/clang++
  set(COMMON_FLAGS "${COMMON_FLAGS} -Qunused-arguments")
endif ()

# Force __LP64__ scheme on Mac OSX
if(APPLE)
  set(COMMON_FLAGS "${COMMON_FLAGS} -m64")
  # Bogus on OSX?
  set(CMAKE_INCLUDE_SYSTEM_FLAG_C "-isystem ")
  set(CMAKE_INCLUDE_SYSTEM_FLAG_CXX "-isystem ")
endif(APPLE)

# For both RELEASE and DEBUG builds
if(APPLE AND CMAKE_COMPILER_IS_GNUCC)
  set(COMMON_FLAGS "${COMMON_FLAGS} -Wno-long-long -Wno-variadic-macros")
elseif(CYGWIN)
  set(COMMON_CXX_FLAGS "-std=gnu++0x")
  set(COMMON_C_FLAGS "-std=gnu99")
else()
  set(COMMON_CXX_FLAGS "-std=c++0x")
  set(COMMON_C_FLAGS "-std=c99")
endif()

# For specific builds
set(COMMON_RELEASE_FLAGS "-O2 -mtune=generic -DNDEBUG")

# Note: CLang does not work well with BZ_DEBUG
set(COMMON_DEBUG_FLAGS "-g -DBOB_DEBUG")

if(NOT APPLE OR CMAKE_COMPILER_IS_GNUCC) #linux builds or apple+gcc4.2
  set(COMMON_DEBUG_FLAGS "${COMMON_DEBUG_FLAGS} -DBZ_DEBUG")
endif()

# These are used in type checks for cmake, be aware and don't change those
set(CMAKE_CXX_FLAGS "${COMMON_CXX_FLAGS} ${COMMON_FLAGS}" CACHE STRING "Flags used by the compiler during release builds" FORCE)
set(CMAKE_C_FLAGS "${COMMON_C_FLAGS} ${COMMON_FLAGS}" CACHE STRING "Flags used by the compiler during release builds" FORCE)

# Cache into CMake
set(CMAKE_CXX_FLAGS_RELEASE "${COMMON_RELEASE_FLAGS}" CACHE STRING "Flags used by the C++ compiler during release builds" FORCE)
set(CMAKE_C_FLAGS_RELEASE "${COMMON_RELEASE_FLAGS}" CACHE STRING "Flags used by the C compiler during release builds" FORCE)
set(CMAKE_CXX_FLAGS_DEBUG "${COMMON_DEBUG_FLAGS}" CACHE STRING "Flags used by the C++ compiler during debug builds" FORCE)
set(CMAKE_C_FLAGS_DEBUG "${COMMON_DEBUG_FLAGS}" CACHE STRING "Flags used by the compiler C during debug builds" FORCE)

if(SET_PUBLIC_LIBRARY_PATH)
  if (APPLE)
    set(CMAKE_INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/lib")
  else()
    # This is our RPATH policy under Linux
    set(CMAKE_SKIP_BUILD_RPATH  FALSE)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  endif()
endif()

# This will make sure we consider pkg-config directories that may have been
# added when the user set CMAKE_PREFIX_PATH.
if(CMAKE_PREFIX_PATH)
  string(REPLACE ":" ";" PKG_CONFIG_CHECK_PATH ${CMAKE_PREFIX_PATH})
  foreach (ext ${PKG_CONFIG_CHECK_PATH})
    if(EXISTS ${ext}/${CMAKE_INSTALL_LIBDIR}/pkgconfig)
      set(ENV{PKG_CONFIG_PATH} "${ext}/${CMAKE_INSTALL_LIBDIR}/pkgconfig:$ENV{PKG_CONFIG_PATH}")
    endif()
  endforeach()
endif()

# Enables the testing framework
enable_testing()

set(BUILD_SHARED_LIBS "ON" CACHE BOOL "Build shared libs")

# Install libraries and executables
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
if(CYGWIN)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
else()
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})
endif()

# Include file handling and installation
install(DIRECTORY include/bob DESTINATION include FILES_MATCHING PATTERN "*.h")

# ----------------------------------------------------------------------------
# Some macros
# ----------------------------------------------------------------------------

# Adds a C++ Bob library.
#
# package: name of the c++ package
# src: list of c++ files to compile
#
# Example: bob_add_library(bob_core "foo.cc;bar.cc")
macro(bob_add_library name src)
  include_directories(BEFORE 
    "${CMAKE_SOURCE_DIR}/include" "${CMAKE_BINARY_DIR}/include")
  add_library(${name} ${src})
  if (BOB_SOVERSION)
    # adds versioning information
    set_target_properties(${name} PROPERTIES VERSION ${BOB_VERSION})
    set_target_properties(${name} PROPERTIES SOVERSION ${BOB_SOVERSION})
  endif()
  #message("CMAKE_INSTALL_LIBDIR = " ${CMAKE_INSTALL_LIBDIR})
  install(TARGETS ${name} EXPORT bob-targets DESTINATION ${CMAKE_INSTALL_LIBDIR})
endmacro()

# Creates a standard test for Bob
#
# package: subpackage where the test is sitting
# name: test name
# src: test source files
#
# Example: bob_add_test(bob_io array test/array.cc)
macro(bob_add_test package name src)
  set(bin_name test_${package}_${name})
  string(REPLACE "test_bob_" "" test_name ${bin_name})
  include_directories(BEFORE 
    "${CMAKE_SOURCE_DIR}/include" "${CMAKE_BINARY_DIR}/include")
  add_executable(${bin_name} ${src})
  target_link_libraries(${bin_name} ${package};${Boost_UNIT_TEST_FRAMEWORK_LIBRARY_RELEASE})
  add_test(${test_name} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${bin_name} --log_level=test_suite)
  string(REPLACE "bob_" "" short_package_name ${package})
  set_property(TEST ${test_name} APPEND PROPERTY ENVIRONMENT "BOB_TESTDATA_DIR=${CMAKE_SOURCE_DIR}/testdata/${short_package_name}")
endmacro()

# Creates a standard Bob binary application.
#
# package: package the test belongs to
# name: application name
# src: test source files
#
# Example: bob_add_executable(io array test/array.cc)
macro(bob_add_executable package name src)
  set(ename ${package}_${name})
  include_directories(BEFORE 
    "${CMAKE_SOURCE_DIR}/include" "${CMAKE_BINARY_DIR}/include")
  add_executable(${ename} ${src})
  target_link_libraries(${ename} ${package})
  install(TARGETS ${ename} DESTINATION bin)
endmacro()

# Creates and installs a pkg-config file for each subpackage
#
# package: package the test belongs to
# requires: a list of other bob packages this package requires
#
# Example: bob_pkgconfig(bob_ip "bob_core;bob_sp;bob_math")
macro(bob_pkgconfig package requires)
  set(pkgconfigdir "${CMAKE_INSTALL_LIBDIR}/pkgconfig")
  string(REPLACE "_" "-" BOB_PACKAGE_NAME ${package})
  set(BOB_PACKAGE_LIBRARY ${package})
  
  get_directory_property(my_defines COMPILE_DEFINITIONS)
  list(REMOVE_DUPLICATES my_defines)
  set(BOB_PACKAGE_DEFINES "")
  foreach(def ${my_defines})
    set(BOB_PACKAGE_DEFINES "${BOB_PACKAGE_DEFINES} -D${def}")
  endforeach()

  get_directory_property(my_includes INCLUDE_DIRECTORIES)
  list(REMOVE_ITEM my_includes "${CMAKE_SOURCE_DIR}/include" 
    "${CMAKE_BINARY_DIR}/include")
  list(REMOVE_DUPLICATES my_includes)
  set(BOB_PACKAGE_INCLUDES "")
  foreach(inc ${my_includes})
    set(BOB_PACKAGE_INCLUDES "${BOB_PACKAGE_INCLUDES} -I${inc}")
  endforeach()

  set(BOB_PACKAGE_REQUIRES "")
  foreach(req ${requires})
    string(REPLACE "_" "-" myreq ${req})
    set(BOB_PACKAGE_REQUIRES "${BOB_PACKAGE_REQUIRES} ${myreq} = ${BOB_VERSION}")
  endforeach()

  # pkgconfig files for the build area
  set(BOB_CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR})
  configure_file(${CMAKE_SOURCE_DIR}/pkgconfig/bob-package.pc.in ${CMAKE_BINARY_DIR}/${pkgconfigdir}/${BOB_PACKAGE_NAME}.pc @ONLY)

  # pkgconfig files for the install area
  set(BOB_CMAKE_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
  configure_file(${CMAKE_SOURCE_DIR}/pkgconfig/bob-package.pc.in ${CMAKE_BINARY_DIR}/to_install/${pkgconfigdir}/${BOB_PACKAGE_NAME}.pc @ONLY)
  install(FILES ${CMAKE_BINARY_DIR}/to_install/${pkgconfigdir}/${BOB_PACKAGE_NAME}.pc DESTINATION ${pkgconfigdir})

  # Adds this package to the pool
  list(APPEND ENABLED_PACKAGES ${package})
  set(ENABLED_PACKAGES "${ENABLED_PACKAGES}" PARENT_SCOPE)
endmacro()

# Sorts hears in order of priority:
#
# list_name: The name of the list you want to sort
#
# Example: bob_sort_headers(incdir) # incdir must be a list
macro(bob_sort_headers list_name)
  execute_process(COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/bin/sort_headers.py "${CMAKE_CXX_COMPILER}" "${CMAKE_PREFIX_PATH}" "${${list_name}}" OUTPUT_VARIABLE sorted_list OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(${list_name} ${sorted_list})
endmacro()

# ----------------------------------------------------------------------------
# End of macros
# ----------------------------------------------------------------------------

# Project files
set(ENABLED_PACKAGES "")
add_subdirectory(src)
add_subdirectory(python)
add_subdirectory(doc)

# Calculation of Bob versions (and .so versions)
execute_process(COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/bin/version.py --letter=a --counter=0 OUTPUT_VARIABLE BOB_DISCOVERED_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
set(BOB_VERSION_INTERNAL "${BOB_DISCOVERED_VERSION}" CACHE INTERNAL "The version of Bob that is currently being built")
if(NOT BOB_VERSION)
  set(BOB_VERSION ${BOB_VERSION_INTERNAL})
endif()

set(BOB_SOVERSION_INTERNAL "OFF" CACHE INTERNAL "If we need to use a SOVERSION
on libraries, this should set it")
if(NOT BOB_SOVERSION)
  set(BOB_SOVERSION ${BOB_SOVERSION_INTERNAL})
endif()

message(STATUS "Bob version '${BOB_VERSION}' (${BOB_PLATFORM_STR})")

# Produce the config.h file
configure_file(${CMAKE_SOURCE_DIR}/include/bob/config.h.in ${CMAKE_BINARY_DIR}/include/bob/config.h @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/include/bob/config.h DESTINATION include/bob)

# Install cmake export target
set(cmakedir "${CMAKE_INSTALL_LIBDIR}/bob")
install(EXPORT bob-targets DESTINATION ${cmakedir})
install(FILES cmake/bob-config.cmake DESTINATION ${cmakedir})
configure_file(cmake/bob-config-version.cmake.in
  ${CMAKE_BINARY_DIR}/${cmakedir}/bob-config-version.cmake @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/${cmakedir}/bob-config-version.cmake
  DESTINATION ${cmakedir})

# Global pkg-config file
set(pkgconfigdir "${CMAKE_INSTALL_LIBDIR}/pkgconfig")
set(BOB_PACKAGES "")
foreach(pack ${ENABLED_PACKAGES})
  string(REPLACE "_" "-" pkg_name ${pack})
  set(BOB_PACKAGES "${BOB_PACKAGES} ${pkg_name} = ${BOB_VERSION}")
endforeach()
# For the build area
set(BOB_CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR})
configure_file(${CMAKE_SOURCE_DIR}/pkgconfig/bob.pc.in ${CMAKE_BINARY_DIR}/${pkgconfigdir}/bob.pc @ONLY)
# For the installation area
set(BOB_CMAKE_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
configure_file(${CMAKE_SOURCE_DIR}/pkgconfig/bob.pc.in ${CMAKE_BINARY_DIR}/to_install/${pkgconfigdir}/bob.pc @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/to_install/${pkgconfigdir}/bob.pc DESTINATION ${pkgconfigdir})
