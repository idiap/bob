# Andre Anjos <andre.anjos@idiap.ch>
# 4/August/2010

# These are a few handy macros for Torch compilation.

# Builds and installs a shared library with dependencies
macro(torch_shlib libname sources dependencies externals installdir)
  add_library(${libname} SHARED ${sources})
  if(NOT ("${dependencies}" STREQUAL ""))
    foreach(dep ${dependencies})
      target_link_libraries(${libname} torch_${dep})
    endforeach(dep ${dependencies})
  endif(NOT ("${dependencies}" STREQUAL ""))
  if(NOT ("${externals}" STREQUAL ""))
    foreach(ext ${externals})
      target_link_libraries(${libname} ${ext})
    endforeach(ext ${externals})
  endif(NOT ("${externals}" STREQUAL ""))
  install(TARGETS ${libname} EXPORT torch LIBRARY DESTINATION ${installdir})
endmacro(torch_shlib libname sources dependencies)

# Builds and installs a shared library with dependencies
macro(torch_archive libname sources dependencies installdir)
  add_library(${libname}-static STATIC ${sources})
  if(NOT ("${dependencies}" STREQUAL ""))
    foreach(dep ${dependencies})
      target_link_libraries(${libname}-static torch_${dep}-static)
    endforeach(dep ${dependencies})
  endif(NOT ("${dependencies}" STREQUAL ""))
  set_target_properties(${libname}-static PROPERTIES OUTPUT_NAME ${libname})
  set_target_properties(${libname}-static PROPERTIES PREFIX "lib")
  install(TARGETS ${libname}-static EXPORT torch ARCHIVE DESTINATION ${installdir})
endmacro(torch_archive sources dependencies)

# Builds libraries for a subproject and installs headers. Wraps every of those
# items in an exported CMake module to be used by other libraries in or outside
# the project.
# 
# The parameters:
# torch_library -- This macro's name
# package -- The base name of this package, so everything besides "torch_",
# which will get automatically prefixed
# src -- The sources for the libraries generated
# deps -- This is a list of other subprojects that this project depends on.
# shared -- This is a list of shared libraries to which shared libraries
# generated on this project must link against.
macro(torch_library package src deps shared)
  # We set this so we don't need to become repetitive.
  set(libname torch_${package})
  set(libdir lib)
  set(incdir include/torch)

  include_directories(${CMAKE_CURRENT_SOURCE_DIR})

  # This adds target (library) torch_<package>, exports into "torch"
  torch_shlib(${libname} "${src}" "${deps}" "${shared}" ${libdir})

  # This adds target (library) torch_<package>-static, exports into "torch"
  torch_archive(${libname} "${src}" "${deps}" ${libdir})

  # This installs all headers to the destination directory
  add_custom_command(TARGET ${libname} POST_BUILD COMMAND mkdir -p ${CMAKE_INSTALL_PREFIX}/${incdir} COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR}/${package} ${CMAKE_INSTALL_PREFIX}/${incdir} COMMENT "Installing ${package} headers...")
endmacro(torch_library)

# Creates a standard Torch test.
macro(torch_test package name src)
  set(testname torchtest_${package}_${name})

  # Include the Boost Unit Test Framework
  include_directories(${Boost_INCLUDE_DIRS})
  link_directories(${Boost_LIBRARY_DIRS})

  # Please note we don't install test executables
  add_executable(${testname} ${src})
  target_link_libraries(${testname} torch_${package})
  target_link_libraries(${testname} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
  add_test(cxx-${package}-${name} ${testname})
endmacro(torch_test package src)

# Creates a standard Torch benchmark.
macro(torch_benchmark package name src)
  set(bindir bin)
  set(progname torchbench_${package}_${name})

  # Include the Boost Unit Test Framework
  include_directories(${Boost_INCLUDE_DIRS})

  add_executable(${progname} ${src})
  target_link_libraries(${progname} torch_${package})
  install(TARGETS ${progname} RUNTIME DESTINATION ${bindir})
endmacro(torch_benchmark package name src)

# Creates Torch pythonic bindings to C/C++ code using Boost::Python
macro(torch_python_bindings package src)
  # Some preparatory work
  set(libname pytorch_${package})
  set(libdir lib)

  # Some necessary compilation and linkage includes
  include_directories(${Boost_INCLUDE_DIRS})
  include_directories(${PYTHON_INCLUDE_DIRS})
  link_directories(${Boost_LIBRARY_DIRS})
  link_directories(${PYTHON_LIBRARY_DIRS})

  # Building the library itself
  add_library(${libname} SHARED ${src})
  set_target_properties(${libname} PROPERTIES SUFFIX ".so")
  target_link_libraries(${libname} torch_${package})
  target_link_libraries(${libname} ${Boost_PYTHON_LIBRARY})
  target_link_libraries(${libname} ${PYTHON_LIBRARIES})

  # And its installation
  install(TARGETS ${libname} LIBRARY DESTINATION ${libdir})
endmacro(torch_python_bindings package name src)

# Installs python files and compile them
macro(torch_python_install package)
  set(pydir ${CMAKE_INSTALL_PREFIX}/lib/python${PYTHON_VERSION}/torch)
  file(COPY python/${package} DESTINATION ${pydir}
    FILES_MATCHING PATTERN "*.py")
endmacro(torch_python_install package)
