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

  include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR})

  # This adds target (library) torch_<package>, exports into "torch"
  torch_shlib(${libname} "${src}" "${deps}" "${shared}" ${libdir})

  if ("${TORCH_BUILD_STATIC_LIBS}")
    # This adds target (library) torch_<package>-static, exports into "torch"
    torch_archive(${libname} "${src}" "${deps}" ${libdir})
  endif ("${TORCH_BUILD_STATIC_LIBS}")

  # This installs all headers to the destination directory
  add_custom_command(TARGET ${libname} POST_BUILD COMMAND mkdir -p ${CMAKE_INSTALL_PREFIX}/${incdir} COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR}/${package} ${CMAKE_INSTALL_PREFIX}/${incdir} COMMENT "Installing ${package} headers...")
endmacro(torch_library)

# Creates a standard Torch test.
macro(torch_test package name src)
  set(testname torchtest_${package}_${name})

  # Please note we don't install test executables
  add_executable(${testname} ${src})
  target_link_libraries(${testname} torch_${package})
  target_link_libraries(${testname} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
  add_test(cxx-${package}-${name} ${testname} --log_level=test_suite)
endmacro(torch_test package src)

# Creates a standard Torch benchmark.
macro(torch_benchmark package name src)
  set(bindir bin)
  set(progname torchbench_${package}_${name})

  add_executable(${progname} ${src})
  target_link_libraries(${progname} torch_${package})
  install(TARGETS ${progname} RUNTIME DESTINATION ${bindir})
endmacro(torch_benchmark package name src)

macro(torch_python_bindings package src)
  if (PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND AND Boost_FOUND)
    # Some preparatory work
    set(libname pytorch_${package})
    set(libdir lib)
    include_directories(${python_INCLUDE_DIRS};${CMAKE_CURRENT_SOURCE_DIR}/python/src)

    # Building the library itself
    add_library(${libname} SHARED ${src})
    set_target_properties(${libname} PROPERTIES SUFFIX ".so")
    target_link_libraries(${libname} torch_${package})
    target_link_libraries(${libname} ${Boost_PYTHON_LIBRARY})
    target_link_libraries(${libname} ${PYTHON_LIBRARIES})

    # And its installation
    install(TARGETS ${libname} LIBRARY DESTINATION ${libdir})
  else (PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND AND Boost_FOUND)
    message("Boost::Python bindings for ${package} are DISABLED: externals NOT FOUND!")
  endif (PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND AND Boost_FOUND)
endmacro(torch_python_bindings package name src)

macro(torch_python_submodule package subpackage src)
  if (PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND AND Boost_FOUND)
    # Some preparatory work
    set(libname pytorch_${package}_${subpackage})
    set(libdir lib)
    include_directories(${python_INCLUDE_DIRS};${CMAKE_CURRENT_SOURCE_DIR}/python/src)

    # Building the library itself
    add_library(${libname} SHARED ${src})
    set_target_properties(${libname} PROPERTIES SUFFIX ".so")
    target_link_libraries(${libname} torch_${package})
    target_link_libraries(${libname} ${Boost_PYTHON_LIBRARY})
    target_link_libraries(${libname} ${PYTHON_LIBRARIES})

    # And its installation
    install(TARGETS ${libname} LIBRARY DESTINATION ${libdir})
  else (PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND AND Boost_FOUND)
    message("Boost::Python bindings for ${package}_${subpackage} are DISABLED: externals NOT FOUND!")
  endif (PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND AND Boost_FOUND)
endmacro(torch_python_submodule package name src)

# Installs python files and compile them
macro(torch_python_install package)
  if (PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND AND Boost_FOUND)
    set(pydir ${CMAKE_INSTALL_PREFIX}/lib/python${PYTHON_VERSION}/torch)
    add_custom_target(${package}-python-install cp -r ${CMAKE_CURRENT_SOURCE_DIR}/python/${package} ${pydir} COMMENT "Installing ${package} python files...")
    add_dependencies(${package}-python-install pytorch_${package})
    add_dependencies(${package}-python-install torch-python-install)
    add_dependencies(python-compilation ${package}-python-install)
  endif (PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND AND Boost_FOUND)
endmacro(torch_python_install package)

# This macro helps users to add python tests to cmake
function(torch_python_add_test)
  add_test(${ARGV})
  set_property(TEST ${ARGV0} APPEND PROPERTY ENVIRONMENT
    "DYLD_LIBRARY_PATH=${CMAKE_INSTALL_PREFIX}/lib:$ENV{DYLD_LIBRARY_PATH};LD_LIBRARY_PATH=${CMAKE_INSTALL_PREFIX}/lib:$ENV{LD_LIBRARY_PATH};PYTHONPATH=${CMAKE_INSTALL_PREFIX}/lib:${PYTHON_INSTALL_DIRECTORY}:$ENV{PYTHONPATH}")
endfunction(torch_python_add_test)
