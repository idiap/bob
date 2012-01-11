# Andre Anjos <andre.anjos@idiap.ch>
# 4/August/2010

# These are a few handy macros for bob compilation.

# Builds and installs a shared library with dependencies
macro(bob_shlib libname sources dependencies externals installdir)
  add_library(${libname} SHARED ${sources})
  if(NOT ("${dependencies}" STREQUAL ""))
    foreach(dep ${dependencies})
      target_link_libraries(${libname} bob_${dep})
    endforeach(dep ${dependencies})
  endif(NOT ("${dependencies}" STREQUAL ""))
  if(NOT ("${externals}" STREQUAL ""))
    foreach(ext ${externals})
      target_link_libraries(${libname} ${ext})
    endforeach(ext ${externals})
  endif(NOT ("${externals}" STREQUAL ""))
  install(TARGETS ${libname} EXPORT bob LIBRARY DESTINATION ${installdir})
endmacro(bob_shlib libname sources dependencies)

# Builds and installs a shared library with dependencies
macro(bob_archive libname sources dependencies installdir)
  add_library(${libname}-static STATIC ${sources})
  if(NOT ("${dependencies}" STREQUAL ""))
    foreach(dep ${dependencies})
      target_link_libraries(${libname}-static bob_${dep}-static)
    endforeach(dep ${dependencies})
  endif(NOT ("${dependencies}" STREQUAL ""))
  set_target_properties(${libname}-static PROPERTIES OUTPUT_NAME ${libname})
  set_target_properties(${libname}-static PROPERTIES PREFIX "lib")
  install(TARGETS ${libname}-static EXPORT bob ARCHIVE DESTINATION ${installdir})
endmacro(bob_archive sources dependencies)

# Installs header files to the default include directory
#
# The parameters:
# package -- The base name of this package, so everything besides "bob_"
# comment -- This will be used to print a nice comment when installing the
#            header. Something like "bob::core" will look nice
macro(bob_header_install target_name package comment)
  set(incdir include/bob)
  add_custom_target(${target_name}_header_install ALL COMMAND mkdir -p ${CMAKE_INSTALL_PREFIX}/${incdir} COMMAND rsync -a ${CMAKE_CURRENT_SOURCE_DIR}/${package}/ ${CMAKE_INSTALL_PREFIX}/${incdir}/${package} COMMENT "Installing ${comment} headers...")
endmacro(bob_header_install package comment)

# Builds libraries for a subproject. Wraps every of those items in an exported
# CMake module to be used by other libraries in or outside the project.
# 
# The parameters:
# bob_library -- This macro's name
# package -- The base name of this package, so everything besides "bob_",
# which will get automatically prefixed
# src -- The sources for the libraries generated
# deps -- This is a list of other subprojects that this project depends on.
# shared -- This is a list of shared libraries to which shared libraries
# generated on this project must link against.
macro(bob_library package src deps shared)
  # We set this so we don't need to become repetitive.
  set(libname bob_${package})
  set(libdir lib)

  include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR})

  # This adds target (library) bob_<package>, exports into "bob"
  bob_shlib(${libname} "${src}" "${deps}" "${shared}" ${libdir})

  if ("${BOB_BUILD_STATIC_LIBS}")
    # This adds target (library) bob_<package>-static, exports into "bob"
    bob_archive(${libname} "${src}" "${deps}" ${libdir})
  endif ("${BOB_BUILD_STATIC_LIBS}")
endmacro(bob_library)

# Creates a standard bob test.
macro(bob_test package name src)
  set(testname bobtest_${package}_${name})

  # Please note we don't install test executables
  add_executable(${testname} ${src})
  add_dependencies(${testname} bob_${package})
  target_link_libraries(${testname} bob_${package})
  target_link_libraries(${testname} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY_RELEASE})
  add_test(cxx-${package}-${name} ${testname} --log_level=test_suite)
endmacro(bob_test package src)

# Creates a standard bob benchmark.
macro(bob_benchmark package name src)
  set(bindir bin)
  set(progname bobbench_${package}_${name})

  add_executable(${progname} ${src})
  target_link_libraries(${progname} bob_${package})
  install(TARGETS ${progname} RUNTIME DESTINATION ${bindir})
endmacro(bob_benchmark package name src)

# Adds a include directory to a target
macro(target_include_directories target directories)
  foreach(dir ${directories})
    get_target_property(save ${target} COMPILE_FLAGS)
    if(NOT save) 
      set(save "")
    endif(NOT save) 
    set_target_properties(${target} PROPERTIES COMPILE_FLAGS "${save} -I${dir}")
  endforeach(dir ${directories})
endmacro(target_include_directories)

macro(target_include_system_directories target directories)
  foreach(dir ${directories})
    get_target_property(save ${target} COMPILE_FLAGS)
    if(NOT save) 
      set(save "")
    endif(NOT save) 
    set_target_properties(${target} PROPERTIES COMPILE_FLAGS "${save} -isystem ${dir}")
  endforeach(dir ${directories})
endmacro(target_include_system_directories)

# Builds a top-level python module. Arguments are:
# bname -- the bindings core name "pybob_${bname}" will be the resulting lib
# src -- a list of sources to compile into the package
# dependencies -- a list of internal bob package dependencies
# externals -- a list of external bob package dependencies as libraries to
#              link the resulting bindings to. Please note you don't need to
#              specify either linkage to libpython* or libboost_python as this
#              is done by default.
# pch (optional) -- a list of C++ header files to precompile
function(bob_python_bindings bname src dependencies externals)
  if (PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND AND Boost_FOUND)
    # Some preparatory work
    set(libname pybob_${bname})
    set(libdir lib)

    # Our compilation flags
    set(pycxx_flags "-Wno-long-long -Wno-unused-function -Winvalid-pch")

    include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR})

    # Building the library itself
    add_library(${libname} SHARED ${src})
    set_target_properties(${libname} PROPERTIES SUFFIX ".so")
    set_target_properties(${libname} PROPERTIES COMPILE_FLAGS ${pycxx_flags})
    target_include_system_directories(${libname} "${python_INCLUDE_DIRS}")
    if(ARGN) #pch support
      target_include_directories(${libname} ${CMAKE_CURRENT_BINARY_DIR}) 
    endif(ARGN)

    # Links dependencies
    if(NOT ("${dependencies}" STREQUAL ""))
      foreach(dep ${dependencies})
        target_link_libraries(${libname} bob_${dep})
      endforeach(dep ${dependencies})
    endif(NOT ("${dependencies}" STREQUAL ""))
    if(NOT ("${externals}" STREQUAL ""))
      foreach(ext ${externals})
        target_link_libraries(${libname} ${ext})
      endforeach(ext ${externals})
    endif(NOT ("${externals}" STREQUAL ""))

    target_link_libraries(${libname} ${Boost_PYTHON_LIBRARY_RELEASE})
    target_link_libraries(${libname} ${PYTHON_LIBRARIES})

    # And its installation
    install(TARGETS ${libname} LIBRARY DESTINATION ${libdir})
  else (PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND AND Boost_FOUND)
    message("Boost::Python bindings for ${bname} are DISABLED: externals NOT FOUND!")
  endif (PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND AND Boost_FOUND)
endfunction(bob_python_bindings bname src dependencies externals)

# Builds a top-level python module for a specific package. Arguments are:
# package -- the package name
# src -- a list of sources to compile into the package
# pch (optional) -- a list of C++ header files to precompile
function(bob_python_package_bindings package src)
  if(ARGN) #pch support
    bob_python_bindings(${package} "${src}" "${package}" "" "${ARGN}")
  else(ARGN) #pch support
    bob_python_bindings(${package} "${src}" "${package}" "")
  endif(ARGN) #pch support
endfunction(bob_python_package_bindings package src)

# Builds a python module that will be nested in a package, as a submodule. 
# Arguments are:
# package -- the package name
# subpackage -- the name of the subpackage
# src -- a list of sources to compile into the package
# pch (optional) -- a list of C++ header files to precompile
function(bob_python_submodule_bindings package subpackage src)
  if(ARGN) #pch support
    bob_python_bindings("${package}_${subpackage}" "${src}" "${package}" "" "${ARGN}")
  else(ARGN) #pch support
    bob_python_bindings("${package}_${subpackage}" "${src}" "${package}" "")
  endif(ARGN) #pch support
endfunction(bob_python_submodule_bindings package name src)

# Installs python files and compile them
macro(bob_python_install package)
  if (PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND AND Boost_FOUND)
    set(pydir ${CMAKE_INSTALL_PREFIX}/lib/python${PYTHON_VERSION}/bob)
    add_custom_target(${package}-python-install rsync -a ${CMAKE_CURRENT_SOURCE_DIR}/lib/ ${pydir}/${package}/ COMMENT "Installing bob::${package} python files...")
    add_dependencies(${package}-python-install pybob_${package})
    add_dependencies(${package}-python-install bob-python-install)
    add_dependencies(python-compilation ${package}-python-install)
  endif (PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND AND Boost_FOUND)
endmacro(bob_python_install package)

# This macro helps users to add python tests to cmake
function(bob_python_add_test)
  add_test(${ARGV})
  set_property(TEST ${ARGV0} APPEND PROPERTY ENVIRONMENT
    "DYLD_LIBRARY_PATH=${CMAKE_INSTALL_PREFIX}/lib:$ENV{DYLD_LIBRARY_PATH};LD_LIBRARY_PATH=${CMAKE_INSTALL_PREFIX}/lib:$ENV{LD_LIBRARY_PATH};PYTHONPATH=${CMAKE_INSTALL_PREFIX}/lib:${PYTHON_INSTALL_DIRECTORY}:$ENV{PYTHONPATH}")
endfunction(bob_python_add_test)

# This macro installs an example in a standard location
macro(bob_example_install subsys package file)
  set(exdir share/doc/examples/${subsys}/${package})
  install(PROGRAMS ${file} DESTINATION ${exdir})
endmacro(bob_example_install subsys package file)
