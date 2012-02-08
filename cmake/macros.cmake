# Andre Anjos <andre.anjos@idiap.ch>
# 4/August/2010

macro(bob_library package src dependencies shared)
  string(TOUPPER "${package}" PACKAGE)

  set(deps_list "")
  set(header_list "")
  set(compile_flags "")
  if(NOT ("${dependencies}" STREQUAL ""))
    foreach(dep ${dependencies})
      string(TOUPPER "${dep}" DEP)
      list(APPEND deps_list bob_${dep})
      list(APPEND header_list "${BOB_${DEP}_HEADER_DIRS}")
    endforeach(dep)
  endif(NOT ("${dependencies}" STREQUAL ""))

  list(REMOVE_DUPLICATES header_list)

  set(BOB_${PACKAGE}_HEADER_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${header_list} CACHE INTERNAL "${package} header dirs")
  #message(STATUS "${package} : ${BOB_${PACKAGE}_HEADER_DIRS}")
  include_directories(${BOB_${PACKAGE}_HEADER_DIRS})
  add_library(bob_${package} ${src})
  target_link_libraries(bob_${package} ${deps_list} ${shared})
  set_target_properties(bob_${package} PROPERTIES LIBRARY_OUTPUT_DIRECTORY  ${CMAKE_BINARY_DIR}/lib)
  install(TARGETS bob_${package} EXPORT bob
          LIBRARY DESTINATION lib
          ARCHIVE DESTINATION lib)
  install(DIRECTORY ${package} DESTINATION include/bob FILES_MATCHING PATTERN "*.h")
endmacro(bob_library)


# Creates a standard Bob test.
macro(bob_test package name src)
  set(testname bobtest_${package}_${name})

  # Please note we don't install test executables
  add_executable(${testname} ${src})
  target_link_libraries(${testname} bob_${package} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY_RELEASE})
  add_test(cxx-${package}-${name} ${testname} --log_level=test_suite)
  set_property(TEST cxx-${package}-${name} APPEND PROPERTY ENVIRONMENT "BOB_TESTDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR}/test/data")
endmacro(bob_test package src)


# Creates a standard Bob benchmark.
macro(bob_benchmark package name src)
  set(bindir bin)
  set(progname bobbench_${package}_${name})

  add_executable(${progname} ${src})
  target_link_libraries(${progname} bob_${package})
  install(TARGETS ${progname} RUNTIME DESTINATION ${bindir})
endmacro(bob_benchmark package name src)


macro(copy_if_different target files destination)
  foreach(file ${files})
    get_filename_component(file_name ${file} NAME)
    add_custom_command(TARGET ${target} POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different ${file} ${destination}/${file_name})
  endforeach(file)
endmacro()

macro(copy_files input_dir regex output_dir install_dir output_files target program)
  set(input_files "")
  foreach(exp ${regex})
    file(GLOB_RECURSE files RELATIVE "${input_dir}" "${input_dir}/${exp}")
    list(APPEND input_files ${files})
  endforeach()

  set(${output_files} "")
  foreach(input_file_rel ${input_files})
    set(input_file "${input_dir}/${input_file_rel}")
    set(output_file "${output_dir}/${input_file_rel}")

    add_custom_command(OUTPUT "${output_file}"
                       DEPENDS "${input_file}"
                       COMMAND ${CMAKE_COMMAND} -E copy "${input_file}" "${output_file}"
                       COMMENT "Copying ${input_file_rel} for ${target}")
                       #COMMENT "") ## Use this one to remove output text

    if (NOT install_dir STREQUAL "")
      get_filename_component(rel_path ${input_file_rel} PATH)
      if (program)
        install(PROGRAM ${input_file} DESTINATION "${install_dir}/${rel_path}")
      else()
        install(FILES ${input_file} DESTINATION "${install_dir}/${rel_path}")
      endif()
    endif()

    list(APPEND ${output_files} "${output_file}")
  endforeach()
endmacro()

macro(bob_python_bindings cxx_package package src pydependencies)
  string(TOUPPER "${package}" PACKAGE)
  string(TOUPPER "${cxx_package}" CXX_PACKAGE)

  set(pydeps_list "")
  set(pyheader_list "")
  if(NOT ("${pydependencies}" STREQUAL ""))
    foreach(dep ${pydependencies})
      string(TOUPPER "${dep}" DEP)
      list(APPEND pydeps_list pybob_${dep})
      list(APPEND pyheader_list "${BOB_PYTHON_${DEP}_HEADER_DIRS}")
    endforeach(dep)
  endif(NOT ("${pydependencies}" STREQUAL ""))

  list(REMOVE_DUPLICATES pyheader_list)

  set(BOB_PYTHON_${PACKAGE}_HEADER_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${BOB_${CXX_PACKAGE}_HEADER_DIRS} ${pyheader_list} CACHE INTERNAL "${package} header dirs")
  include_directories(${BOB_PYTHON_${PACKAGE}_HEADER_DIRS} ${python_INCLUDE_DIRS})
#   message(STATUS "${pydependencies}")
#   message(STATUS "${pyheader_list} !! ${pydeps_list}")
#   message(STATUS "${package}/${cxx_package} : ${BOB_PYTHON_${PACKAGE}_HEADER_DIRS} - ${BOB_${CXX_PACKAGE}_HEADER_DIRS}")

  if ("${src}" STREQUAL "")
    add_custom_target(pybob_${package} ALL)
    ## TODO Add correct dependencies
  else()
    add_library(pybob_${package} SHARED ${src})

    target_link_libraries(pybob_${package} bob_${cxx_package} ${pydeps_list} ${Boost_PYTHON_LIBRARY_RELEASE} ${PYTHON_LIBRARIES})
    set(pycxx_flags "-Wno-long-long -Wno-unused-function -Winvalid-pch")
    set_target_properties(pybob_${package} PROPERTIES OUTPUT_NAME "cxx")
    set_target_properties(pybob_${package} PROPERTIES PREFIX "_")
    set_target_properties(pybob_${package} PROPERTIES SUFFIX ".so")
    set_target_properties(pybob_${package} PROPERTIES COMPILE_FLAGS ${pycxx_flags})
    set_target_properties(pybob_${package} PROPERTIES LIBRARY_OUTPUT_DIRECTORY  ${CMAKE_BINARY_DIR}/lib/python${PYTHON_VERSION}/bob/${cxx_package})

    string(REPLACE "_" "/" package_path ${package})
    install(TARGETS pybob_${package} LIBRARY DESTINATION lib)#lib/python${PYTHON_VERSION}/bob/${package_path})
  endif()
  
  # Install scripts only if not a subpackage
  if (NOT "${package}" MATCHES "_")
    # Copy python files from lib folder
    copy_files("${CMAKE_CURRENT_SOURCE_DIR}/lib" "*.py" "${CMAKE_BINARY_DIR}/lib/python${PYTHON_VERSION}/bob/${cxx_package}"
               "lib/python${PYTHON_VERSION}/bob/${cxx_package}" output_lib_files "pybob_${package}" FALSE)

    # Copy python files from script folder
    copy_files("${CMAKE_CURRENT_SOURCE_DIR}/script" "*.py" "${CMAKE_BINARY_DIR}/bin" "bin" output_script_files "pybob_${package}" TRUE)
    
    add_custom_target(pybob_${package}_files DEPENDS ${output_lib_files} ${output_script_files})
    add_dependencies(pybob_${package} pybob_${package}_files)
  endif()
endmacro(bob_python_bindings)

macro(bob_python_package_bindings package src pydependencies)
  bob_python_bindings("${package}" "${package}" "${src}" "${pydependencies}")
endmacro(bob_python_package_bindings)

macro(bob_python_submodule_bindings package subpackage src pydependencies)
  bob_python_bindings("${package}" "${package}_${subpackage}" "${src}" "${pydependencies}")
  set_target_properties(pybob_${package}_${subpackage} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/python${PYTHON_VERSION}/bob/${package}/${subpackage})
endmacro(bob_python_submodule_bindings)

# This macro helps users to add python tests to cmake
function(bob_python_add_test)
  add_test(${ARGV})

  if (APPLE)
    # In OSX dlopen @ python requires the dyld path to be correctly set
    # for any C/C++ bindings you may have. It does not use the rpath for
    # some obscure reason - AA
    set_property(TEST ${ARGV0} APPEND PROPERTY ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/lib/python${PYTHON_VERSION}:$ENV{PYTHONPATH}")
    set_property(TEST ${ARGV0} APPEND PROPERTY ENVIRONMENT "DYLD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/lib:$ENV{DYLD_LIBRARY_PATH}")
  else ()
    # This must be Linux...
    set_property(TEST ${ARGV0} APPEND PROPERTY ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/lib/python${PYTHON_VERSION}:$ENV{PYTHONPATH}")
  endif ()

  set_property(TEST ${ARGV0} APPEND PROPERTY ENVIRONMENT "BOB_TESTDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR}/test/data")
  set_property(TEST ${ARGV0} APPEND PROPERTY ENVIRONMENT "BOB_VERSION=${BOB_VERSION}")
  set_property(TEST ${ARGV0} APPEND PROPERTY ENVIRONMENT "BOB_PLATFORM=${BOB_PLATFORM}")
endfunction(bob_python_add_test)

# This macro installs an example in a standard location
macro(bob_example_install subsys package file)
  set(exdir share/doc/examples/${subsys}/${package})
  install(PROGRAMS ${file} DESTINATION ${exdir})
endmacro(bob_example_install subsys package file)
