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
endmacro()

# Creates a standard Bob test.
macro(bob_test package name src)
  set(testname bobtest_${package}_${name})

  # Please note we don't install test executables
  add_executable(${testname} ${src})
  target_link_libraries(${testname} bob_${package} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY_RELEASE})
  add_test(cxx-${package}-${name} ${testname} --log_level=test_suite)
  set_property(TEST cxx-${package}-${name} APPEND PROPERTY ENVIRONMENT "BOB_TESTDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR}/data")
endmacro()

# Creates a standard Bob benchmark.
macro(bob_benchmark package name src)
  set(bindir bin)
  set(progname bobbench_${package}_${name})

  add_executable(${progname} ${src})
  target_link_libraries(${progname} bob_${package})
  install(TARGETS ${progname} RUNTIME DESTINATION ${bindir})
endmacro()

# Installs data that needs copying to specific locations inside the python
# directory structure.
macro(bob_python_data package_name filenames relative)

  foreach(filename ${filenames})

    string(REPLACE "${relative}" "" stem "${filename}")

    set(output_stem "lib/python${PYTHON_VERSION}/bob/${package_name}/${stem}")

    set(output_file "${CMAKE_BINARY_DIR}/${output_stem}")
    add_custom_command(
      OUTPUT "${output_file}"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${filename}"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/${filename}" "${output_file}"
      COMMENT "Copying data file ${package}/${filename}")

    get_filename_component(basename ${filename} NAME)

    add_custom_target(pybob_${package_name}_${basename}_data DEPENDS "${output_file}")
    add_dependencies(pybob_${package_name} pybob_${package_name}_${basename}_data)

    # this will make the script available to the installation tree
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/${filename} DESTINATION ${output_stem})

  endforeach()

endmacro()

# Builds and installs a new script similar to what setuptools do for the
# command section of a setup.py build recipe.
macro(bob_python_script package_name script_name python_module python_method)

  # figures out the module name from the input file dependence name
  string(REPLACE ".py" "" module_name "${python_module}")
  string(REPLACE "/" "." module_name "${module_name}")
  string(REPLACE "lib." "bob.${package_name}." module_name "${module_name}")

  set(output_file "${CMAKE_BINARY_DIR}/bin/${script_name}")

  if(BOB_NON_ROOT_INSTALL)

    add_custom_command(
      OUTPUT "${output_file}"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${python_module}" "${CMAKE_SOURCE_DIR}/bin/make_wrapper.py"
      COMMAND ${PYTHON_EXECUTABLE}
      ARGS ${CMAKE_SOURCE_DIR}/bin/make_wrapper.py --non-root "${module_name}" "${python_method}" "${output_file}"
      COMMENT "Generating script ${script_name}")

  else(BOB_NON_ROOT_INSTALL)

    add_custom_command(
      OUTPUT "${output_file}"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${python_module}" "${CMAKE_SOURCE_DIR}/bin/make_wrapper.py"
      COMMAND ${PYTHON_EXECUTABLE}
      ARGS ${CMAKE_SOURCE_DIR}/bin/make_wrapper.py "${module_name}" "${python_method}" "${output_file}"
      COMMENT "Generating script ${script_name}")

  endif(BOB_NON_ROOT_INSTALL)

  get_filename_component(script_basename ${script_name} NAME_WE)

  add_custom_target(script_${script_basename} DEPENDS "${output_file}")
  add_dependencies(pybob_${package_name} script_${script_basename})

  # this will make the script available to the installation tree
  install(PROGRAMS ${CMAKE_BINARY_DIR}/bin/${script_name} DESTINATION bin)

endmacro()

# Tags version and platform, and set this to core of the python package
macro(bob_python_tag_build source)

  get_filename_component(version_file ${source} NAME)
  
  set(output_stem lib/python${PYTHON_VERSION}/bob/${version_file})
  set(output_file ${CMAKE_BINARY_DIR}/${output_stem})

  # TODO: This is not working as expected!
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${source} ${output_file})

  set(module_name "root.${version_file}")
  string(REPLACE ".py" "" module_name "${module_name}")
    
  # this will compile the version file
  add_custom_command(
    OUTPUT "${output_file}c"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${source}"
    COMMAND ${PYTHON_EXECUTABLE} -m py_compile "${output_file}"
    COMMENT "Compiling ${module_name}")

  # this will hook-up the dependencies so all works after the package is built
  add_custom_target(pybob_root.build DEPENDS "${output_file}c")
  add_dependencies(pybob_root pybob_root.build)

  # this will actually install the file
  get_filename_component(output_dir ${output_stem} PATH)
  install(FILES "${output_file}" "${output_file}c" DESTINATION ${output_dir})
endmacro()

# Installs and compiles all files given 
macro(bob_python_module package_name sources)

  foreach(source ${sources})

    # figures out the module name from the input file dependence name
    string(REPLACE "lib/" "" dest_name "${source}")

    set(module_name "${package_name}.${dest_name}")
    string(REPLACE ".py" "" module_name "${module_name}")
    string(REPLACE "/" "." module_name "${module_name}")
    
    # this is the temporary location for build tests
    if(${package_name} STREQUAL "root")
      set(output_stem lib/python${PYTHON_VERSION}/bob/${dest_name})
    else()
      set(output_stem lib/python${PYTHON_VERSION}/bob/${package_name}/${dest_name})
    endif()
    set(output_file ${CMAKE_BINARY_DIR}/${output_stem})

    # this will copy the file so it can be used during testing
    add_custom_command(
      OUTPUT "${output_file}c"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${source}"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/${source}" "${output_file}"
      COMMAND ${PYTHON_EXECUTABLE} -m py_compile "${output_file}"
      COMMENT "Copying and compiling ${module_name}")

    # this will hook-up the dependencies so all works after the package is built
    add_custom_target(pybob_${module_name} DEPENDS "${output_file}c")
    add_dependencies(pybob_${package_name} pybob_${module_name})

    # this will actually install the files
    get_filename_component(output_dir ${output_stem} PATH)
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/${source} DESTINATION ${output_dir})
    install(FILES ${output_file}c DESTINATION ${output_dir})

  endforeach(source ${sources})

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
  #message(STATUS "${pydependencies}")
  #message(STATUS "${pyheader_list} !! ${pydeps_list}")
  #message(STATUS "${package}/${cxx_package} : ${BOB_PYTHON_${PACKAGE}_HEADER_DIRS} - ${BOB_${CXX_PACKAGE}_HEADER_DIRS}")

  if("${src}" STREQUAL "")
    add_custom_target(pybob_${package} ALL)
    ## TODO Add correct dependencies

  else()
    add_library(pybob_${package} SHARED ${src})

    target_link_libraries(pybob_${package} bob_${cxx_package} ${pydeps_list} ${Boost_PYTHON_LIBRARY_RELEASE} ${PYTHON_LIBRARIES})
    set(pycxx_flags "-Wno-long-long -Wno-unused-function -Winvalid-pch")
    set_target_properties(pybob_${package} PROPERTIES OUTPUT_NAME "${package}")
    set_target_properties(pybob_${package} PROPERTIES PREFIX "_")
    set_target_properties(pybob_${package} PROPERTIES SUFFIX ".so")
    set_target_properties(pybob_${package} PROPERTIES COMPILE_FLAGS ${pycxx_flags})
    set_target_properties(pybob_${package} PROPERTIES LIBRARY_OUTPUT_DIRECTORY  ${CMAKE_BINARY_DIR}/lib/python${PYTHON_VERSION}/bob/${cxx_package})

    string(REPLACE "_" "/" package_path ${package})
    install(TARGETS pybob_${package} LIBRARY DESTINATION lib)#lib/python${PYTHON_VERSION}/bob/${package_path})
  endif()
  
endmacro(bob_python_bindings cxx_package package src pydependencies)

macro(bob_python_package_bindings package src pysrc pydependencies)
  bob_python_bindings("${package}" "${package}" "${src}" "${pydependencies}")
  bob_python_module("${package}" "${pysrc}")
endmacro()

macro(bob_python_submodule_bindings package subpackage src pydependencies)
  bob_python_bindings("${package}" "${package}_${subpackage}" "${src}" "${pydependencies}")
  set_target_properties(pybob_${package}_${subpackage} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/python${PYTHON_VERSION}/bob/${package}/${subpackage})
endmacro()

# This macro helps users to add python tests coded with the unittest module
macro(bob_python_add_unittest package_name python_module python_method)

  # figures out the module name from the input file dependence name
  string(REPLACE ".py" "" module_name "${python_module}")
  string(REPLACE "/" "." module_name "${module_name}")
  string(REPLACE "lib." "bob.${package_name}." module_name "${module_name}")
  string(REPLACE "." "_" test_filename "${module_name}")

  set(output_file "${CMAKE_BINARY_DIR}/bin/test_python_${test_filename}.py")
  
  get_filename_component(script_name ${output_file} NAME)
  
  get_filename_component(test_name_suffix ${python_module} NAME_WE)
  set(test_name_suffix "${test_name_suffix}-${python_method}")
  string(REPLACE "." "_" test_name "python-${package_name}-${test_name_suffix}")

  add_custom_command(
    OUTPUT "${output_file}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${python_module}" "${CMAKE_SOURCE_DIR}/bin/make_wrapper.py"
    COMMAND ${PYTHON_EXECUTABLE}
    ARGS ${CMAKE_SOURCE_DIR}/bin/make_wrapper.py "${module_name}" "${python_method}" "${output_file}"
    COMMENT "Generating unittest script ${script_name}")

  add_custom_target(${test_name} DEPENDS "${output_file}")
  add_dependencies(pybob_${package_name} ${test_name})

  # The 4th parameter is optional, it indicates the cwd for the test
  if(${ARGC} EQUAL 4)
    add_test(${test_name} ${output_file} --cwd=${ARGV3} --verbose --verbose)
  else()
    add_test(${test_name} ${output_file} --verbose --verbose)
  endif()

  # Common properties to all tests
  set_property(TEST ${test_name} APPEND PROPERTY ENVIRONMENT "BOB_TESTDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR}/data")
  set_property(TEST ${test_name} APPEND PROPERTY ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/lib/python${PYTHON_VERSION}")

  if(APPLE)
    set_property(TEST ${test_name} APPEND PROPERTY ENVIRONMENT "DYLD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/lib")
  endif()

endmacro()

# This macro helps users to add python tests to cmake
function(bob_python_add_test)

  list(GET ARGV 0 test_name)
  list(GET ARGV 1 prog)
  list(REMOVE_AT ARGV 0) #pop from front
  list(REMOVE_AT ARGV 0) #pop from front

  get_filename_component(prog_filename ${prog} NAME)

  # temporary hack to get the other tests working
  if ("${prog}" STREQUAL "${prog_filename}")
    # new style testing
    get_filename_component(prog_filename_we ${prog} NAME_WE)
    set(test_name "python-${test_name}-${prog_filename_we}")
    add_test(${test_name};${CMAKE_BINARY_DIR}/bin/${prog_filename};${ARGV})
  else()
    # TODO: get rid of this once all tests have been migrated
    add_test(${test_name};${prog};${ARGV})
  endif()

  # Common properties to all python tests
  set_property(TEST ${test_name} APPEND PROPERTY ENVIRONMENT "BOB_TESTDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR}/data")
  set_property(TEST ${test_name} APPEND PROPERTY ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/lib/python${PYTHON_VERSION}")

  if(APPLE)
    set_property(TEST ${test_name} APPEND PROPERTY ENVIRONMENT "DYLD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/lib")
  endif()

endfunction()

# This macro helps users to add python tests to cmake that depends on other
# tests
function(bob_python_add_dependent_test)

  list(GET ARGV 0 test_name)
  list(GET ARGV 1 dependencies)
  list(GET ARGV 2 prog)
  list(REMOVE_AT ARGV 0) #pop from front
  list(REMOVE_AT ARGV 0) #pop from front
  list(REMOVE_AT ARGV 0) #pop from front

  bob_python_add_test(${test_name};${prog};${ARGV})

  get_filename_component(prog_filename_we ${prog} NAME_WE)
  set(test_name "python-${test_name}-${prog_filename_we}") 

  set_property(TEST ${test_name} APPEND PROPERTY DEPENDS "${dependencies}")

endfunction()

# This macro installs an example in a standard location
macro(bob_example_install subsys package file)
  set(exdir share/doc/examples/${subsys}/${package})
  install(PROGRAMS ${file} DESTINATION ${exdir})
endmacro(bob_example_install subsys package file)
