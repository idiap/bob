# Andre Anjos <andre.anjos@idiap.ch>
# 4/August/2010

#################
# BEGIN C++ macro
#################

# Add a c++ bob package.
#
# package: name of the c++ package
# src: list of c++ file to compile
# dependencies: list of package name this package depends on. Dependent headers
#               are automatically available for the current target. Transitivity
#               is correctly handled
# shared: additional libraries to link with.
#
# Example: bob_library(io "foo.cc;bar.cc" "core" "foo.so")
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

  # rpath override rule for OSX
  set_target_properties(bob_${package} PROPERTIES INSTALL_NAME_DIR
    "${CMAKE_INSTALL_PREFIX}/lib")

  install(TARGETS bob_${package} EXPORT bob
          LIBRARY DESTINATION lib
          ARCHIVE DESTINATION lib)
  install(DIRECTORY ${package} DESTINATION include/bob FILES_MATCHING PATTERN "*.h")
endmacro()

# Creates a standard Bob test.
#
# package: package the test belongs to
# name: test name
# src: test source files
#
# Example: bob_test(io array test/array.cc)
macro(bob_test package name src)
  set(testname bobtest_${package}_${name})

  # Please note we don't install test executables
  add_executable(${testname} ${src})
  target_link_libraries(${testname} bob_${package} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY_RELEASE})
  add_test(cxx-${package}-${name} ${testname} --log_level=test_suite)
  set_property(TEST cxx-${package}-${name} APPEND PROPERTY ENVIRONMENT "BOB_TESTDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR}/data")
endmacro()

# Creates a standard Bob benchmark.
#
# package: package the benchmark belongs to
# name: benchmark name
# src: benchmark source files
#
# Example: bob_benchmark(core bigtensor2d "benchmark/bigtensor2d.cc")
macro(bob_benchmark package name src)
  set(bindir bin)
  set(progname bobbench_${package}_${name})

  add_executable(${progname} ${src})
  target_link_libraries(${progname} bob_${package})
  install(TARGETS ${progname} RUNTIME DESTINATION ${bindir})
endmacro()

#################
# END C++ macro
#################


# Installs an example in a standard location
#
# subsys: subsystem name (currently cxx and python are supported)
# package: package of the example
# file: list of files to install as example
#
# Example: bob_example_install(cxx core benchmark/bigtensor2d.cc)
macro(bob_example_install subsys package file)
  set(exdir share/doc/examples/${subsys}/${package})
  install(PROGRAMS ${file} DESTINATION ${exdir})
endmacro(bob_example_install subsys package file)


####################
# BEGIN python macro
####################

# Internal macro
# Convert a path to a python file in lib directory to a python module name
#
# package_name: package name of the file
# python_path: path to the file. Must begin with lib
# module_name: [output] converted module name
macro(bob_convert_file_path_to_module_name package_name python_path module_name)
  # figures out the module name from the input file dependence name
  string(REGEX REPLACE ".py$" "" ${module_name} "${python_path}")
  string(REPLACE "/" "." ${module_name} "${${module_name}}")

  if(${package_name} STREQUAL "root")
    string(REGEX REPLACE "^lib." "bob." ${module_name} "${${module_name}}")
  else()
    string(REGEX REPLACE "^lib." "bob.${package_name}." ${module_name} "${${module_name}}")
  endif()
endmacro()

# Internal macro
# Wrap a python function. Make a standalone python file which calls a python
# function.
#
# The generated script uses files build in binary tree. If file_to_install is
# not empty, another script for the install tree is generated.
#
# package_name: package name
# file_path: path to the python file containing the function. Must begin with lib.
# output_path: path where the standalone script is created.
# python_method: python function to warp (if "" set to "main")
# file_to_install: [output] path to the file to install. If empty no install file
#                  is generated.
macro(bob_wrap_python_file package_name file_path output_path python_method file_to_install)
   bob_convert_file_path_to_module_name(${package_name} ${file_path} module_name)

  set(BOB_PYTHONPATH ${CMAKE_BINARY_DIR}/lib/python${PYTHON_VERSION})
  configure_file(${CMAKE_SOURCE_DIR}/python/bin/wrapper.py.in ${output_path})

  if(NOT ${file_to_install} STREQUAL "")
    # Compute the temporary filename
    get_filename_component(filename ${output_path} NAME)

    # We add md5 of the full path to prevent name collision
    # The string(MD5 ...) command doesn't exists before 2.8.7
    if(CMAKE_VERSION VERSION_LESS "2.8.7")
      set(md5 "")
    else()
      string(MD5 md5 ${output_path})
    endif()

    set(BOB_MODULE ${module_name})
    if(python_method STREQUAL "")
      set(BOB_METHOD "main")
    else()
      set(BOB_METHOD "${python_method}")
    endif()

    if(IS_ABSOLUTE ${CMAKE_INSTALL_PREFIX})
      set(ABSOLUTE_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
    else()
      set(ABSOLUTE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_PREFIX})
    endif()

    if(IS_ABSOLUTE ${build_path})
      set(ABSOLUTE_build_path ${build_path})
    else()
      set(ABSOLUTE_build_path ${CMAKE_BINARY_DIR}/${build_path})
    endif()

    if(IS_ABSOLUTE ${install_dir})
      set(ABSOLUTE_install_dir ${install_dir})
    else()
      set(ABSOLUTE_install_dir ${ABSOLUTE_INSTALL_PREFIX}/${install_dir})
    endif()

    set(BOB_PYTHONPATH ${ABSOLUTE_INSTALL_PREFIX}/lib/python${PYTHON_VERSION})

    set(${file_to_install} ${CMAKE_BINARY_DIR}/tmp/${filename}${md5})
    configure_file(${CMAKE_SOURCE_DIR}/python/bin/wrapper.py.in ${${file_to_install}})

  endif()
endmacro()

# Creates and installs a python script (in bin directory) from a python method.
#
#   bob_python_script(package_name script_name file_path [python_method])
#
# package_name: package name
# script_name: name of the output script
# file_path: path to the python file containing the method.
# python_method: python method to execute by the script (default "main")
#
# Example: bob_python_script(ip blockDCT.py lib/script/blockDCT.py)
macro(bob_python_script package_name script_name file_path)
  if(${ARGC} LESS 4)
    set(python_method "main")
  else()
    set(python_method "${ARGV3}")
  endif()

  set(output_file "${CMAKE_BINARY_DIR}/bin/${script_name}")
  bob_wrap_python_file(${package_name} ${file_path} ${output_file} "${python_method}" file_to_install)

  # this will make the script available to the installation tree
  install(PROGRAMS ${file_to_install} DESTINATION bin)
endmacro()

# Creates and installs a python script as an example from a python method.
# In the binary tree, the script is located in bin directory. In install tree,
# the script is in the standard example directory.
#
#   bob_python_example(package_name script_name file_path [python_method])
#
# package_name: package name
# script_name: name of the output script
# file_path: path to the python file containing the method.
# python_method: python method to execute by the script (default "main")
#
# Example: bob_python_example(io video2frame.py lib/example/video2frame.py)
macro(bob_python_example package_name script_name file_path)
  if(${ARGC} LESS 4)
    set(python_method "main")
  else()
    set(python_method "${ARGV3}")
  endif()

  set(output_file "${CMAKE_BINARY_DIR}/bin/${script_name}")
  bob_wrap_python_file(${package_name} ${file_path} ${output_file} "${python_method}" "")

  bob_example_install(python ${package_name} ${file_path})
endmacro()

# Add python tests coded with the unittest module
#
#   bob_python_add_unittest(package_name file_path [python_method] [working_directory])
#
# package_name: package name
# file_path: path to the python file containing the test method
# python_method: python test method (default "main")
# working_directory: working directory where the test are executed. Default to
#                    "data" dir in current source dir.
#
# Example: bob_python_add_unittest(io lib/test/array.py)
macro(bob_python_add_unittest package_name file_path)
  if(${ARGC} LESS 3)
    set(python_method "main")
  else()
    set(python_method "${ARGV2}")
  endif()

  # The 4th parameter is optional, it indicates the cwd for the test
  if(${ARGC} LESS 4)
    if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/data")
      set(cwd "${CMAKE_CURRENT_SOURCE_DIR}/data")
    else()
      set(cwd "")
    endif()
  else()
    set(cwd "${ARGV3}")
  endif()

  string(REGEX REPLACE ".py$" "" test_filename "${file_path}")
  string(REPLACE "/" "." test_filename "${test_filename}")
  string(REGEX REPLACE "^lib." "bob.${package_name}." test_filename "${test_filename}")
  string(REPLACE "." "_" test_filename "${test_filename}")
  set(output_file "${CMAKE_BINARY_DIR}/bin/test_python_${test_filename}.py")

  bob_wrap_python_file(${package_name} ${file_path} ${output_file} "${python_method}" "")

  get_filename_component(test_name_suffix ${file_path} NAME_WE)
  set(test_name_suffix "${test_name_suffix}-${python_method}")
  string(REPLACE "." "_" test_name "python-${package_name}-${test_name_suffix}")

  if(cwd STREQUAL "")
    add_test(${test_name} ${output_file} --verbose --verbose)
  else()
    add_test(${test_name} ${output_file} --cwd=${cwd} --verbose --verbose)
  endif()

  # Common properties to all tests
  set_property(TEST ${test_name} APPEND PROPERTY ENVIRONMENT "BOB_TESTDATA_DIR=${CMAKE_CURRENT_SOURCE_DIR}/data")
endmacro()

# Internal macro.
# Recursively copy files from a folder to the build tree and install them. The
# macro respects the relative path of the file.
# Warning: this macro only generate custom command to copy the files. To really
# copy the files you need to have a target that depends on these files.
#
#  copy_files(input_dir include_regex exclude_regex output_dir install_dir output_files target program install_exclude_paths)
#
# input_dir: directory with the files to copy
# include_regex: globbing expressions for the included files
# exclude_regex: globbing expressions for the excluded files
# output_dir: directory to copy the files (path relative to CMAKE_BINARY_DIR)
# install_dir: directory to install the files, if "" no file is installed. (path
#              relative to CMAKE_INSTALL_PREFIX)
# output_files: [output] list of file generated by the commands
# target: related target (only used to display state)
# install_exclude_paths: list of regex applies to the relative input path. Files
#                        that match these regex are not installed.
#
macro(copy_files input_dir include_regex exclude_regex output_dir install_dir output_files target program install_exclude_paths)
  set(input_files "")
  foreach(exp ${include_regex})
    file(GLOB_RECURSE files RELATIVE "${input_dir}" "${input_dir}/${exp}")
    list(APPEND input_files ${files})
  endforeach()

  foreach(exp ${exclude_regex})
    file(GLOB_RECURSE files RELATIVE "${input_dir}" "${input_dir}/${exp}")
    list(REMOVE_ITEM input_files "${files}")
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

    set(install_exclude FALSE)
    foreach(install_exclude_path ${install_exclude_paths})
      if(${input_file_rel} MATCHES ${install_exclude_path})
        set(install_exclude TRUE)
        break()
      endif()
    endforeach()

    if (NOT install_dir STREQUAL "" AND NOT install_exclude)
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

# Internal macro.
# Add a new python package. The target automatically depends on the
# corresponding c++ package and his python binding.
#
#   bob_python_bindings(cxx_package package cxx_src pydependencies)
#
# cxx_package: corresponding c++ package
# package: name of the python package
# cxx_src: c++ source for the package
# pydependencies: list of additional python package dependencies
macro(bob_python_bindings cxx_package package cxx_src pydependencies)
  if(${ARGC} LESS 5)
    set(subpackage "FALSE")
  else()
    set(subpackage "${ARGV4}")
  endif()

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

  if("${cxx_src}" STREQUAL "")
    add_custom_target(pybob_${package} ALL)
    ## TODO Add correct dependencies

  else()
    add_library(pybob_${package} SHARED ${cxx_src})

    target_link_libraries(pybob_${package} bob_${cxx_package} ${pydeps_list} ${Boost_PYTHON_LIBRARY_RELEASE} ${PYTHON_LIBRARIES})
    set(pycxx_flags "-Wno-long-long -Wno-unused-function -Winvalid-pch")
    set_target_properties(pybob_${package} PROPERTIES OUTPUT_NAME "${package}")
    set_target_properties(pybob_${package} PROPERTIES PREFIX "_")
    set_target_properties(pybob_${package} PROPERTIES SUFFIX ".so")
    set_target_properties(pybob_${package} PROPERTIES COMPILE_FLAGS ${pycxx_flags})
    set_target_properties(pybob_${package} PROPERTIES LIBRARY_OUTPUT_DIRECTORY  ${CMAKE_BINARY_DIR}/lib/python${PYTHON_VERSION}/bob/${cxx_package})

    string(REPLACE "_" "/" package_path ${package})

    # rpath override rule for OSX
    set_target_properties(pybob_${package} PROPERTIES INSTALL_NAME_DIR
      "${CMAKE_INSTALL_PREFIX}/lib/python${PYTHON_VERSION}/bob/${package_path}")

    # makes sure bindings to the right places
    install(TARGETS pybob_${package} LIBRARY DESTINATION lib/python${PYTHON_VERSION}/bob/${package_path})

    # makes sure headers are installed
    install(DIRECTORY ${package} DESTINATION include/bob FILES_MATCHING PATTERN "*.h")

  endif()

  # Install scripts only if not a subpackage
  if (NOT subpackage)
    if(CXX_PACKAGE STREQUAL "ROOT")
      set(bin_path "${CMAKE_BINARY_DIR}/lib/python${PYTHON_VERSION}/bob")
      set(install_path "lib/python${PYTHON_VERSION}/bob")
    else()
      set(bin_path "${CMAKE_BINARY_DIR}/lib/python${PYTHON_VERSION}/bob/${cxx_package}")
      set(install_path "lib/python${PYTHON_VERSION}/bob/${cxx_package}")
    endif()

    set(input_dir "${CMAKE_CURRENT_SOURCE_DIR}/lib")

    # Copy python files from lib folder
    copy_files("${input_dir}" "*" "*~;.*.swp;.swp*;*.pyc;*.in" ${bin_path}
                ${install_path} output_lib_files "pybob_${package}" FALSE "^test;^example")

    file(GLOB_RECURSE files RELATIVE "${input_dir}" "${input_dir}/*.in")
    foreach(file ${files})
      string(REGEX REPLACE "\\.in$" "" outputfile "${file}")
      configure_file("${input_dir}/${file}" "${bin_path}/${outputfile}" @ONLY)
      install(FILES "${bin_path}/${outputfile}"
              DESTINATION ${install_path})
    endforeach()

    add_custom_target(pybob_${package}_files DEPENDS ${output_lib_files} ${output_script_files})
    add_dependencies(pybob_${package} pybob_${package}_files)
  endif()

  if(NOT TARGET pybob_compile_python_files)
    add_custom_target(pybob_compile_python_files ALL
                      COMMAND ${PYTHON_EXECUTABLE} -m compileall -q "${CMAKE_BINARY_DIR}/lib/python${PYTHON_VERSION}/bob")
  endif()

  add_dependencies(pybob_compile_python_files pybob_${package})

endmacro()

# Add a new python package binding c++ code. The target automatically depends on
# the corresponding c++ package and his python binding.
#
#   bob_python_package_bindings(package cxx_src pydependencies)
#
# package: name of the python package (should be the same as the corresponding
#          cxx package)
# cxx_src: c++ source for the package
# pydependencies: list of additional python package dependencies
#
# Example: bob_python_package_bindings(io "src/foo.cc;src/bar.cc" core_array)
macro(bob_python_package_bindings package cxx_src pydependencies)
  bob_python_bindings("${package}" "${package}" "${cxx_src}" "${pydependencies}" FALSE)
endmacro()

# Add a new python subpackage binding c++ code. The target automatically depends
# on the corresponding c++ package and his python binding.
#
#   bob_python_subpackage_bindings(package subpackage cxx_src pydependencies)
#
# package: name of the python package (should be the same as the corresponding
#          cxx package)
# subpackage: name of the python subpackage
# cxx_src: c++ source for the package
# pydependencies: list of additional python package dependencies
#
# Example: bob_python_subpackage_bindings(core array "src/foo.cc" "")
macro(bob_python_subpackage_bindings package subpackage cxx_src pydependencies)
  bob_python_bindings("${package}" "${package}_${subpackage}" "${cxx_src}" "${pydependencies}" TRUE)
  set_target_properties(pybob_${package}_${subpackage} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/python${PYTHON_VERSION}/bob/${package}/${subpackage})
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
