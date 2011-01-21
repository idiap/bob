project(python)

cmake_minimum_required(VERSION 2.6)

# Installs python top-level wrapper
if (PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND AND Boost_FOUND)
 
  # We need first to create the basic python targets before including the
  # directories containing the python modules
  add_custom_target(torch-python-install ALL mkdir -p ${PYTHON_INSTALL_DIRECTORY} COMMAND rsync -a ${CMAKE_CURRENT_SOURCE_DIR}/lib/ ${PYTHON_INSTALL_DIRECTORY}/torch/ COMMENT "Installing root torch python files...")
  # Installs all python scripts
  file(COPY script/ DESTINATION ${CMAKE_INSTALL_PREFIX}/bin FILES_MATCHING PATTERN "*.py")

  # Compiles all installed python files
  add_custom_target(python-compilation ALL ${PYTHON_EXECUTABLE} -m compileall . 
    WORKING_DIRECTORY ${PYTHON_INSTALL_DIRECTORY} COMMENT "Compiling python files")

  add_subdirectory(core)
  add_subdirectory(database)
  add_subdirectory(ip)
  add_subdirectory(machine)
  add_subdirectory(scanning)
  add_subdirectory(sp)

  include(../cmake/macros.cmake)

  # Finally, adds all python tests
  torch_python_add_test(python-array-tests
    ${CMAKE_CURRENT_SOURCE_DIR}/test/test_array.py)
  torch_python_add_test(python-exception-tests 
    ${CMAKE_CURRENT_SOURCE_DIR}/test/test_exception.py)
  torch_python_add_test(python-video-tests
    ${CMAKE_CURRENT_SOURCE_DIR}/test/test_video.py)
  torch_python_add_test(python-facefinder-tests 
    ${CMAKE_CURRENT_SOURCE_DIR}/test/test_facefinder.py)
  torch_python_add_test(python-cropping-tests 
    ${CMAKE_CURRENT_SOURCE_DIR}/test/test_cropping.py)
  torch_python_add_test(python-videotensor-tests 
    ${CMAKE_CURRENT_SOURCE_DIR}/test/test_videotensor.py)
  torch_python_add_test(python-lbp-tests 
    ${CMAKE_CURRENT_SOURCE_DIR}/test/test_lbp.py)
  torch_python_add_test(python-filter-tests 
    ${CMAKE_CURRENT_SOURCE_DIR}/test/test_filters.py)
  torch_python_add_test(python-listdataset-tests 
    ${CMAKE_CURRENT_SOURCE_DIR}/test/test_listdataset.py)
  torch_python_add_test(python-gmm-tests 
    ${CMAKE_CURRENT_SOURCE_DIR}/test/test_gmm.py)
  torch_python_add_test(python-sp_naive-tests 
    ${CMAKE_CURRENT_SOURCE_DIR}/test/test_sp_naive.py)
  torch_python_add_test(python-sp_fftpack-tests 
    ${CMAKE_CURRENT_SOURCE_DIR}/test/test_sp_fftpack.py)
endif (PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND AND Boost_FOUND)


