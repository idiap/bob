project(python)

cmake_minimum_required(VERSION 2.6)

# Installs python top-level wrapper
if (PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND AND Boost_FOUND)
  add_custom_target(torch-python-install ALL mkdir -p ${PYTHON_INSTALL_DIRECTORY} COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR}/torch ${PYTHON_INSTALL_DIRECTORY} COMMENT "Installing torch python files...")
  # Installs all python scripts
  file(COPY script/ DESTINATION ${CMAKE_INSTALL_PREFIX}/bin FILES_MATCHING PATTERN "*.py")

  # Compiles all installed python files
  add_custom_target(python-compilation ALL ${PYTHON_EXECUTABLE} -m compileall . 
    WORKING_DIRECTORY ${PYTHON_INSTALL_DIRECTORY} COMMENT "Compiling python files")

  include(../cmake/macros.cmake)

  # Finally, adds all python tests
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
  torch_python_add_test(python-transforms-tests 
    ${CMAKE_CURRENT_SOURCE_DIR}/test/test_transforms.py)
endif (PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND AND Boost_FOUND)


