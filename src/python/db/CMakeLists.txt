project(db-python)
cmake_minimum_required(VERSION 2.6)

include(../../cmake/macros.cmake)

torch_python_install(db)

# Installs all python scripts
file(COPY script/ DESTINATION ${CMAKE_INSTALL_PREFIX}/bin FILES_MATCHING PATTERN "*.py")

# Our test units
#torch_python_add_test(python-db-array-pycodec-tests
#  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_codec.py)

# Our examples (run and install)
torch_python_add_test(python-db-iris_lda-example
  ${CMAKE_CURRENT_SOURCE_DIR}/example/iris_lda.py --self-test)
torch_example_install("python" "io" example/iris_lda.py)
