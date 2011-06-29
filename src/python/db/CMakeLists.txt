project(db-python)
cmake_minimum_required(VERSION 2.6)

include(../../cmake/macros.cmake)

torch_python_install(db)

# Installs all python scripts
file(COPY script/ DESTINATION ${CMAKE_INSTALL_PREFIX}/bin FILES_MATCHING PATTERN "*.py")

# Our test units
#torch_python_add_test(python-db-array-pycodec-tests
#  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_codec.py)

# Test some of our scripts

# Iris database driver and example tests
torch_python_add_test(python-db-iris-dump
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py iris dump --self-test)
torch_python_add_test(python-db-iris-dumplist2
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py iris dump --class=versicolor --self-test)
torch_python_add_test(python-db-iris-lda-example
  ${CMAKE_CURRENT_SOURCE_DIR}/example/iris_lda.py --self-test)

# Replay attack database driver tests
torch_python_add_test(python-db-replay-dumplist
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py replay dumplist --self-test)
torch_python_add_test(python-db-replay-dumplist2
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py replay dumplist --class=attack
  --group=devel --support=hand --device=highdef --self-test)
# This can only be tested at Idiap!
#torch_python_add_test(python-db-replay-create
#  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py replay create --recreate)

# Our examples (run and install)
torch_example_install("python" "io" example/iris_lda.py)
