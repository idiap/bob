project(db-python)
cmake_minimum_required(VERSION 2.6)

include(../../cmake/macros.cmake)

torch_python_install(db)

# Installs all python scripts
file(COPY script/ DESTINATION ${CMAKE_INSTALL_PREFIX}/bin FILES_MATCHING PATTERN "*.py")

# Test some of our scripts

# Database creation can only be tested at Idiap as we have all the files in
# the right places to create the databases (original protocol files and data).
# To trigger the database creation, a special CMake variable needs to be set.
# This is done once per nightly build.
if (TORCH_CREATE_DATABASES)

  # Replay Attack Database
  torch_python_add_test(python-db-replay-create
    ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py replay create --recreate)
  set_property(TEST python-db-replay-create APPEND PROPERTY ENVIRONMENT
  "TORCH_DB_DIR=${CMAKE_INSTALL_PREFIX}/share/database")
  
  # BANCA Database
  torch_python_add_test(python-db-banca-create
    ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py banca create --recreate)
  set_property(TEST python-db-banca-create APPEND PROPERTY ENVIRONMENT
  "TORCH_DB_DIR=${CMAKE_INSTALL_PREFIX}/share/database")

  # Biosecure Database
  torch_python_add_test(python-db-biosecure-create
    ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py biosecure create --recreate)
  set_property(TEST python-db-biosecure-create APPEND PROPERTY ENVIRONMENT
  "TORCH_DB_DIR=${CMAKE_INSTALL_PREFIX}/share/database")

  set(DATABASE_TEST_DEPENDENCIES "python-db-replay-create;python-db-banca-create;python-db-biosecure-create")

else (TORCH_CREATE_DATABASES)

  # The next tests will complete the installation of torch by downloading the
  # database files required for each database operation.
  torch_python_add_test(python-db-replay-download
    ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py replay download
    --url=${TORCH_DATABASE_URL} --version=${TORCH_DATABASE_VERSION})

  torch_python_add_test(python-db-banca-download
    ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py banca download
    --url=${TORCH_DATABASE_URL} --version=${TORCH_DATABASE_VERSION})

  torch_python_add_test(python-db-biosecure-download
    ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py biosecure download
    --url=${TORCH_DATABASE_URL} --version=${TORCH_DATABASE_VERSION})

  set(DATABASE_TEST_DEPENDENCIES "python-db-replay-download;python-db-banca-download;python-db-biosecure-download")

endif (TORCH_CREATE_DATABASES)

# Iris database driver and example tests
torch_python_add_test(python-db-iris-dump
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py iris dump --self-test)
torch_python_add_test(python-db-iris-dumplist2
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py iris dump --class=versicolor --self-test)
torch_python_add_test(python-db-iris-lda-example
  ${CMAKE_CURRENT_SOURCE_DIR}/example/iris_lda.py --self-test)

# Replay attack database driver tests.
# ######### Notice test dependencies!! ##################
torch_python_add_test(python-db-replay-dumplist
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py replay dumplist --self-test)
set_property(TEST python-db-replay-dumplist APPEND PROPERTY DEPENDS
  ${DATABASE_TEST_DEPENDENCIES})
torch_python_add_test(python-db-replay-dumplist2
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py replay dumplist --class=attack
  --group=devel --support=hand --device=highdef --self-test)
set_property(TEST python-db-replay-dumplist2 APPEND PROPERTY DEPENDS 
  ${DATABASE_TEST_DEPENDENCIES})

# BANCA database driver tests.
# ######### Notice test dependencies!! ##################
torch_python_add_test(python-db-banca-dumplist
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py banca dumplist --self-test)
set_property(TEST python-db-banca-dumplist APPEND PROPERTY DEPENDS
  ${DATABASE_TEST_DEPENDENCIES})
torch_python_add_test(python-db-banca-dumplist2
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py banca dumplist --class=client
  --group=g1 --purpose=enrol --self-test)
set_property(TEST python-db-banca-dumplist2 APPEND PROPERTY DEPENDS 
  ${DATABASE_TEST_DEPENDENCIES})

# Biosecure database driver tests.
# ######### Notice test dependencies!! ##################
torch_python_add_test(python-db-biosecure-dumplist
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py biosecure dumplist --self-test)
set_property(TEST python-db-biosecure-dumplist APPEND PROPERTY DEPENDS
  ${DATABASE_TEST_DEPENDENCIES})
torch_python_add_test(python-db-biosecure-dumplist2
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py biosecure dumplist --class=client
  --group=g1 --purpose=enrol --self-test)
set_property(TEST python-db-biosecure-dumplist2 APPEND PROPERTY DEPENDS 
  ${DATABASE_TEST_DEPENDENCIES})

# Our examples (run and install)
torch_example_install("python" "io" example/iris_lda.py --self-test)
