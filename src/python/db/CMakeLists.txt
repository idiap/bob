project(db-python)
cmake_minimum_required(VERSION 2.6)

include(../../cmake/macros.cmake)

torch_python_install(db)

# Installs all python scripts
file(COPY script/ DESTINATION ${CMAKE_INSTALL_PREFIX}/bin FILES_MATCHING PATTERN "*.py")

# A macro for controlling creation, downloading and dependencies automatically
# All tests should be dependent on the "${python-db-<dbname>-instantiation}" 
# test (pointer) target after this macro is called for a certain database.
macro(torch_build_db dbname)
  
  # If we are supposed to create the database, create it on the install
  # directory at an indicated global position where the databases can be used
  # for downloading through our web server.
  if (TORCH_CREATE_DATABASES)
    torch_python_add_test(python-db-${dbname}-create
      ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py ${dbname} create 
      --recreate)
    set(python-db-${dbname}-instantiation ${python-db-${dbname}-create})

  else (TORCH_CREATE_DATABASES)
    torch_python_add_test(python-db-${dbname}-download
      ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py ${dbname} download
      --url=${TORCH_DATABASE_URL} --version=${TORCH_DATABASE_VERSION})
    set(python-db-${dbname}-instantiation ${python-db-${dbname}-download})
  endif (TORCH_CREATE_DATABASES)

  # If the copying has been set through the commandline, after creation or
  # downloading, copies the database to a given location.
  if (TORCH_DB_INSTALL_PREFIX)
    torch_python_add_test(python-db-${dbname}-copy
      ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py ${dbname} copy
      ${TORCH_DB_INSTALL_PREFIX})
    set_property(TEST python-db-${dbname}-copy APPEND PROPERTY DEPENDS
      ${python-db-${dbname}-instantiation})
  endif (TORCH_DB_INSTALL_PREFIX)

endmacro(torch_build_db dbname)

# Run creation or downloading
torch_build_db(replay)
torch_build_db(banca)
torch_build_db(biosecure)
torch_build_db(multipie)

# Iris database driver and example tests
torch_python_add_test(python-db-iris-dump
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py iris dump --self-test)
torch_python_add_test(python-db-iris-dumplist2
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py iris dump --class=versicolor --self-test)

# Replay attack database driver tests.
# ######### Notice test dependencies!! ##################
torch_python_add_test(python-db-replay-dumplist
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py replay dumplist --self-test)
set_property(TEST python-db-replay-dumplist APPEND PROPERTY DEPENDS
  ${python-db-replay-instantiation})
torch_python_add_test(python-db-replay-dumplist2
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py replay dumplist --class=attack
  --group=devel --support=hand --device=highdef --self-test)
set_property(TEST python-db-replay-dumplist2 APPEND PROPERTY DEPENDS 
  ${python-db-replay-instantiation})

# BANCA database driver tests.
# ######### Notice test dependencies!! ##################
torch_python_add_test(python-db-banca-dumplist
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py banca dumplist --self-test)
set_property(TEST python-db-banca-dumplist APPEND PROPERTY DEPENDS
  ${python-db-banca-instantiation})
torch_python_add_test(python-db-banca-dumplist2
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py banca dumplist --class=client
  --group=g1 --purpose=enrol --self-test)
set_property(TEST python-db-banca-dumplist2 APPEND PROPERTY DEPENDS 
  ${python-db-banca-instantiation})

# Biosecure database driver tests.
# ######### Notice test dependencies!! ##################
torch_python_add_test(python-db-biosecure-dumplist
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py biosecure dumplist --self-test)
set_property(TEST python-db-biosecure-dumplist APPEND PROPERTY DEPENDS
  ${python-db-biosecure-instantiation})
torch_python_add_test(python-db-biosecure-dumplist2
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py biosecure dumplist --class=client --group=dev --purpose=enrol --self-test)
set_property(TEST python-db-biosecure-dumplist2 APPEND PROPERTY DEPENDS 
  ${python-db-biosecure-instantiation})

# Multi-PIE database driver tests.
# ######### Notice test dependencies!! ##################
torch_python_add_test(python-db-multipie-dumplist
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py multipie dumplist --self-test)
set_property(TEST python-db-biosecure-dumplist APPEND PROPERTY DEPENDS
  ${python-db-biosecure-instantiation})
torch_python_add_test(python-db-multipie-dumplist2
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py multipie dumplist --class=client --group=dev --purpose=enrol --self-test)
set_property(TEST python-db-biosecure-dumplist2 APPEND PROPERTY DEPENDS 
  ${python-db-biosecure-instantiation})

# Our examples (run and install)
torch_python_add_test(python-db-iris-lda-example
  ${CMAKE_CURRENT_SOURCE_DIR}/example/iris_lda.py --self-test)
torch_example_install("python" "db" example/iris_lda.py)
