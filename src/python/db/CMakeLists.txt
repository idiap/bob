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
    set(python-db-${dbname}-instantiation python-db-${dbname}-create)

  else (TORCH_CREATE_DATABASES)
    torch_python_add_test(python-db-${dbname}-download
      ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py ${dbname} download
      --url=${TORCH_DATABASE_URL} --version=${TORCH_DATABASE_VERSION})
    set(python-db-${dbname}-instantiation python-db-${dbname}-download)
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
torch_build_db(mobio)
torch_build_db(multipie)
torch_build_db(scface)
torch_build_db(xm2vts)

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
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py banca dumplist --protocol=P --classes=client
  --groups=g1 --purposes=enrol --self-test)
set_property(TEST python-db-banca-dumplist2 APPEND PROPERTY DEPENDS 
  ${python-db-banca-instantiation})

# Biosecure database driver tests.
# ######### Notice test dependencies!! ##################
torch_python_add_test(python-db-biosecure-dumplist
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py biosecure dumplist --self-test)
set_property(TEST python-db-biosecure-dumplist APPEND PROPERTY DEPENDS
  ${python-db-biosecure-instantiation})
torch_python_add_test(python-db-biosecure-dumplist2
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py biosecure dumplist --protocol=ca0 --classes=client 
  --groups=dev --purposes=enrol --self-test)
set_property(TEST python-db-biosecure-dumplist2 APPEND PROPERTY DEPENDS 
  ${python-db-biosecure-instantiation})

# MOBIO database driver tests.
# ######### Notice test dependencies!! ##################
torch_python_add_test(python-db-mobio-dumplist
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py mobio dumplist --self-test)
set_property(TEST python-db-mobio-dumplist APPEND PROPERTY DEPENDS
  ${python-db-mobio-instantiation})
torch_python_add_test(python-db-mobio-dumplist2
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py mobio dumplist --protocol=male --classes=client 
  --groups=dev --purposes=enrol --self-test)
set_property(TEST python-db-mobio-dumplist2 APPEND PROPERTY DEPENDS 
  ${python-db-mobio-instantiation})

# Multi-PIE database driver tests.
# ######### Notice test dependencies!! ##################
torch_python_add_test(python-db-multipie-dumplist
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py multipie dumplist --self-test)
set_property(TEST python-db-multipie-dumplist APPEND PROPERTY DEPENDS
  ${python-db-multipie-instantiation})
torch_python_add_test(python-db-multipie-dumplist2
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py multipie dumplist --protocol=M0 --classes=client 
  --groups=dev --purposes=enrol --self-test)
set_property(TEST python-db-multipie-dumplist2 APPEND PROPERTY DEPENDS 
  ${python-db-multipie-instantiation})

# SCface database driver tests.
# ######### Notice test dependencies!! ##################
torch_python_add_test(python-db-scface-dumplist
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py scface dumplist --self-test)
set_property(TEST python-db-scface-dumplist APPEND PROPERTY DEPENDS
  ${python-db-scface-instantiation})
torch_python_add_test(python-db-scface-dumplist2
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py scface dumplist --protocol=combined --classes=client 
  --groups=dev --purposes=enrol --self-test)
set_property(TEST python-db-scface-dumplist2 APPEND PROPERTY DEPENDS 
  ${python-db-scface-instantiation})

# XM2VTS database driver tests.
# ######### Notice test dependencies!! ##################
torch_python_add_test(python-db-xm2vts-dumplist
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py xm2vts dumplist --self-test)
set_property(TEST python-db-xm2vts-dumplist APPEND PROPERTY DEPENDS
  ${python-db-xm2vts-instantiation})
torch_python_add_test(python-db-xm2vts-dumplist2
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py xm2vts dumplist --protocol=lp1 --classes=client 
  --groups=dev --purposes=enrol --self-test)
set_property(TEST python-db-xm2vts-dumplist2 APPEND PROPERTY DEPENDS 
  ${python-db-xm2vts-instantiation})

# Our examples (run and install)
torch_python_add_test(python-db-iris-lda-example
  ${CMAKE_CURRENT_SOURCE_DIR}/example/iris_lda.py --self-test)
torch_example_install("python" "db" example/iris_lda.py)
torch_python_add_test(python-db-iris-rprop-example
  ${CMAKE_CURRENT_SOURCE_DIR}/example/iris_rprop.py --self-test)
torch_example_install("python" "db" example/iris_rprop.py)
torch_python_add_test(python-db-iris-backprop-example
  ${CMAKE_CURRENT_SOURCE_DIR}/example/iris_backprop.py --self-test)
torch_example_install("python" "db" example/iris_backprop.py)
