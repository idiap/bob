project(db-python)
cmake_minimum_required(VERSION 2.6)

#bob_python_package_bindings(db "src/main.cc" "core")
bob_python_package_bindings(db "" "core")

# A macro for controlling creation, downloading and dependencies automatically
# All tests should be dependent on the "${python-db-<dbname>-instantiation}" 
# test (pointer) target after this macro is called for a certain database.
macro(bob_build_db dbname)
  
  # If we are supposed to create the database, create it on the install
  # directory at an indicated global position where the databases can be used
  # for downloading through our web server.
  if (BOB_CREATE_DATABASES)
    bob_python_add_test(python-db-${dbname}-create
      ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py ${dbname} create 
      --recreate)
    set(python-db-${dbname}-instantiation python-db-${dbname}-create)

  else (BOB_CREATE_DATABASES)
    bob_python_add_test(python-db-${dbname}-download
      ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py ${dbname} download
      --url=${BOB_DATABASE_URL} --version=${BOB_DATABASE_VERSION})
    set(python-db-${dbname}-instantiation python-db-${dbname}-download)
  endif (BOB_CREATE_DATABASES)

  # If the copying has been set through the commandline, after creation or
  # downloading, copies the database to a given location.
  if (BOB_DB_INSTALL_PREFIX)
    bob_python_add_test(python-db-${dbname}-copy
      ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py ${dbname} copy
      ${BOB_DB_INSTALL_PREFIX})
    set_property(TEST python-db-${dbname}-copy APPEND PROPERTY DEPENDS
      ${python-db-${dbname}-instantiation})
  endif (BOB_DB_INSTALL_PREFIX)

endmacro(bob_build_db dbname)

# Run creation or downloading
bob_build_db(replay)
bob_build_db(banca)
bob_build_db(banca_small)
bob_build_db(biosecure)
bob_build_db(lfw)
bob_build_db(mobio)
bob_build_db(multipie)
bob_build_db(scface)
bob_build_db(xm2vts)

# Iris database driver and example tests
bob_python_add_test(python-db-iris-dump
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py iris dump --self-test)
bob_python_add_test(python-db-iris-dumplist2
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py iris dump --class=versicolor --self-test)

# Replay attack database driver tests.
# ######### Notice test dependencies!! ##################
bob_python_add_test(python-db-replay-dumplist
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py replay dumplist --self-test)
set_property(TEST python-db-replay-dumplist APPEND PROPERTY DEPENDS
  ${python-db-replay-instantiation})
bob_python_add_test(python-db-replay-dumplist2
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py replay dumplist --class=attack
  --group=devel --support=hand --protocol=highdef --self-test)
set_property(TEST python-db-replay-dumplist2 APPEND PROPERTY DEPENDS 
  ${python-db-replay-instantiation})
bob_python_add_test(python-db-replay-query ${CMAKE_CURRENT_SOURCE_DIR}/test/test_replay.py)
set_property(TEST python-db-replay-query APPEND PROPERTY DEPENDS 
  ${python-db-replay-instantiation})

# BANCA database driver tests.
# ######### Notice test dependencies!! ##################
bob_python_add_test(python-db-banca-dumplist
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py banca dumplist --self-test)
set_property(TEST python-db-banca-dumplist APPEND PROPERTY DEPENDS
  ${python-db-banca-instantiation})
bob_python_add_test(python-db-banca-dumplist2
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py banca dumplist --protocol=P --classes=client
  --groups=g1 --purposes=enrol --self-test)
set_property(TEST python-db-banca-dumplist2 APPEND PROPERTY DEPENDS 
  ${python-db-banca-instantiation})

# BANCA_SMALL database driver tests.
# ######### Notice test dependencies!! ##################
bob_python_add_test(python-db-banca_small-dumplist
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py banca_small dumplist --self-test)
set_property(TEST python-db-banca_small-dumplist APPEND PROPERTY DEPENDS
  ${python-db-banca_small-instantiation})
bob_python_add_test(python-db-banca_small-dumplist2
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py banca_small dumplist --protocol=P --classes=client
  --groups=g1 --purposes=enrol --self-test)
set_property(TEST python-db-banca_small-dumplist2 APPEND PROPERTY DEPENDS 
  ${python-db-banca_small-instantiation})

# Biosecure database driver tests.
# ######### Notice test dependencies!! ##################
bob_python_add_test(python-db-biosecure-dumplist
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py biosecure dumplist --self-test)
set_property(TEST python-db-biosecure-dumplist APPEND PROPERTY DEPENDS
  ${python-db-biosecure-instantiation})
bob_python_add_test(python-db-biosecure-dumplist2
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py biosecure dumplist --protocol=ca0 --classes=client 
  --groups=dev --purposes=enrol --self-test)
set_property(TEST python-db-biosecure-dumplist2 APPEND PROPERTY DEPENDS 
  ${python-db-biosecure-instantiation})

# Faceverif_fl database (uses file lists).
bob_python_add_test(python-db-faceverif_fl-query ${CMAKE_CURRENT_SOURCE_DIR}/test/test_faceverif_fl.py)

# LFW database driver tests.
# ######### Notice test dependencies!! ##################
bob_python_add_test(python-db-lfw-dumplist
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py lfw dumplist --self-test)
set_property(TEST python-db-lfw-dumplist APPEND PROPERTY DEPENDS
  ${python-db-lfw-instantiation})
bob_python_add_test(python-db-lfw-dumplist2
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py lfw dumplist --view=view1 --subset=fold1 --classes=matched --self-test)
set_property(TEST python-db-lfw-dumplist2 APPEND PROPERTY DEPENDS
  ${python-db-lfw-instantiation})

# MOBIO database driver tests.
# ######### Notice test dependencies!! ##################
bob_python_add_test(python-db-mobio-dumplist
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py mobio dumplist --self-test)
set_property(TEST python-db-mobio-dumplist APPEND PROPERTY DEPENDS
  ${python-db-mobio-instantiation})
bob_python_add_test(python-db-mobio-dumplist2
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py mobio dumplist --protocol=male --classes=client 
  --groups=dev --purposes=enrol --self-test)
set_property(TEST python-db-mobio-dumplist2 APPEND PROPERTY DEPENDS 
  ${python-db-mobio-instantiation})

# Multi-PIE database driver tests.
# ######### Notice test dependencies!! ##################
bob_python_add_test(python-db-multipie-dumplist
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py multipie dumplist --self-test)
set_property(TEST python-db-multipie-dumplist APPEND PROPERTY DEPENDS
  ${python-db-multipie-instantiation})
bob_python_add_test(python-db-multipie-dumplist2
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py multipie dumplist --protocol=M --classes=client 
  --groups=dev --purposes=enrol --self-test)
set_property(TEST python-db-multipie-dumplist2 APPEND PROPERTY DEPENDS 
  ${python-db-multipie-instantiation})

# SCface database driver tests.
# ######### Notice test dependencies!! ##################
bob_python_add_test(python-db-scface-dumplist
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py scface dumplist --self-test)
set_property(TEST python-db-scface-dumplist APPEND PROPERTY DEPENDS
  ${python-db-scface-instantiation})
bob_python_add_test(python-db-scface-dumplist2
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py scface dumplist --protocol=combined --classes=client 
  --groups=dev --purposes=enrol --self-test)
set_property(TEST python-db-scface-dumplist2 APPEND PROPERTY DEPENDS 
  ${python-db-scface-instantiation})

# XM2VTS database driver tests.
# ######### Notice test dependencies!! ##################
bob_python_add_test(python-db-xm2vts-dumplist
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py xm2vts dumplist --self-test)
set_property(TEST python-db-xm2vts-dumplist APPEND PROPERTY DEPENDS
  ${python-db-xm2vts-instantiation})
bob_python_add_test(python-db-xm2vts-dumplist2
  ${CMAKE_CURRENT_SOURCE_DIR}/script/dbmanage.py xm2vts dumplist --protocol=lp1 --classes=client 
  --groups=dev --purposes=enrol --self-test)
set_property(TEST python-db-xm2vts-dumplist2 APPEND PROPERTY DEPENDS 
  ${python-db-xm2vts-instantiation})

# Our examples (run and install)
bob_python_add_test(python-db-iris-lda-example
  ${CMAKE_CURRENT_SOURCE_DIR}/example/iris_lda.py --self-test)
bob_example_install("python" "db" example/iris_lda.py)
bob_python_add_test(python-db-iris-rprop-example
  ${CMAKE_CURRENT_SOURCE_DIR}/example/iris_rprop.py --self-test)
bob_example_install("python" "db" example/iris_rprop.py)
bob_python_add_test(python-db-iris-backprop-example
  ${CMAKE_CURRENT_SOURCE_DIR}/example/iris_backprop.py --self-test)
bob_example_install("python" "db" example/iris_backprop.py)
