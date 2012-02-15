project(db-python)
cmake_minimum_required(VERSION 2.6)

set(pysrc
   "lib/__init__.py"
   "lib/banca/__init__.py"
   "lib/banca/commands.py"
   "lib/banca/create.py"
   "lib/banca/dumplist.py"
   "lib/banca/models.py"
   "lib/banca/query.py"
   "lib/banca_small/__init__.py"
   "lib/banca_small/commands.py"
   "lib/banca_small/create.py"
   "lib/banca_small/dumplist.py"
   "lib/banca_small/models.py"
   "lib/banca_small/query.py"
   "lib/biosecure/__init__.py"
   "lib/biosecure/commands.py"
   "lib/biosecure/create.py"
   "lib/biosecure/dumplist.py"
   "lib/biosecure/models.py"
   "lib/biosecure/query.py"
   "lib/faceverif_fl/__init__.py"
   "lib/faceverif_fl/commands.py"
   "lib/faceverif_fl/dumplist.py"
   "lib/faceverif_fl/query.py"
   "lib/iris/__init__.py"
   "lib/lfw/__init__.py"
   "lib/lfw/commands.py"
   "lib/lfw/create.py"
   "lib/lfw/dumplist.py"
   "lib/lfw/models.py"
   "lib/lfw/query.py"
   "lib/manage.py"
   "lib/mobio/__init__.py"
   "lib/mobio/commands.py"
   "lib/mobio/create.py"
   "lib/mobio/dumplist.py"
   "lib/mobio/models.py"
   "lib/mobio/query.py"
   "lib/multipie/__init__.py"
   "lib/multipie/commands.py"
   "lib/multipie/create.py"
   "lib/multipie/dumplist.py"
   "lib/multipie/models.py"
   "lib/multipie/query.py"
   "lib/openanything.py"
   "lib/replay/__init__.py"
   "lib/replay/commands.py"
   "lib/replay/create.py"
   "lib/replay/dumplist.py"
   "lib/replay/models.py"
   "lib/replay/query.py"
   "lib/scface/__init__.py"
   "lib/scface/commands.py"
   "lib/scface/create.py"
   "lib/scface/dumplist.py"
   "lib/scface/models.py"
   "lib/scface/query.py"
   "lib/sqlalchemy_migration.py"
   "lib/utils.py"
   "lib/xm2vts/__init__.py"
   "lib/xm2vts/commands.py"
   "lib/xm2vts/create.py"
   "lib/xm2vts/dumplist.py"
   "lib/xm2vts/models.py"
   "lib/xm2vts/query.py"

   "lib/example/__init__.py"
   "lib/example/iris_backprop.py"
   "lib/example/iris_lda.py"
   "lib/example/iris_rprop.py"
   "lib/example/replay_example.py"
   "lib/script/__init__.py"
   "lib/script/dbmanage.py"
   "lib/test/__init__.py"
   "lib/test/faceverif_fl.py"
   "lib/test/replay.py"
   )

bob_python_package_bindings(db "" "${pysrc}" "core_array")

# A macro for controlling creation, downloading and dependencies automatically
# All tests should be dependent on the "${python-db-<dbname>-instantiation}" 
# test (pointer) target after this macro is called for a certain database.
macro(bob_build_db dbname)
  
  # If we are supposed to create the database, create it on the install
  # directory at an indicated global position where the databases can be used
  # for downloading through our web server.
  if (BOB_CREATE_DATABASES)
    bob_python_add_test(db-${dbname}-create dbmanage.py ${dbname} create --recreate)
    set(python-db-${dbname}-instantiation python-db-${dbname}-create-dbmanage)

  else ()
    bob_python_add_test(db-${dbname}-download dbmanage.py ${dbname} download --url=${BOB_DATABASE_URL} --version=${BOB_DATABASE_VERSION})
    set(python-db-${dbname}-instantiation python-db-${dbname}-download-dbmanage)
  endif ()

  # If the copying has been set through the commandline, after creation or
  # downloading, copies the database to a given location.
  if (BOB_DB_INSTALL_PREFIX)
    bob_python_add_test(db-${dbname}-copy dbmanage.py ${dbname} copy ${BOB_DB_INSTALL_PREFIX})
    set_property(TEST python-db-${dbname}-copy-dbmanage APPEND PROPERTY DEPENDS
      ${python-db-${dbname}-instantiation})
  endif ()

endmacro(bob_build_db dbname)

# Installs our script driver
bob_python_script(db "dbmanage.py" "lib/script/dbmanage.py" "main")

# Installs all text/extra data for the databases
set(data
  "lib/iris/iris.names"
  "lib/iris/iris.data"
  )
bob_python_data(db "${data}" "lib/")

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
bob_python_add_test(db-iris-dump dbmanage.py iris dump --self-test)
bob_python_add_test(db-iris-dump-2 dbmanage.py iris dump --class=versicolor --self-test)

# Replay attack database driver tests.
bob_python_add_dependent_test(db-replay-dumplist 
  "${python-db-replay-instantiation}" 
  dbmanage.py replay dumplist --self-test)

bob_python_add_dependent_test(db-replay-dumplist2 
  "${python-db-replay-instantiation}" 
  dbmanage.py replay dumplist --class=attack --group=devel --support=hand --protocol=highdef --self-test)

bob_python_add_unittest(db lib/test/replay.py main)
set_property(TEST python-db-replay-main APPEND PROPERTY DEPENDS 
  ${python-db-replay-instantiation})

# BANCA database driver tests.
bob_python_add_dependent_test(db-banca-dumplist 
  "${python-db-banca-instantiation}"
  dbmanage.py banca dumplist --self-test)

bob_python_add_dependent_test(db-banca-dumplist2
  "${python-db-banca-instantiation}"
  dbmanage.py banca dumplist --protocol=P --classes=client --groups=g1 --purposes=enrol --self-test)

# BANCA_SMALL database driver tests.
bob_python_add_dependent_test(db-banca_small-dumplist
  "${python-db-banca_small-instantiation}" 
  dbmanage.py banca_small dumplist --self-test)

bob_python_add_dependent_test(db-banca_small-dumplist2 
  "${python-db-banca_small-instantiation}"
  dbmanage.py banca_small dumplist --protocol=P --classes=client --groups=g1 --purposes=enrol --self-test)

# Biosecure database driver tests.
bob_python_add_dependent_test(db-biosecure-dumplist 
  "${python-db-banca_small-instantiation}" 
  dbmanage.py biosecure dumplist --self-test)

bob_python_add_dependent_test(db-biosecure-dumplist2
  "${python-db-banca_small-instantiation}" 
  dbmanage.py biosecure dumplist --protocol=ca0 --classes=client --groups=dev --purposes=enrol --self-test)

# Faceverif_fl database (uses file lists).
bob_python_add_unittest(db lib/test/faceverif_fl.py main
  ${CMAKE_CURRENT_SOURCE_DIR}/data)

# LFW database driver tests.
bob_python_add_dependent_test(db-lfw-dumplist
  "${python-db-banca_small-instantiation}" 
  dbmanage.py lfw dumplist --self-test)

bob_python_add_dependent_test(db-lfw-dumplist2
  "${python-db-banca_small-instantiation}" 
  dbmanage.py lfw dumplist --view=view1 --subset=fold1 --classes=matched --self-test)

# MOBIO database driver tests.
bob_python_add_dependent_test(db-mobio-dumplist 
  "${python-db-banca_small-instantiation}" 
  dbmanage.py mobio dumplist --self-test)

bob_python_add_dependent_test(db-mobio-dumplist2 
  "${python-db-banca_small-instantiation}" 
  dbmanage.py mobio dumplist --protocol=male --classes=client --groups=dev --purposes=enrol --self-test)

# Multi-PIE database driver tests.
bob_python_add_dependent_test(db-multipie-dumplist 
  "${python-db-banca_small-instantiation}" 
  dbmanage.py multipie dumplist --self-test)

bob_python_add_dependent_test(db-multipie-dumplist2 
  "${python-db-banca_small-instantiation}" 
  dbmanage.py multipie dumplist --protocol=M --classes=client --groups=dev --purposes=enrol --self-test)

# SCface database driver tests.
bob_python_add_dependent_test(db-scface-dumplist 
  "${python-db-banca_small-instantiation}" 
  dbmanage.py scface dumplist --self-test)

bob_python_add_dependent_test(db-scface-dumplist2 
  "${python-db-banca_small-instantiation}" 
  dbmanage.py scface dumplist --protocol=combined --classes=client --groups=dev --purposes=enrol --self-test)

# XM2VTS database driver tests.
bob_python_add_dependent_test(db-xm2vts-dumplist 
  "${python-db-banca_small-instantiation}" 
  dbmanage.py xm2vts dumplist --self-test)

bob_python_add_dependent_test(db-xm2vts-dumplist2 
  "${python-db-banca_small-instantiation}" 
  dbmanage.py xm2vts dumplist --protocol=lp1 --classes=client --groups=dev --purposes=enrol --self-test)

# TODO: Our examples (run and install)
#bob_python_add_test(python-db-iris-lda-example
#  ${CMAKE_CURRENT_SOURCE_DIR}/example/iris_lda.py --self-test)
#bob_example_install("python" "db" example/iris_lda.py)
#bob_python_add_test(python-db-iris-rprop-example
#  ${CMAKE_CURRENT_SOURCE_DIR}/example/iris_rprop.py --self-test)
#bob_example_install("python" "db" example/iris_rprop.py)
#bob_python_add_test(python-db-iris-backprop-example
#  ${CMAKE_CURRENT_SOURCE_DIR}/example/iris_backprop.py --self-test)
#bob_example_install("python" "db" example/iris_backprop.py)
