project(db-python)
cmake_minimum_required(VERSION 2.6)

bob_python_package_bindings(db "" "")

# Manages database copying
macro(bob_setup_db dbname)

  if (BOB_DATABASE_URL)

    set(sqlite_input "${CMAKE_SOURCE_DIR}/databases/${dbname}.sql3")
    set(sqlite_output "${CMAKE_BINARY_DIR}/lib/python${PYTHON_VERSION}/bob/db/${dbname}/db.sql3")
    add_custom_target(pybob_db_${dbname}_sqlite DEPENDS ${sqlite_input} COMMAND ${CMAKE_COMMAND} -E copy_if_different ${sqlite_input} ${sqlite_output})
    add_dependencies(pybob_db pybob_db_${dbname}_sqlite)

  endif (BOB_DATABASE_URL)

endmacro(bob_setup_db dbname)

# Run creation or copying
bob_setup_db(replay)
bob_setup_db(banca)
bob_setup_db(banca_small)
bob_setup_db(biosecure)
bob_setup_db(lfw)
bob_setup_db(mobio)
bob_setup_db(multipie)
bob_setup_db(scface)
bob_setup_db(xm2vts)

# Installs our script driver
bob_python_script(db "dbmanage.py" "lib/script/dbmanage.py" "main")

# Iris database driver and example tests
bob_python_add_test(db-iris-dump dbmanage.py iris dump --self-test)
bob_python_add_test(db-iris-dump-2 dbmanage.py iris dump --class=versicolor --self-test)

# Replay attack database driver tests.
bob_python_add_dependent_test(db-replay-dumplist python-db-replay-create
  dbmanage.py replay dumplist --self-test)

bob_python_add_dependent_test(db-replay-dumplist2 python-db-replay-create
  dbmanage.py replay dumplist --class=attack --group=devel --support=hand --protocol=highdef --self-test)

bob_python_add_unittest(db lib/test/replay.py)
set_property(TEST python-db-replay-main APPEND PROPERTY DEPENDS
  python-db-replay-create)

#bob_python_example(db "replay_example.py" lib/example/replay_example.py main)
#bob_python_add_test(db replay_example.py)

# BANCA database driver tests.
bob_python_add_dependent_test(db-banca-dumplist python-db-banca-create
  dbmanage.py banca dumplist --self-test)

bob_python_add_dependent_test(db-banca-dumplist2 python-db-banca-create
  dbmanage.py banca dumplist --protocol=P --classes=client --groups=g1 --purposes=enrol --self-test)

# BANCA_SMALL database driver tests.
bob_python_add_dependent_test(db-banca_small-dumplist
  python-db-banca_small-create
  dbmanage.py banca_small dumplist --self-test)

bob_python_add_dependent_test(db-banca_small-dumplist2
  python-db-banca_small-create
  dbmanage.py banca_small dumplist --protocol=P --classes=client --groups=g1 --purposes=enrol --self-test)

# Biosecure database driver tests.
bob_python_add_dependent_test(db-biosecure-dumplist python-db-biosecure-create
  dbmanage.py biosecure dumplist --self-test)

bob_python_add_dependent_test(db-biosecure-dumplist2 python-db-biosecure-create
  dbmanage.py biosecure dumplist --protocol=ca0 --classes=client --groups=dev --purposes=enrol --self-test)

# Faceverif_fl database (uses file lists).
bob_python_add_unittest(db lib/test/faceverif_fl.py)

# LFW database driver tests.
bob_python_add_dependent_test(db-lfw-dumplist python-db-lfw-create
  dbmanage.py lfw dumplist --self-test)

bob_python_add_dependent_test(db-lfw-dumplist2 python-db-lfw-create
  dbmanage.py lfw dumplist --view=view1 --subset=fold1 --classes=matched --self-test)

# MOBIO database driver tests.
bob_python_add_dependent_test(db-mobio-dumplist python-db-mobio-create
  dbmanage.py mobio dumplist --self-test)

bob_python_add_dependent_test(db-mobio-dumplist2 python-db-mobio-create
  dbmanage.py mobio dumplist --protocol=male --classes=client --groups=dev --purposes=enrol --self-test)

# Multi-PIE database driver tests.
bob_python_add_dependent_test(db-multipie-dumplist python-db-multipie-create
  dbmanage.py multipie dumplist --self-test)

bob_python_add_dependent_test(db-multipie-dumplist2 python-db-multipie-create
  dbmanage.py multipie dumplist --protocol=M --classes=client --groups=dev --purposes=enrol --self-test)

# Nuaa database driver tests.
bob_python_add_unittest(db lib/test/nuaa.py main)

# CASIA-FASD database driver tests.
bob_python_add_unittest(db lib/test/casia_fasd.py main)

# SCface database driver tests.
bob_python_add_dependent_test(db-scface-dumplist python-db-scface-create
  dbmanage.py scface dumplist --self-test)

bob_python_add_dependent_test(db-scface-dumplist2 python-db-scface-create
  dbmanage.py scface dumplist --protocol=combined --classes=client --groups=dev --purposes=enrol --self-test)

# XM2VTS database driver tests.
bob_python_add_dependent_test(db-xm2vts-dumplist python-db-xm2vts-create
  dbmanage.py xm2vts dumplist --self-test)

bob_python_add_dependent_test(db-xm2vts-dumplist2 python-db-xm2vts-create
  dbmanage.py xm2vts dumplist --protocol=lp1 --classes=client --groups=dev --purposes=enrol --self-test)

# Our examples (run and install)
bob_python_example(db iris_lda.py lib/example/iris_lda.py)
bob_python_add_test(db iris_lda.py --self-test)

bob_python_example(db iris_rprop.py lib/example/iris_rprop.py)
bob_python_add_test(db iris_rprop.py --self-test)

bob_python_example(db iris_backprop.py lib/example/iris_backprop.py)
bob_python_add_test(db iris_backprop.py --self-test)
