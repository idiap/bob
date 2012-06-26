project(db-python)
cmake_minimum_required(VERSION 2.6)

bob_python_package_bindings(db "" "")

# Manages database copying
macro(bob_setup_db dbname)

  if (BOB_DATABASE_URL)

    # The normal way to build: user has download package with databases or the
    # databases have been previously downloaded @ ${CMAKE_SOURCE_DIR}/databases
    set(sqlite_input "${CMAKE_SOURCE_DIR}/databases/${dbname}.sql3")
    set(sqlite_output "${PYTHON_SITE_PACKAGES}/bob/db/${dbname}")
    add_custom_target(pybob_db_${dbname}_sqlite DEPENDS ${sqlite_input} COMMAND ${CMAKE_COMMAND} -E copy_if_different ${sqlite_input} ${CMAKE_BINARY_DIR}/${sqlite_output}/db.sql3)
    add_dependencies(pybob_db pybob_db_${dbname}_sqlite)
    install(FILES ${sqlite_input} DESTINATION ${sqlite_output} RENAME db.sql3)

  else()

    # If the current build creates the databases, you have to call:
    #  * make
    #  * make install
    #  * make db-install
    add_custom_target(pybob_db_create_${dbname}_sqlite COMMAND ${CMAKE_BINARY_DIR}/bin/bob_dbmanage.py ${dbname} create --recreate COMMENT "Generating ${dbname} sqlite database")
    add_dependencies(db-create pybob_db_create_${dbname}_sqlite)
    set(sqlite_base "${PYTHON_SITE_PACKAGES}/bob/db/${dbname}/db.sql3")
    set(sqlite_input "${CMAKE_BINARY_DIR}/${sqlite_base}")
    set(sqlite_output "${CMAKE_INSTALL_PREFIX}/${sqlite_base}")
    add_custom_target(pybob_db_install_${dbname}_sqlite ${CMAKE_COMMAND} -E copy_if_different ${sqlite_input} ${sqlite_output} COMMENT "Installing ${dbname} sqlite database")
    add_dependencies(pybob_db_install_${dbname}_sqlite pybob_db_create_${dbname}_sqlite)
    add_dependencies(db-install pybob_db_install_${dbname}_sqlite)

  endif()

endmacro(bob_setup_db dbname)

# Run creation or copying
add_custom_target(db-create)
add_custom_target(db-install)
bob_setup_db(replay)
bob_setup_db(banca)
bob_setup_db(banca_small)
bob_setup_db(biosecure)
bob_setup_db(lfw)
bob_setup_db(mobio)
bob_setup_db(multipie)
bob_setup_db(scface)
bob_setup_db(xm2vts)
bob_setup_db(fir)
bob_setup_db(gbu)

# Installs our script driver
bob_python_script(db "bob_dbmanage.py" "lib/script/dbmanage.py" "main")

# Iris database driver and example tests
bob_python_add_test(db-iris-dump bob_dbmanage.py iris dump --self-test)
bob_python_add_test(db-iris-dump-2 bob_dbmanage.py iris dump --class=versicolor --self-test)

# Replay attack database driver tests.
bob_python_add_dependent_test(db-replay-dumplist python-db-replay-create
  bob_dbmanage.py replay dumplist --self-test)

bob_python_add_dependent_test(db-replay-dumplist2 python-db-replay-create
  bob_dbmanage.py replay dumplist --class=attack --group=devel --support=hand --protocol=highdef --self-test)

# Just check functionality
bob_python_add_dependent_test(db-replay-checkfiles python-db-replay-create
  bob_dbmanage.py replay checkfiles --self-test)

bob_python_add_unittest(db lib/test/replay.py)
set_property(TEST python-db-replay-main APPEND PROPERTY DEPENDS
  python-db-replay-create)

#bob_python_example(db "replay_example.py" lib/example/replay_example.py main)
#bob_python_add_test(db replay_example.py)

# BANCA database driver tests.
bob_python_add_dependent_test(db-banca-dumplist python-db-banca-create
  bob_dbmanage.py banca dumplist --self-test)

bob_python_add_dependent_test(db-banca-dumplist2 python-db-banca-create
  bob_dbmanage.py banca dumplist --protocol=P --classes=client --groups=g1 --purposes=enrol --self-test)

bob_python_add_dependent_test(db-banca-checkfiles python-db-banca-create
  bob_dbmanage.py banca checkfiles --self-test)

# BANCA_SMALL database driver tests.
bob_python_add_dependent_test(db-banca_small-dumplist
  python-db-banca_small-create
  bob_dbmanage.py banca_small dumplist --self-test)

bob_python_add_dependent_test(db-banca_small-dumplist2
  python-db-banca_small-create
  bob_dbmanage.py banca_small dumplist --protocol=P --classes=client --groups=g1 --purposes=enrol --self-test)

bob_python_add_dependent_test(db-banca_small-checkfiles python-db-banca_small-create
  bob_dbmanage.py banca_small checkfiles --self-test)

# Biosecure database driver tests.
bob_python_add_dependent_test(db-biosecure-dumplist python-db-biosecure-create
  bob_dbmanage.py biosecure dumplist --self-test)

bob_python_add_dependent_test(db-biosecure-dumplist2 python-db-biosecure-create
  bob_dbmanage.py biosecure dumplist --protocol=ca0 --classes=client --groups=dev --purposes=enrol --self-test)

bob_python_add_dependent_test(db-biosecure-checkfiles python-db-biosecure-create
  bob_dbmanage.py biosecure checkfiles --self-test)

# Faceverif_fl database (uses file lists).
bob_python_add_unittest(db lib/test/faceverif_fl.py)

# LFW database driver tests.
bob_python_add_unittest(db lib/test/lfw.py main)

bob_python_add_dependent_test(db-lfw-dumplist python-db-lfw-create
  bob_dbmanage.py lfw dumplist --self-test)

bob_python_add_dependent_test(db-lfw-dumppairs python-db-lfw-create
  bob_dbmanage.py lfw dumppairs --self-test)

bob_python_add_dependent_test(db-lfw-checkfiles python-db-lfw-create
  bob_dbmanage.py lfw checkfiles --self-test)

# MOBIO database driver tests.
bob_python_add_dependent_test(db-mobio-dumplist python-db-mobio-create
  bob_dbmanage.py mobio dumplist --self-test)

bob_python_add_dependent_test(db-mobio-dumplist2 python-db-mobio-create
  bob_dbmanage.py mobio dumplist --protocol=male --classes=client --groups=dev --purposes=enrol --self-test)

bob_python_add_dependent_test(db-mobio-checkfiles python-db-mobio-create
  bob_dbmanage.py mobio checkfiles --self-test)

# Multi-PIE database driver tests.
bob_python_add_dependent_test(db-multipie-dumplist python-db-multipie-create
  bob_dbmanage.py multipie dumplist --self-test)

bob_python_add_dependent_test(db-multipie-dumplist2 python-db-multipie-create
  bob_dbmanage.py multipie dumplist --protocol=M --classes=client --groups=dev --purposes=enrol --self-test)

bob_python_add_dependent_test(db-multipie-checkfiles python-db-multipie-create
  bob_dbmanage.py multipie checkfiles --self-test)

# Nuaa database driver tests.
bob_python_add_unittest(db lib/test/nuaa.py main)
bob_python_add_test(db-nuaa-dumplist bob_dbmanage.py nuaa dumplist --self-test)
bob_python_add_test(db-nuaa-checkfiles bob_dbmanage.py nuaa checkfiles --self-test)

# CASIA-FASD database driver tests.
bob_python_add_unittest(db lib/test/casia_fasd.py main)
bob_python_add_test(db-casia_fasd-dumplist bob_dbmanage.py casia_fasd dumplist --self-test)
bob_python_add_test(db-casia_fasd-checkfiles bob_dbmanage.py casia_fasd checkfiles --self-test)

# SCface database driver tests.
bob_python_add_dependent_test(db-scface-dumplist python-db-scface-create
  bob_dbmanage.py scface dumplist --self-test)

bob_python_add_dependent_test(db-scface-dumplist2 python-db-scface-create
  bob_dbmanage.py scface dumplist --protocol=combined --classes=client --groups=dev --purposes=enrol --self-test)

bob_python_add_dependent_test(db-scface-checkfiles python-db-scface-create
  bob_dbmanage.py scface checkfiles --self-test)

# XM2VTS database driver tests.
bob_python_add_dependent_test(db-xm2vts-dumplist python-db-xm2vts-create
  bob_dbmanage.py xm2vts dumplist --self-test)

bob_python_add_dependent_test(db-xm2vts-dumplist2 python-db-xm2vts-create
  bob_dbmanage.py xm2vts dumplist --protocol=lp1 --classes=client --groups=dev --purposes=enrol --self-test)

bob_python_add_dependent_test(db-xm2vts-checkfiles python-db-xm2vts-create
  bob_dbmanage.py xm2vts checkfiles --self-test)

# FIR database driver tests.
bob_python_add_unittest(db lib/test/fir.py main)

bob_python_add_dependent_test(db-fir-dumplist python-db-fir-create
  bob_dbmanage.py fir dumplist --self-test)

bob_python_add_dependent_test(db-fir-dumplist2 python-db-fir-create
  bob_dbmanage.py fir dumplist --protocol=noir --groups=dev --purposes=enrol --self-test)

bob_python_add_dependent_test(db-fir-checkfiles python-db-fir-create
  bob_dbmanage.py fir checkfiles --self-test)

# AT&T database driver tests.
bob_python_add_unittest(db lib/test/atnt.py main)
bob_python_add_test(db-atnt-dumplist bob_dbmanage.py atnt dumplist --self-test)
bob_python_add_test(db-atnt-checkfiles bob_dbmanage.py atnt checkfiles -d "." --self-test)

# GBU database driver tests.
bob_python_add_unittest(db lib/test/gbu.py main)
bob_python_add_test(db-gbu-dumplist bob_dbmanage.py gbu dumplist --self-test)
bob_python_add_test(db-gbu-checkfiles bob_dbmanage.py gbu checkfiles -d "." --self-test)
bob_python_add_test(db-gbu-create-eye-files bob_dbmanage.py gbu create-eye-files -d "." --self-test)

# Our examples (run and install)
bob_python_example(db iris_lda.py lib/example/iris_lda.py)
bob_python_add_test(db iris_lda.py --self-test)

bob_python_example(db iris_rprop.py lib/example/iris_rprop.py)
bob_python_add_test(db iris_rprop.py --self-test)

bob_python_example(db iris_backprop.py lib/example/iris_backprop.py)
bob_python_add_test(db iris_backprop.py --self-test)
