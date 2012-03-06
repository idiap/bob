project(db-python)
cmake_minimum_required(VERSION 2.6)

bob_python_package_bindings(db "" "")

# A macro for controlling creation, downloading and dependencies automatically
# All tests should be dependent on the python-db-<dbname>-instantiation 
# test (pointer) target after this macro is called for a certain database.
macro(bob_build_db dbname)
  
  # If we are supposed to create the database, create it on the install
  # directory at an indicated global position where the databases can be used
  # for downloading through our web server.
  if (NOT BOB_DATABASE_URL)
    bob_python_add_test(db-${dbname}-instantiation dbmanage.py ${dbname} create --recreate)
  else ()
    bob_python_add_test(db-${dbname}-instantiation dbmanage.py ${dbname} download --url=${BOB_DATABASE_URL} --version=${BOB_DATABASE_VERSION})
  endif ()

endmacro(bob_build_db dbname)

# Installs our script driver
bob_python_script(db "dbmanage.py" "lib/script/dbmanage.py" "main")

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
  python-db-replay-instantiation 
  dbmanage.py replay dumplist --self-test)

bob_python_add_dependent_test(db-replay-dumplist2 
  python-db-replay-instantiation 
  dbmanage.py replay dumplist --class=attack --group=devel --support=hand --protocol=highdef --self-test)

bob_python_add_unittest(db lib/test/replay.py)
set_property(TEST python-db-replay-main APPEND PROPERTY DEPENDS 
  python-db-replay-instantiation)

#bob_python_example(db "replay_example.py" lib/example/replay_example.py main)
#bob_python_add_test(db replay_example.py)

# BANCA database driver tests.
bob_python_add_dependent_test(db-banca-dumplist 
  python-db-banca-instantiation
  dbmanage.py banca dumplist --self-test)

bob_python_add_dependent_test(db-banca-dumplist2
  python-db-banca-instantiation
  dbmanage.py banca dumplist --protocol=P --classes=client --groups=g1 --purposes=enrol --self-test)

# BANCA_SMALL database driver tests.
bob_python_add_dependent_test(db-banca_small-dumplist
  python-db-banca_small-instantiation 
  dbmanage.py banca_small dumplist --self-test)

bob_python_add_dependent_test(db-banca_small-dumplist2 
  python-db-banca_small-instantiation
  dbmanage.py banca_small dumplist --protocol=P --classes=client --groups=g1 --purposes=enrol --self-test)

# Biosecure database driver tests.
bob_python_add_dependent_test(db-biosecure-dumplist 
  python-db-biosecure-instantiation 
  dbmanage.py biosecure dumplist --self-test)

bob_python_add_dependent_test(db-biosecure-dumplist2
  python-db-biosecure-instantiation 
  dbmanage.py biosecure dumplist --protocol=ca0 --classes=client --groups=dev --purposes=enrol --self-test)

# Faceverif_fl database (uses file lists).
bob_python_add_unittest(db lib/test/faceverif_fl.py)

# LFW database driver tests.
bob_python_add_dependent_test(db-lfw-dumplist
  python-db-lfw-instantiation 
  dbmanage.py lfw dumplist --self-test)

bob_python_add_dependent_test(db-lfw-dumplist2
  python-db-lfw-instantiation 
  dbmanage.py lfw dumplist --view=view1 --subset=fold1 --classes=matched --self-test)

# MOBIO database driver tests.
bob_python_add_dependent_test(db-mobio-dumplist 
  python-db-mobio-instantiation 
  dbmanage.py mobio dumplist --self-test)

bob_python_add_dependent_test(db-mobio-dumplist2 
  python-db-mobio-instantiation 
  dbmanage.py mobio dumplist --protocol=male --classes=client --groups=dev --purposes=enrol --self-test)

# Multi-PIE database driver tests.
bob_python_add_dependent_test(db-multipie-dumplist 
  python-db-multipie-instantiation 
  dbmanage.py multipie dumplist --self-test)

bob_python_add_dependent_test(db-multipie-dumplist2 
  python-db-multipie-instantiation 
  dbmanage.py multipie dumplist --protocol=M --classes=client --groups=dev --purposes=enrol --self-test)

# Nuaa database driver tests.
bob_python_add_unittest(db lib/test/nuaa.py main)

# CASIA-FASD database driver tests.
bob_python_add_unittest(db lib/test/casia_fasd.py main)

# SCface database driver tests.
bob_python_add_dependent_test(db-scface-dumplist 
  python-db-scface-instantiation 
  dbmanage.py scface dumplist --self-test)

bob_python_add_dependent_test(db-scface-dumplist2 
  python-db-scface-instantiation 
  dbmanage.py scface dumplist --protocol=combined --classes=client --groups=dev --purposes=enrol --self-test)

# XM2VTS database driver tests.
bob_python_add_dependent_test(db-xm2vts-dumplist 
  python-db-xm2vts-instantiation 
  dbmanage.py xm2vts dumplist --self-test)

bob_python_add_dependent_test(db-xm2vts-dumplist2 
  python-db-xm2vts-instantiation 
  dbmanage.py xm2vts dumplist --protocol=lp1 --classes=client --groups=dev --purposes=enrol --self-test)

# Our examples (run and install)
bob_python_example(db iris_lda.py lib/example/iris_lda.py)
bob_python_add_test(db iris_lda.py --self-test)

bob_python_example(db iris_rprop.py lib/example/iris_rprop.py)
bob_python_add_test(db iris_rprop.py --self-test)

bob_python_example(db iris_backprop.py lib/example/iris_backprop.py)
bob_python_add_test(db iris_backprop.py --self-test)
