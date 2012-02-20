project(python)

cmake_minimum_required(VERSION 2.6)

# Installs python top-level wrapper
if (PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND AND Boost_FOUND)

  # Dummy target for the python root package
  add_custom_target(pybob_root ALL)

  # Tags version and platform and add this to the core of the distribution
  bob_python_configure("lib/build.py.in" "build.py")
  bob_python_configure_program("bin/shell.py.in" "shell.py")

  # Adds specific python sources
  set(pysrc 
    "lib/__init__.py"
    "lib/helper.py"
    "lib/script/__init__.py"
    "lib/script/config.py"
    )
  bob_python_module(root "${pysrc}")
  
  # Installs and test our scripts
  bob_python_script(root "bob_config.py" "lib/script/config.py" "main")
  bob_python_add_test(root bob_config.py)

  # Shell.py for user support
  #bob_python_add_test(root shell.py)

  add_subdirectory(core)
  add_subdirectory(io)
  add_subdirectory(math)
  add_subdirectory(measure)
  add_subdirectory(sp)
  add_subdirectory(ip)
  add_subdirectory(db)
  add_subdirectory(machine)
  add_subdirectory(trainer)
  add_subdirectory(face_verification)

  if (NOT QT4_FOUND)
    message(STATUS "Qt4 NOT FOUND, supressing visioner-python from build")
  else (NOT QT4_FOUND)
    add_subdirectory(visioner)
    add_subdirectory(daq)
  endif (NOT QT4_FOUND)

endif (PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND AND Boost_FOUND)
