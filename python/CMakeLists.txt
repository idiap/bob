project(python)

cmake_minimum_required(VERSION 2.6)

# Installs python top-level wrapper
if (PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND AND Boost_FOUND)

  # Dummy target for the python root package
  add_custom_target(pybob_root ALL)

  # Tags version and platform and add this to the core of the distribution
  bob_python_tag_build("lib/build.py.in" "build.py")

  # Adds specific python sources
  set(pysrc 
    "lib/__init__.py"
    "lib/helper.py"
    )
  bob_python_module(root "${pysrc}")
  
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
