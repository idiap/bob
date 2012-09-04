# Tries to find a local version of Boost installed
# Andre Anjos - 02.july.2010

include(FindBoost)

# Compiles against mt versions
set(Boost_USE_MULTITHREADED ON)

# Determine here the components you need so the system can verify
find_package(Boost 1.40.0 REQUIRED 
  COMPONENTS 
    python 
    unit_test_framework
    iostreams 
    serialization
    thread
    filesystem
    date_time
    program_options
    system
    regex
  )

# Renaming so all works automagically
set(boost_INCLUDE_DIRS ${Boost_INCLUDE_DIRS} CACHE INTERNAL "incdirs")
include_directories(SYSTEM ${boost_INCLUDE_DIRS})
set(boost_LIBRARY_DIRS ${Boost_LIBRARY_DIRS} CACHE INTERNAL "libdirs")
