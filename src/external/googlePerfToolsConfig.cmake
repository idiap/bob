# Finds and configures google-perftools if it exists on the system. 
# Andre Anjos - 07.september.2010

# We start by defining googlePerfTools_FOUND to false
set(googlePerfTools_FOUND "NO" CACHE INTERNAL "package")

# Here we do some variable cleanup and adjustments
find_path(googlePerfTools_INCLUDE NAMES google/profiler.h)
set(googlePerfTools_INCLUDE_DIRS ${googlePerfTools_INCLUDE})
include_directories(SYSTEM ${googlePerfTools_INCLUDE_DIRS})

find_library(googlePerfTools_LIBRARY NAMES profiler)
set(googlePerfTools_LIBRARIES ${googlePerfTools_LIBRARY} CACHE INTERNAL "libraries")

if(googlePerfTools_INCLUDE AND googlePerfTools_LIBRARY)
  set(googlePerfTools_FOUND "YES" CACHE INTERNAL "package")
else(googlePerfTools_INCLUDE AND googlePerfTools_LIBRARY)
  # This will say why we have got to that conclusion
  set(googlePerfTools_FOUND "NO")
  if (NOT googlePerfTools_INCLUDE)
    message("--   google-perftools <google/profiler.h> not found!")
  endif (NOT googlePerfTools_INCLUDE)
  if (NOT googlePerfTools_LIBRARY)
    message("--   google-perftools libprofiler.so not found!")
  endif (NOT googlePerfTools_LIBRARY)
endif(googlePerfTools_INCLUDE AND googlePerfTools_LIBRARY)
