# Configures Blitz++
# Andre Anjos <andre.anjos@idiap.ch>
# Sat  1 Sep 20:36:15 2012 CEST

execute_process(COMMAND ${PKG_CONFIG_EXECUTABLE} blitz --silence-errors --modversion OUTPUT_VARIABLE PKG_CONFIG_blitz_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)

if(PKG_CONFIG_blitz_VERSION)
  #use pkg-config to find blitz
  pkg_check_modules(Blitz REQUIRED blitz)

  # Makes sure we have at least 0.10
  if(Blitz_VERSION VERSION_LESS "0.10")
    message(FATAL_ERROR "error: Bob depends on Blitz++ version 0.10 or superior.")
  endif()

  # Resolve Blitz library to a precise path
  set(Blitz_INCLUDE_DIR ${Blitz_INCLUDE_DIRS})
  set(Blitz_RESOLVED_LIBRARY "")
  resolve_library(${Blitz_LIBRARIES} "${Blitz_LIBRARY_DIRS}" Blitz_RESOLVED_LIBRARY)
  set(Blitz_RESOLVED_LIBRARY ${Blitz_RESOLVED_LIBRARY} CACHE INTERNAL "Resolved Blitz library")
else()
  message(FATAL_ERROR "error: Bob depends on Blitz++ version 0.10 or superior, but I could not find its pkg-config file. If you have Blitz++ installed, make sure blitz.pc is on your PKG_CONFIG_PATH or that CMAKE_PREFIX_PATH is pointing to the right place.")
endif()

mark_as_advanced(Blitz_INCLUDE_DIR Blitz_LIBRARY)
