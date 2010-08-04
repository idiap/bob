# Andre Anjos <andre.anjos@idiap.ch>
# 4/August/2010

# These are a few handy macros for Torch compilation.

# Builds and installs a shared library with dependencies
macro(torch_shlib libname sources dependencies externals installdir)
  add_library(${libname} SHARED ${sources})
  add_dependencies(${libname} ${dependencies})
  target_link_libraries(${libname} ${externals})
  install(TARGETS ${libname} LIBRARY DESTINATION ${installdir})
endmacro(torch_shlib libname sources dependencies)

# Builds and installs a shared library with dependencies
macro(torch_archive libname sources dependencies externals installdir)
  add_library(${libname}-static STATIC ${src})
  foreach(dep ${dependencies})
    add_dependencies(${libname}-static ${dependencies}-static)
  endforeach(dep ${dependencies})
  target_link_libraries(${libname} ${externals})
  set_target_properties(${libname}-static PROPERTIES OUTPUT_NAME "${libname}")
  set_target_properties(${libname}-static PROPERTIES PREFIX "lib")
  install(TARGETS ${libname}-static ARCHIVE DESTINATION ${installdir})
endmacro(torch_archive sources dependencies)
