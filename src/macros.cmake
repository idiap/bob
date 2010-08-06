# Andre Anjos <andre.anjos@idiap.ch>
# 4/August/2010

# These are a few handy macros for Torch compilation.

# Builds and installs a shared library with dependencies
macro(torch_shlib libname sources dependencies externals installdir)
  add_library(${libname} SHARED ${sources})
  if(NOT ("${dependencies}" STREQUAL ""))
    foreach(dep ${dependencies})
      target_link_libraries(${libname} torch_${dep})
    endforeach(dep "${dependencies}")
  endif(NOT ("${dependencies}" STREQUAL ""))
  if(NOT ("${externals}" STREQUAL ""))
    foreach(ext ${externals})
      target_link_libraries(${libname} ${ext})
    endforeach(ext "${externals}")
  endif(NOT ("${externals}" STREQUAL ""))
  install(TARGETS ${libname} LIBRARY DESTINATION ${installdir})
endmacro(torch_shlib libname sources dependencies)

# Builds and installs a static library with dependencies
macro(torch_archive libname sources dependencies installdir)
  add_library(${libname}-static STATIC ${sources})
  if(NOT ("${dependencies}" STREQUAL ""))
    foreach(dep ${dependencies})
      target_link_libraries(${libname}-static torch_${dep}-static)
    endforeach(dep "${dependencies}")
  endif(NOT ("${dependencies}" STREQUAL ""))
  set_target_properties(${libname}-static PROPERTIES OUTPUT_NAME ${libname})
  set_target_properties(${libname}-static PROPERTIES PREFIX "lib")
  install(TARGETS ${libname}-static ARCHIVE DESTINATION ${installdir})
endmacro(torch_archive sources dependencies)

# Builds libraries for a subproject and installs headers
macro(torch_package package src deps shared)
  set(libname torch_${package})
  set(libdir ${INSTALL_DIR}/lib)
  set(incdir ${INCLUDE_DIR}/torch/${package})
  torch_shlib(${libname} "${src}" "${deps}" "${shared}" ${libdir})
  torch_archive(${libname} "${src}" "${deps}" ${libdir})
  file(COPY ./ DESTINATION ${incdir} FILES_MATCHING PATTERN "*.h")
endmacro(torch_package)
