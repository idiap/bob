project(visioner)

add_subdirectory(cxx)
add_subdirectory(programs)
add_subdirectory(python)
 
set(ENABLED_PACKAGES "${ENABLED_PACKAGES}" PARENT_SCOPE)
