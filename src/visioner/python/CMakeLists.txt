project(bob_visioner_py)

# This defines the dependencies of this package
set(bob_deps "bob_visioner;bob_python")
set(shared "${bob_deps}")
set(incdir "${python_INCLUDE_DIRS};${QT_INCLUDES}")
include (${QT_USE_FILE})

# This defines the list of source files inside this package.
set(src
   "version.cc"
   "localize.cc"
   "train.cc"
   "main.cc"
   )
 
# Define the library, compilation and linkage options
list(REMOVE_DUPLICATES incdir)
foreach(inc ${incdir})
  include_directories(SYSTEM ${inc})
endforeach()
bob_add_library(${PROJECT_NAME} "${src}")
target_link_libraries(${PROJECT_NAME} ${shared})

# Pkg-Config generator
bob_pkgconfig(${PROJECT_NAME} "${bob_deps}")
