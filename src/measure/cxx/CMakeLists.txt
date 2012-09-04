project(bob_measure)

# This defines the dependencies of this package
set(bob_deps "bob_core")
set(shared "${bob_deps}")

# This defines the list of source files inside this package.
set(src
    "error.cc"
    )

bob_add_library(bob_${PROJECT_NAME} "${src}")
target_link_libraries(bob_${PROJECT_NAME} ${shared})

# Pkg-Config generator
bob_pkgconfig(${PROJECT_NAME} "${bob_deps}")
