project(bob_lbfgs)

bob_add_library(${PROJECT_NAME} "lbfgs.c")

# Pkg-Config generator
bob_pkgconfig(${PROJECT_NAME} "")
