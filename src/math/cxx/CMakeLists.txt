PROJECT(bob_math)

set(suffixes sse2)
find_library(lapack_LIBRARY NAMES lapack PATH_SUFFIXES ${suffixes}) 
find_library(blas_LIBRARY NAMES cblas blas PATH_SUFFIXES ${suffixes})
set(lapack_LIBRARIES ${lapack_LIBRARY};${blas_LIBRARY})
list(REMOVE_DUPLICATES lapack_LIBRARIES)
set(lapack_LIBRARIES ${lapack_LIBRARIES} CACHE INTERNAL "libraries")

if(blas_LIBRARY AND lapack_LIBRARY)
  find_package_message(LAPACK "LAPACK/BLAS libraries: ${lapack_LIBRARIES}" "[${lapack_LIBRARIES}]")
else()
  set(lapack_FOUND "NO")
  if (NOT lapack_LIBRARY)
    message(FATAL_ERROR "LAPACK library not found!")
  endif()
  if (NOT blas_LIBRARY)
    message(FATAL_ERROR "BLAS library not found!")
  endif()
endif()

# This defines the dependencies of this package
set(bob_deps "bob_core")
set(shared "${bob_deps};${lapack_LIBRARIES}")
set(incdir ${cxx_incdir})

set(src
  "Exception.cc"
  "norminv.cc"
  "log.cc"
  "eig.cc"
  "linsolve.cc"
  "lu.cc"
  "det.cc"
  "inv.cc"
  "pinv.cc"
  "sqrtm.cc"
  "svd.cc"
  "LPInteriorPoint.cc"
  "pavx.cc"
)

# Define the library, compilation and linkage options
bob_sort_headers(incdir)
foreach(inc ${incdir})
  include_directories(SYSTEM ${inc})
endforeach()
bob_add_library(${PROJECT_NAME} "${src}")
target_link_libraries(${PROJECT_NAME} ${shared})

# Defines tests for this package
bob_add_test(${PROJECT_NAME} eig test/eig.cc)
bob_add_test(${PROJECT_NAME} gradient test/gradient.cc)
bob_add_test(${PROJECT_NAME} linear test/linear.cc)
bob_add_test(${PROJECT_NAME} linsolve test/linsolve.cc)
bob_add_test(${PROJECT_NAME} lu_det_inv test/lu_det_inv.cc)
bob_add_test(${PROJECT_NAME} norm test/norm.cc)
bob_add_test(${PROJECT_NAME} norminv test/norminv.cc)
bob_add_test(${PROJECT_NAME} pinv test/pinv.cc)
bob_add_test(${PROJECT_NAME} sqrtm test/sqrtm.cc)
bob_add_test(${PROJECT_NAME} svd test/svd.cc)
bob_add_test(${PROJECT_NAME} LPInteriorPoint test/LPInteriorPoint.cc)

# Pkg-Config generator
bob_pkgconfig(${PROJECT_NAME} "${bob_deps}")
