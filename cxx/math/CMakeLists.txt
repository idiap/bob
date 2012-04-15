PROJECT(math-cxx)
cmake_minimum_required(VERSION 2.6)

# This defines the dependencies of this package
set(deps "core") #other bob subprojects
set(shared "${lapack_LIBRARIES}")

# This defines the list of source files inside this package.
set(src
   "src/Exception.cc"
   "src/norminv.cc"
   "src/log.cc"
)

set(src_blas
   ""
)

set(src_lapack
  "src/eig.cc"
  "src/linsolve.cc"
  "src/lu.cc"
  "src/det.cc"
  "src/inv.cc"
  "src/sqrtm.cc"
  "src/svd.cc"
)

set(src_optimization
  "src/interiorpointLP.cc"
)

set(src 
   "${src}" 
   "${src_blas}" 
   "${src_lapack}"
   "${src_optimization}")

bob_library(math "${src}" "${deps}" "${shared}")

# Defines tests for this package
bob_test(math eig test/eig.cc)
bob_test(math gradient test/gradient.cc)
bob_test(math linear test/linear.cc)
bob_test(math linsolve test/linsolve.cc)
bob_test(math lu_det_inv test/lu_det_inv.cc)
bob_test(math norminv test/norminv.cc)
bob_test(math sqrtm test/sqrtm.cc)
bob_test(math svd test/svd.cc)
bob_test(math interiorpointLP test/interiorpointLP.cc)
