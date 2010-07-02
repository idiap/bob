# Tries to find a local version of Boost installed
# Andre Anjos - 02.july.2010

include(FindBoost)

if (${TORCH_LINKAGE} STREQUAL static)
  set(Boost_USE_STATIC_LIBS ON)
endif (${TORCH_LINKAGE} STREQUAL static)

# Compiles against mt versions
set(Boost_USE_MULTITHREADED ON)

# Determine here the components you need so the system can verify
find_package(Boost COMPONENTS python)
