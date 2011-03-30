# Finds and configures hdf5 libraries if they exists on the system. 
# Andre Anjos - 30.march.2011
include(FindHDF5)
add_definitions("-DHAVE_HDF5=1")
