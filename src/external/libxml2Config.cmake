# Finds and configures libxml2 if it exists on the system. 
# Laurent El Shafey - 25.november.2010

# This includes in OSX, the MacPorts installation path
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} /opt/local/lib)

include(FindLibXml2)

set(libxml2_INCLUDE_DIRS ${LIBXML2_INCLUDE_DIR} CACHE INTERNAL "incdirs")
include_directories(SYSTEM "${LIBXML2_INCLUDE_DIR}")
set(libxml2_LIBRARY_DIRS ${LIBXML2_LIBRARY_DIRS} CACHE INTERNAL "libdirs")
