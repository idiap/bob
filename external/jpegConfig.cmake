# Finds and configures jpeglib if it exists on the system. 
# Andre Anjos - 07.august.2010

# This includes in OSX, the MacPorts installation path
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} /opt/local/lib)

include(FindJPEG)

