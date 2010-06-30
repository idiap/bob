# This just does some iterative work common to all subpackages of torch

set(${THIS}_INCLUDE ${THIS} CACHE INTERNAL "includes")

# adds up stuff from subdirectories
if(DEFINED ${THIS}_SUBDIRS)
  foreach(subdir ${${THIS}_SUBDIRS})
    # Add the includes from the subdirectories
    set(${THIS}_INCLUDE	"${${THIS}_INCLUDE};${THIS}/${subdir}" CACHE INTERNAL "includes")

    add_subdirectory(${subdir}) #this changes the value of ${THIS}
    # Add the sources from the subdirectories
    foreach(subsrc ${${subdir}_SRC})
      set(${THIS}_SRC	"${subsrc};${${THIS}_SRC}" CACHE INTERNAL "sources")
    endforeach(subsrc ${${subdir}_SRC})

  endforeach(subdir)
endif(DEFINED ${THIS}_SUBDIRS)

set(tmp)
foreach(src ${${THIS}_SRC})
  set(tmp ${tmp};${THIS}/${src})
endforeach(src ${THIS}_SRC)
set(${THIS}_SRC ${tmp} CACHE INTERNAL "sources")
