# Tries to find a local version of Qt4 installed
# Andre Anjos - 21.july.2011

find_package (Qt4 COMPONENTS QtCore QtGui)

if(QT4_FOUND)
  add_definitions("-D HAVE_QT4=1")
endif(QT4_FOUND)
