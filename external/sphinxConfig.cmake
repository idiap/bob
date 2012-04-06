# This modules defines
#  SPHINX_EXECUTABLE
#  SPHINX_FOUND

find_program(SPHINX_EXECUTABLE NAMES sphinx-build-${PYTHON_VERSION} sphinx-build
  HINTS $ENV{SPHINX_DIR} PATH_SUFFIXES bin
  DOC "Sphinx documentation generator"
  )

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(Sphinx DEFAULT_MSG SPHINX_EXECUTABLE)
mark_as_advanced(SPHINX_EXECUTABLE)
