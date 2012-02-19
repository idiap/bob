# New targets for building/downloading and cloning databases

add_custom_target(install-database-download
  COMMAND ${CMAKE_INSTALL_PREFIX}/bin/dbmanage.py all download --verbose
  COMMENT "Downloading bob databases version '${BOB_DATABASE_VERSION}' from '${BOB_DATABASE_URL}'" VERBATIM
)

add_custom_target(install-database-create
  COMMAND ${CMAKE_INSTALL_PREFIX}/bin/dbmanage.py all create --recreate --verbose
  COMMENT "Creating bob databases from scratch" VERBATIM
)
