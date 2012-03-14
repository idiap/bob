# New targets for building/downloading and cloning databases

add_custom_target(db-download
  COMMAND ${CMAKE_SOURCE_DIR}/bin/dbdownload.py --verbose --server=${BOB_DATABASE_URL} --version=${BOB_DATABASE_VERSION} --destination=${CMAKE_SOURCE_DIR}/databases COMMENT "Downloading/refreshing databases from '${BOB_DATABASE_URL}/${BOB_DATABASE_VERSION}' to '${CMAKE_SOURCE_DIR}/databases'")

add_custom_target(db-try-download
  COMMAND ${CMAKE_SOURCE_DIR}/bin/dbdownload.py --verbose --server=${BOB_DATABASE_URL} --version=${BOB_DATABASE_VERSION} --destination=${CMAKE_SOURCE_DIR}/databases --try COMMENT "Trying to download/refresh databases from '${BOB_DATABASE_URL}/${BOB_DATABASE_VERSION}' to '${CMAKE_SOURCE_DIR}/databases'")

add_custom_target(db-refresh DEPENDS db-try-download
  COMMAND ${CMAKE_BINARY_DIR}/bin/dbmanage.py all copyfrom --verbose ${CMAKE_SOURCE_DIR}/databases COMMENT "Refreshing databases installed at '${CMAKE_BINARY_DIR}' using downloads at '${CMAKE_SOURCE_DIR}/databases'")

add_custom_target(db-refresh-install DEPENDS db-try-download
  COMMAND ${CMAKE_INSTALL_PREFIX}/bin/dbmanage.py all copyfrom --verbose ${CMAKE_SOURCE_DIR}/databases COMMENT "Refreshing databases installed at '${CMAKE_INSTALL_PREFIX}' using downloads at '${CMAKE_SOURCE_DIR}/databases'")

add_custom_target(db-create
  COMMAND ${CMAKE_BINARY_DIR}/bin/dbmanage.py all create --recreate --verbose ${CMAKE_SOURCE_DIR}/databases COMMENT "Creating databases at '${CMAKE_BINARY_DIR}' from scratch")
