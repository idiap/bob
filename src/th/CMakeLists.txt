#
# Add the source files to the list of files to be added to the library
#
project(th)
set(THIS th)

set(${THIS}_SRC
		"THBlas.c"
		"THGeneral.c"
		"THLogAdd.c"
		"THRandom.c"
		"THStorage.c"
		"THTensor.c"
		"THTensorMath.c")

add_library(torch_${THIS} SHARED ${${THIS}_SRC})
add_library(torch_${THIS}-static STATIC ${${THIS}_SRC})
set_target_properties(torch_${THIS}-static PROPERTIES OUTPUT_NAME "torch_${THIS}")
set_target_properties(torch_${THIS}-static PROPERTIES PREFIX "lib")
install(TARGETS torch_${THIS} LIBRARY DESTINATION ${INSTALL_DIR}/lib)
install(TARGETS torch_${THIS}-static ARCHIVE DESTINATION ${INSTALL_DIR}/lib)
file(COPY ./ DESTINATION ${INCLUDE_DIR}/torch/${THIS} FILES_MATCHING PATTERN "*.h")
