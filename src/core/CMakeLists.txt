#
# Add the source files to the list of files to be added to the library
#
SET(	core_SRC
		"${CMAKE_CURRENT_SOURCE_DIR}/general.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/VariableCollector.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/Object.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/Parameters.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/File.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/spCore.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/spCoreChain.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/ipCore.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/apCore.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/DataSet.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/MemoryDataSet.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/DiskDataSet.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/ImageScanDataSet.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/Trainer.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/Machine.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/Tensor.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/TensorFile.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/TensorList.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/TensorListTarget.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/MTimer.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/CmdOption.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/FileListCmdOption.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/FileList.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/CmdLine.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/CmdFile.cc"
	CACHE INTERNAL "sources")

SET(	core_INCLUDE
		${CMAKE_CURRENT_SOURCE_DIR}
	CACHE INTERNAL "includes")
